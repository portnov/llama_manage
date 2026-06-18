#!/usr/bin/env python3
"""llama.cpp server management CLI (docker-style)."""

import argparse
import json
import os
import sys
import threading
import time
from fnmatch import fnmatch

import requests


def get_url(args) -> str:
    url = getattr(args, "url", None) or os.environ.get("LLAMA_URL")
    if not url:
        print("Error: no server URL. Set --url or $LLAMA_URL.", file=sys.stderr)
        sys.exit(1)
    return url.rstrip("/") + "/"

def format_number(size, unit : str = '', binary : bool = True):
    if binary:
        divisor = 1024.0
        suffixes = ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei']
    else:
        divisor = 1000.0
        suffixes = ['', 'K', 'M', 'B', 'T', 'P', 'E']
    suffixes = [s + unit for s in suffixes]
    # Support negative numbers and zero
    for suffix in suffixes[:-1]:
        if abs(size) < divisor:
            return f"{size:.2f} {suffix}"
        size /= divisor
    return f"{size:.2f} {suffixes[-1]}"

def get_path(args, model):
    path = model.get("path", None)
    if path is not None:
        return path
    model_args = model.get("status", {}).get("args", [])
    found_path_key = False
    found_hf_key = False
    for arg in model_args:
        if arg == "--model":
            found_path_key = True
        elif arg == "--hf-repo":
            found_hf_key = True
        elif found_path_key:
            if args.full_path:
                return arg
            else:
                return os.path.basename(arg)
        elif found_hf_key:
            return arg
    return "<N/A>"

def show_tags(model):
    tags = model.get("tags", [])
    if not tags:
        return "-"
    return ", ".join(tags)

LS_COLUMNS = ["ID", "TAGS", "PATH", "STATUS", "CONTEXT", "PARAMS", "SIZE"]

def get_ctx_size(model):
    if "meta" not in model:
        return "-"
    n_ctx = model["meta"]["n_ctx"]
    n_ctx_train = model["meta"]["n_ctx_train"]
    return format_number(n_ctx, binary=False) + " / " + format_number(n_ctx_train, binary=False)

def get_n_params(model):
    if "meta" not in model:
        return "-"
    return format_number(model["meta"]["n_params"], binary=False)

def get_size(model):
    if "meta" not in model:
        return "-"
    return format_number(model["meta"]["size"], unit='B', binary=True)

def cmd_ls(args):
    url = get_url(args)
    resp = requests.get(url + "models")
    resp.raise_for_status()
    data = resp.json()["data"]

    if not args.all and not (args.id or args.tag):
        data = [m for m in data if m["status"]["value"] == "loaded"]

    rows = []
    for m in data:
        if args.tag:
            model_tags = set(m.get("tags", []))
            arg_tags = set(args.tag)
            if not arg_tags.issubset(model_tags):
                continue
        if args.id:
            if not fnmatch(m["id"], args.id):
                continue

        rows.append({
            "ID": m["id"],
            "TAGS": show_tags(m),
            "PATH": get_path(args, m),
            "STATUS": m["status"]["value"],
            "CONTEXT": get_ctx_size(m),
            "PARAMS" : get_n_params(m),
            "SIZE": get_size(m)
        })

    if not rows:
        print("No models found.")
        return

    print_table(rows, LS_COLUMNS)


def cmd_load(args):
    url = get_url(args)
    resp = requests.post(url + "models/load", json={"model": args.id})
    resp.raise_for_status()
    if resp.json().get("success"):
        print(f"Model {args.id} loading started.")
    else:
        print(f"Failed to load model {args.id}.", file=sys.stderr)
        sys.exit(1)


def cmd_unload(args):
    url = get_url(args)
    resp = requests.post(url + "models/unload", json={"model": args.id})
    resp.raise_for_status()
    if resp.json().get("success"):
        print(f"Model {args.id} unloading started.")
    else:
        print(f"Failed to unload model {args.id}.", file=sys.stderr)
        sys.exit(1)


def cmd_rm(args):
    url = get_url(args)
    model_id = args.id

    if not args.force:
        if not input(f"Delete model {model_id}? [y/N] ").strip().lower() == "y":
            print("Aborted.")
            return

    resp = requests.delete(url + "models", params={"model": model_id})
    resp.raise_for_status()
    if resp.json().get("success"):
        print(f"Model {model_id} deleted.")
    else:
        print(f"Failed to delete model {model_id}.", file=sys.stderr)
        sys.exit(1)

def print_table(rows, columns):
    """Print a list of dicts as a formatted table.

    rows: list of dict  — keys must match column names.
    columns: list of str — column names in display order.
    """
    widths = {
        col: max(len(col), max(len(str(row.get(col, ""))) for row in rows))
        for col in columns
    }
    header = "  ".join(f"{col:<{widths[col]}}" for col in columns)
    sep = "  ".join("-" * widths[col] for col in columns)
    print(header)
    print(sep)
    for row in rows:
        print("  ".join(f"{str(row.get(col, '')):<{widths[col]}}" for col in columns))


def check_loaded(args, model_id):
    url = get_url(args)
    resp = requests.get(url + "models")
    resp.raise_for_status()
    data = resp.json()["data"]
    for m in data:
        if m["id"] == model_id and m["status"]["value"] == "loaded":
            return True
    print(f"Model {model_id} is not loaded", file=sys.stderr)
    sys.exit(1)


def _progress_bar(pct, width=30):
    filled = int(width * pct / 100)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}]"


def _print_progress(model_id, done, total, rate):
    pct = (done / total * 100) if total else 0
    bar = _progress_bar(pct)
    line = f"\r{model_id}  {bar} {pct:6.1f}%  {format_number(done, 'B', True):>10} / {format_number(total, 'B', True):>10}"
    if rate is not None:
        line += f"  {format_number(rate, 'B/s', True)}"
    sys.stdout.write(line)
    sys.stdout.flush()


def cmd_pull(args):
    url = get_url(args)
    model_id = args.id

    # Subscribe to SSE first, then trigger download
    sse_resp = requests.get(url + "models/sse", stream=True)
    sse_resp.raise_for_status()

    # Start the download in a separate thread so we don't block SSE
    def trigger_download():
        time.sleep(0.5)
        resp = requests.post(url + "models", json={"model": model_id})
        resp.raise_for_status()
    threading.Thread(target=trigger_download, daemon=True).start()

    total_done = 0
    total_size = 0
    last_done = 0
    last_time = None

    try:
        for line in sse_resp.iter_lines(decode_unicode=True):
            line = line.strip()
            if not line or not line.startswith("data: "):
                continue
            try:
                event = json.loads(line[6:])
            except json.JSONDecodeError:
                continue

            if event.get("model") != model_id:
                continue

            evt = event.get("event")
            data = event.get("data", {})

            if evt == "download_progress":
                # data is {"progress": {url: {done, total}, ...}}
                progress = data.get("progress", data)
                total_done = 0
                total_size = 0
                for info in progress.values():
                    total_done += info.get("done", 0)
                    total_size += info.get("total", 0)

                now = time.time()
                if last_time is not None:
                    dt = now - last_time
                    if dt > 0:
                        rate = (total_done - last_done) / dt
                    else:
                        rate = None
                else:
                    rate = None
                last_done = total_done
                last_time = now

                _print_progress(model_id, total_done, total_size, rate)

            elif evt == "download_finished":
                print()  # newline after progress
                print(f"Model {model_id} downloaded successfully.")
                return

            elif evt == "download_failed":
                print()
                status = data.get("status", "unknown")
                print(f"Download failed for {model_id}: {status}", file=sys.stderr)
                sys.exit(1)
    finally:
        sse_resp.close()

    print("SSE stream ended unexpectedly.", file=sys.stderr)
    sys.exit(1)

PS_COLUMNS = ["ID", "TASK", "CONTEXT", "PROC", "DECODED"]

def get_decoded(slot):
    if "next_token" not in slot:
        return "-"
    tokens = slot["next_token"]
    if isinstance(tokens, list):
        n_decoded = sum([t.get("n_decoded",0) for t in tokens])
    else:
        n_decoded = tokens.get("n_decoded", 0)
    return format_number(n_decoded)

def cmd_ps(args):
    url = get_url(args)
    if args.id is None:
        params = None
    else:
        check_loaded(args, args.id)
        params = {"model": args.id}
    resp = requests.get(url + "slots", params=params)
    resp.raise_for_status()
    slots = resp.json()

    if not args.all:
        slots = [s for s in slots if s["is_processing"]]

    if not slots:
        print("No slots found.")
        return

    rows = []
    for s in slots:
        rows.append({
            "ID": s["id"],
            "TASK": s["id_task"],
            "CONTEXT": s["n_ctx"],
            "PROC": "Y" if s["is_processing"] else "N",
            "DECODED": get_decoded(s),
        })

    print_table(rows, PS_COLUMNS)


def main():
    parser = argparse.ArgumentParser(
        description="llama.cpp server management CLI",
    )
    parser.add_argument("--url", help="Server URL (e.g. http://localhost:8080/)")

    sub = parser.add_subparsers(dest="command")

    # ls
    ls_parser = sub.add_parser("ls", help="List models")
    ls_parser.add_argument("-a", "--all", action="store_true",
                           help="Show all models, not just loaded ones")
    ls_parser.add_argument("-t", "--tag", nargs='+', help="Show only models with these tags")
    ls_parser.add_argument("--full-path", action="store_true",
                           help="Show full paths to model files")
    ls_parser.add_argument("id", nargs='?', help="Model ID (glob mask)")
    ls_parser.set_defaults(func=cmd_ls)

    # load
    load_parser = sub.add_parser("load", help="Load a model")
    load_parser.add_argument("id", help="Model ID to load")
    load_parser.set_defaults(func=cmd_load)

    # unload
    unload_parser = sub.add_parser("unload", help="Unload a model")
    unload_parser.add_argument("id", help="Model ID to unload")
    unload_parser.set_defaults(func=cmd_unload)

    # ps
    ps_parser = sub.add_parser("ps", help="List processing slots")
    ps_parser.add_argument("-a", "--all", action="store_true",
                           help="Show all slots, not just active ones")
    ps_parser.add_argument("id", nargs='?', help="Model ID")
    ps_parser.set_defaults(func=cmd_ps)

    # pull
    pull_parser = sub.add_parser("pull", help="Download a model")
    pull_parser.add_argument("id", help="Model ID to download")
    pull_parser.set_defaults(func=cmd_pull)

    # rm
    rm_parser = sub.add_parser("rm", help="Delete a model from cache")
    rm_parser.add_argument("-f", "--force", action="store_true",
                           help="Delete without confirmation")
    rm_parser.add_argument("id", help="Model ID to delete")
    rm_parser.set_defaults(func=cmd_rm)

    args = parser.parse_args()
    #print(args)
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
