#!/usr/bin/env python3
"""llama.cpp server management CLI (docker-style)."""

import argparse
import json
import os
import signal
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


def get_api_key(args) -> str | None:
    """Get API key from env, file, or CLI arg (in that priority order)."""
    # 1. Environment variable
    key = os.environ.get("LLAMA_API_KEY")
    if key:
        return key

    # 2. --api-key-file
    key_file = getattr(args, "api_key_file", None)
    if key_file:
        try:
            return open(key_file).read().strip()
        except OSError as e:
            print(f"Error reading API key file: {e}", file=sys.stderr)
            sys.exit(1)

    # 3. --api-key
    key = getattr(args, "api_key", None)
    if key:
        return key

    return None


def get_headers(args) -> dict:
    """Build request headers with optional Bearer auth."""
    key = get_api_key(args)
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}

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

def _is_number(s):
    """Check if a string looks like a number (int or float)."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _parse_ls_args(args):
    """Parse ls positional args with heuristic.

    Returns (pattern, interval, count) tuple.
    """
    if args.pattern is not None:
        # -p/--pattern explicitly set — positional args are strictly interval and count
        pos = args.positional or []
        if len(pos) > 2:
            print("Error: with --pattern, at most 2 positional args (interval, count).", file=sys.stderr)
            sys.exit(1)
        interval = float(pos[0]) if len(pos) >= 1 else None
        count = int(pos[1]) if len(pos) >= 2 else None
        return args.pattern, interval, count

    pos = args.positional or []
    if len(pos) == 0:
        return None, None, None
    elif len(pos) == 1:
        if _is_number(pos[0]):
            return None, float(pos[0]), None
        else:
            return pos[0], None, None
    elif len(pos) == 2:
        if _is_number(pos[0]) and _is_number(pos[1]):
            return None, float(pos[0]), int(pos[1])
        elif _is_number(pos[0]):
            print(f"Error: '{pos[1]}' is not a number.", file=sys.stderr)
            sys.exit(1)
        else:
            return pos[0], float(pos[1]), None
    elif len(pos) == 3:
        return pos[0], float(pos[1]), int(pos[2])
    else:
        print("Error: too many positional arguments (max 3).", file=sys.stderr)
        sys.exit(1)


def _fetch_models(args, url, pattern):
    """Fetch and return all model rows (list of dicts)."""
    resp = requests.get(url + "models", headers=get_headers(args))
    resp.raise_for_status()
    data = resp.json()["data"]

    if not args.all and not (pattern or args.tag):
        data = [m for m in data if m["status"]["value"] == "loaded"]

    rows = []
    for m in data:
        if args.tag:
            model_tags = set(m.get("tags", []))
            arg_tags = set(args.tag)
            if not arg_tags.issubset(model_tags):
                continue
        if pattern:
            if not fnmatch(m["id"], pattern):
                continue

        rows.append({
            "ID": m["id"],
            "TAGS": show_tags(m),
            "PATH": get_path(args, m),
            "STATUS": m["status"]["value"],
            "CONTEXT": get_ctx_size(m),
            "PARAMS": get_n_params(m),
            "SIZE": get_size(m)
        })
    return rows


def cmd_ls(args):
    url = get_url(args)
    pattern, interval, count = _parse_ls_args(args)

    if interval is None:
        # single-shot mode (original behavior)
        rows = _fetch_models(args, url, pattern)
        if not rows:
            print("No models found.")
            return
        print_table(rows, LS_COLUMNS)
        return

    # polling mode
    running = True

    def handler(signum, frame):
        nonlocal running
        running = False

    old_handler = signal.signal(signal.SIGINT, handler)
    try:
        iteration = 0
        while running and (count is None or iteration < count):
            rows = _fetch_models(args, url, pattern)
            if rows:
                print_table(rows, LS_COLUMNS)
            else:
                print("No models found.")

            iteration += 1
            if running and (count is None or iteration < count):
                time.sleep(interval)
    finally:
        signal.signal(signal.SIGINT, old_handler)


def cmd_load(args):
    url = get_url(args)
    resp = requests.post(url + "models/load", headers=get_headers(args), json={"model": args.id})
    resp.raise_for_status()
    if resp.json().get("success"):
        print(f"Model {args.id} loading started.")
    else:
        print(f"Failed to load model {args.id}.", file=sys.stderr)
        sys.exit(1)


def cmd_unload(args):
    url = get_url(args)
    resp = requests.post(url + "models/unload", headers=get_headers(args), json={"model": args.id})
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

    resp = requests.delete(url + "models", headers=get_headers(args), params={"model": model_id})
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
    resp = requests.get(url + "models", headers=get_headers(args))
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
    sse_resp = requests.get(url + "models/sse", headers=get_headers(args), stream=True)
    sse_resp.raise_for_status()

    # Start the download in a separate thread so we don't block SSE
    def trigger_download():
        time.sleep(0.5)
        resp = requests.post(url + "models", headers=get_headers(args), json={"model": model_id})
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

PS_COLUMNS = ["SLOT_ID", "MODEL", "TASK#", "CONTEXT", "PROC", "PROMPT", "DECODED"]

def get_decoded(slot):
    if "next_token" not in slot:
        return "-"
    tokens = slot["next_token"]
    if isinstance(tokens, list):
        n_decoded = sum([t.get("n_decoded",0) for t in tokens])
    else:
        n_decoded = tokens.get("n_decoded", 0)
    return format_number(n_decoded)

def get_prompt_process(slot):
    processed = slot.get("n_prompt_tokens_processed", 0)
    total = slot.get("n_prompt_tokens")
    if total == 0 and processed == 0:
        return "-"
    return format_number(processed, binary=False) + " / " + format_number(total, binary=False)

def get_is_router(url, headers):
    resp = requests.get(url + "props", headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data.get("role", "") == "router"

def get_loaded_models(url, headers):
    resp = requests.get(url + "models", headers=headers)
    resp.raise_for_status()
    data = resp.json()["data"]
    return [m["id"] for m in data if m["status"]["value"] == "loaded"]

def _fetch_slots(args, url):
    """Fetch and return all slot rows (list of dicts)."""
    all_slots = []
    headers = get_headers(args)
    if args.model is None:
        if get_is_router(url, headers):
            models = get_loaded_models(url, headers)
            for model_id in models:
                resp = requests.get(url + "slots", headers=headers, params={"model": model_id})
                resp.raise_for_status()
                slots = resp.json()
                for slot in slots:
                    slot["model"] = model_id
                all_slots.extend(slots)
        else:
            resp = requests.get(url + "slots", headers=headers)
            resp.raise_for_status()
            slots = resp.json()
            all_slots.extend(slots)
    else:
        check_loaded(args, args.model)
        resp = requests.get(url + "slots", headers=headers, params={"model": args.model})
        resp.raise_for_status()
        slots = resp.json()
        for slot in slots:
            slot["model"] = args.model
        all_slots.extend(slots)

    if not args.all:
        all_slots = [s for s in all_slots if s["is_processing"]]

    if not all_slots:
        return []

    rows = []
    for s in all_slots:
        rows.append({
            "SLOT_ID": s["id"],
            "MODEL": s.get("model", "-"),
            "TASK#": s["id_task"],
            "CONTEXT": s["n_ctx"],
            "PROC": "Y" if s["is_processing"] else "N",
            "PROMPT": get_prompt_process(s),
            "DECODED": get_decoded(s),
        })
    return rows


def cmd_ps(args):
    url = get_url(args)
    interval = args.interval
    count = args.count

    if interval is None:
        # single-shot mode (original behavior)
        rows = _fetch_slots(args, url)
        if not rows:
            print("No slots found.")
            return
        print_table(rows, PS_COLUMNS)
        return

    # polling mode
    running = True

    def handler(signum, frame):
        nonlocal running
        running = False

    old_handler = signal.signal(signal.SIGINT, handler)
    try:
        iteration = 0
        while running and (count is None or iteration < count):
            rows = _fetch_slots(args, url)
            if rows:
                print_table(rows, PS_COLUMNS)
            else:
                print("No slots found.")

            iteration += 1
            if running and (count is None or iteration < count):
                time.sleep(interval)
    finally:
        signal.signal(signal.SIGINT, old_handler)


def main():
    parser = argparse.ArgumentParser(
        description="llama.cpp server management CLI",
    )
    parser.add_argument("--url", help="Server URL (e.g. http://localhost:8080/)")
    parser.add_argument("--api-key", help="API key for authentication")
    parser.add_argument("--api-key-file", help="Path to file containing API key")

    sub = parser.add_subparsers(dest="command")

    # ls
    ls_parser = sub.add_parser("ls", help="List models")
    ls_parser.add_argument("-a", "--all", action="store_true",
                           help="Show all models, not just loaded ones")
    ls_parser.add_argument("-t", "--tag", nargs='+', help="Show only models with these tags")
    ls_parser.add_argument("--full-path", action="store_true",
                           help="Show full paths to model files")
    ls_parser.add_argument("-p", "--pattern", help="Model ID filter (glob mask)")
    ls_parser.add_argument("positional", nargs='*',
                           help="[pattern] [interval] [count]")
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
    ps_parser.add_argument("-m", "--model", help="Model ID")
    ps_parser.add_argument("interval", nargs='?', type=float,
                           help="Refresh interval in seconds (polling mode)")
    ps_parser.add_argument("count", nargs='?', type=int,
                           help="Number of iterations (default: infinite)")
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
