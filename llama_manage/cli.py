#!/usr/bin/env python3
"""llama.cpp server management CLI (docker-style)."""

import argparse
import os
import sys
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

PS_COLUMNS = ["ID", "TASK", "CONTEXT", "PROC"]


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

    args = parser.parse_args()
    #print(args)
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
