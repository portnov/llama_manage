"""Chat functionality for llama-manage (command `run`)."""

import json
import sys

import requests


def _check_model_loaded(url, headers, model_id):
    """Check if a model with the given ID is loaded on the server."""
    resp = requests.get(url + "models", headers=headers)
    resp.raise_for_status()
    data = resp.json()["data"]
    for m in data:
        if m["id"] == model_id and m["status"]["value"] == "loaded":
            return True
    return False


def _output_non_stream(reasoning, content):
    """Output reasoning and content with appropriate prefixes (non-streaming)."""
    if reasoning:
        print("thinking>", reasoning)
        print()
    if content:
        print("assistant>", content)


def send_message(url, headers, model_id, messages, stream):
    """Send a chat message to the server and return the assistant's content.

    Returns the full content text (for adding to message history).
    Reasoning content is not returned (not stored in history).
    """
    body = {
        "model": model_id,
        "messages": messages,
        "stream": stream,
    }

    resp = requests.post(
        url + "v1/chat/completions",
        headers=headers,
        json=body,
        stream=stream,
    )
    resp.raise_for_status()

    if not stream:
        data = resp.json()
        msg = data["choices"][0]["message"]
        reasoning = msg.get("reasoning_content", "")
        content = msg.get("content", "")
        _output_non_stream(reasoning, content)
        return content

    # Streaming — handled by caller via stream_response()
    # This branch is not reached when stream=True because the caller
    # handles the response directly.
    return ""


def run_once(url, headers, model_id, prompt, system_prompt, stream):
    """Run a single chat request and exit."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    if stream:
        body = {
            "model": model_id,
            "messages": messages,
            "stream": True,
        }
        resp = requests.post(
            url + "v1/chat/completions",
            headers=headers,
            json=body,
            stream=True,
        )
        resp.raise_for_status()
        # For now, streaming in run_once is not implemented (stage 2)
        # Fall back to non-streaming output
        data = resp.json()
        msg = data["choices"][0]["message"]
        reasoning = msg.get("reasoning_content", "")
        content = msg.get("content", "")
        _output_non_stream(reasoning, content)
    else:
        send_message(url, headers, model_id, messages, stream=False)


def cmd_run(args):
    """Entry point for the `run` command."""
    from llama_manage.cli import get_url, get_headers

    url = get_url(args)
    headers = get_headers(args)
    model_id = args.id

    if not _check_model_loaded(url, headers, model_id):
        print(
            f"Error: model {model_id} is not loaded. "
            f"Run 'llama-manage load {model_id}' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.prompt is not None:
        run_once(url, headers, model_id, args.prompt, args.system, not args.no_stream)
    else:
        # REPL mode — not implemented yet (stage 3)
        print("Error: interactive mode is not implemented yet.", file=sys.stderr)
        sys.exit(1)
