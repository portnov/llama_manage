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


def stream_response(resp, prefix_printed=True):
    """Parse an SSE stream from /v1/chat/completions and output tokens.

    Args:
        resp: requests response with stream=True.
        prefix_printed: if True, the caller already printed "assistant> "
            before calling this function. Used to decide whether to print
            "assistant> " when switching from reasoning back to content.

    Returns:
        Full content text (reasoning is not included).
    """
    full_text = ""
    last_type = None  # None, "content", "reasoning"

    for raw_line in resp.iter_lines(decode_unicode=False):
        line = raw_line.decode("utf-8").strip()
        if not line or not line.startswith("data: "):
            continue
        data = line[6:]
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue

        delta = event.get("choices", [{}])[0].get("delta", {})

        # Reasoning tokens
        reasoning = delta.get("reasoning_content", "")
        if reasoning:
            if last_type == "content":
                print()
                sys.stdout.write("thinking> ")
                sys.stdout.flush()
            elif last_type is None:
                # reasoning came first — caller printed "assistant> ",
                # but we need "thinking> " instead. Overwrite with backspace.
                if prefix_printed:
                    # Erase "assistant> " (11 chars) and replace with "thinking> "
                    sys.stdout.write("\r\033[Kthinking> ")
                    sys.stdout.flush()
            sys.stdout.write(reasoning)
            sys.stdout.flush()
            last_type = "reasoning"

        # Content tokens
        content = delta.get("content", "")
        if content:
            if last_type == "reasoning":
                print()
                sys.stdout.write("assistant> ")
                sys.stdout.flush()
            elif last_type is None and not prefix_printed:
                # content came first and no prefix was printed yet
                sys.stdout.write("assistant> ")
                sys.stdout.flush()
            sys.stdout.write(content)
            sys.stdout.flush()
            full_text += content
            last_type = "content"

    return full_text


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


def _send_and_stream(url, headers, model_id, messages, stream, prefix_printed=True):
    """Send messages to the server and stream the response.

    Returns the full content text (for adding to message history).
    Raises requests.exceptions.HTTPError on server errors.
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

    if stream:
        if prefix_printed:
            sys.stdout.write("assistant> ")
            sys.stdout.flush()
        try:
            return stream_response(resp, prefix_printed=prefix_printed)
        except (KeyboardInterrupt, requests.exceptions.ConnectionError):
            # Ctrl-C or network broken during streaming — graceful exit
            print()
            raise
    else:
        data = resp.json()
        msg = data["choices"][0]["message"]
        reasoning = msg.get("reasoning_content", "")
        content = msg.get("content", "")
        _output_non_stream(reasoning, content)
        return content


def run_once(url, headers, model_id, prompt, system_prompt, stream):
    """Run a single chat request and exit."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        _send_and_stream(url, headers, model_id, messages, stream, prefix_printed=True)
        if stream:
            print()
    except requests.exceptions.HTTPError as e:
        print()
        try:
            err = e.response.json().get("error", {})
            msg = err.get("message", str(e))
        except (json.JSONDecodeError, AttributeError):
            msg = str(e)
        print(f"Error: {msg}", file=sys.stderr)
        sys.exit(1)
    except (KeyboardInterrupt, requests.exceptions.ConnectionError):
        print()
        sys.exit(130)  # standard exit code for Ctrl-C


def _format_error(e):
    """Extract a human-readable error message from an HTTPError."""
    try:
        err = e.response.json().get("error", {})
        return err.get("message", str(e))
    except (json.JSONDecodeError, AttributeError):
        return str(e)


def run_repl(url, headers, model_id, system_prompt, stream):
    """Run an interactive chat REPL."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    try:
        while True:
            try:
                line = input("\nuser> ")
            except EOFError:
                break  # Ctrl-D
            except KeyboardInterrupt:
                break  # Ctrl-C — graceful exit

            if not line:
                continue  # empty line — skip

            messages.append({"role": "user", "content": line})
            sys.stdout.write("assistant> ")
            sys.stdout.flush()

            try:
                content = _send_and_stream(
                    url, headers, model_id, messages, stream, prefix_printed=True
                )
                messages.append({"role": "assistant", "content": content})
            except requests.exceptions.HTTPError as e:
                # Server error (e.g., context overflow)
                print()
                print(f"\nError: {_format_error(e)}", file=sys.stderr)
                break
            except (KeyboardInterrupt, requests.exceptions.ConnectionError):
                # Ctrl-C or network broken during generation
                print()
                break

            if stream:
                print()
    except KeyboardInterrupt:
        pass  # graceful exit
    finally:
        print()  # final newline


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
        run_repl(url, headers, model_id, args.system, not args.no_stream)
