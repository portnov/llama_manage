# llama-manage

A `docker`-style CLI for managing a [llama.cpp](https://github.com/ggml-org/llama.cpp) server.

Manage models and monitor slots through the llama.cpp server REST API.

## Installation

```bash
pip install .
# or
pipx install .
```

For development:

```bash
pip install -e .
```

## Configuration

### Server URL

Determined by (in priority order):

1. The `--url` command-line flag
2. The `$LLAMA_URL` environment variable

```bash
export LLAMA_URL=http://localhost:8080/
```

If neither is set, the utility exits with an error.

### Authentication

API key is determined by (in priority order):

1. The `--api-key` flag (key passed directly)
2. The `--api-key-file` flag (reads key from file)
3. The `$LLAMA_API_KEY` environment variable
4. No authentication (may receive 401 from server)

```bash
export LLAMA_API_KEY=your-secret-key
# or
llama-manage --api-key your-secret-key ls
# or
llama-manage --api-key-file ~/.llama-api-key ls
```

## Commands

### `ls` — list models

Prints a table of models: ID, tags, path, status, context, parameters, size.

```bash
llama-manage ls                      # loaded models only
llama-manage ls -a                   # all models
llama-manage ls gemma                # filter by ID (glob)
llama-manage ls -t vision            # filter by tags
llama-manage ls --full-path          # show full file paths
```

**Extended columns** (`-l` / `--long`): adds INPUT (input modalities), OUTPUT (output modalities), and ARGS (server startup args).

```bash
llama-manage ls -l                   # show extended columns
```

**Custom columns** (`-o` / `--output`): specify exactly which columns to display.

```bash
llama-manage ls -o ID,PATH,ARGS      # show only ID, PATH, ARGS
llama-manage ls -o ID,SIZE,INPUT     # any combination
```

Available columns: `ID`, `TAGS`, `PATH`, `STATUS`, `CONTEXT`, `PARAMS`, `SIZE`, `INPUT`, `OUTPUT`, `ARGS`.

**Expanded output** (`-x` / `--expanded`): replaces the table with a `key: value` format, one model per block.

```bash
llama-manage ls -x                   # expanded output
llama-manage ls -l -x                # expanded with all columns
llama-manage ls -o ID,PATH -x        # expanded with custom columns
```

**Auto-pager:** when output is larger than the terminal, it is automatically piped through a pager (`$PAGER`, default `less`). This applies to single-shot mode only (not polling).

**Polling mode** (like `iostat`):

```bash
llama-manage ls 5                    # every 5 seconds (Ctrl-C to exit)
llama-manage ls 5 10                 # every 5 seconds, 10 iterations
llama-manage ls gemma 5              # filter + polling
llama-manage ls gemma 5 10           # filter + polling with limit
```

For numeric patterns, use `-p`:

```bash
llama-manage ls -p "123" 5           # pattern "123", polling every 5s
```

**Positional argument heuristic:**

| Command | Pattern | Interval | Count |
|---------|---------|----------|-------|
| `ls` | — | — | — |
| `ls 5` | — | 5s | ∞ |
| `ls gemma` | gemma | — | — |
| `ls 5 10` | — | 5s | 10 |
| `ls gemma 5` | gemma | 5s | ∞ |
| `ls gemma 5 10` | gemma | 5s | 10 |

### `load` — load a model

```bash
llama-manage load ggml-org/gemma-3-4b-it-GGUF:Q4_K_M
```

### `unload` — unload a model

```bash
llama-manage unload ggml-org/gemma-3-4b-it-GGUF:Q4_K_M
```

### `ps` — list slots

Shows active tasks: slot ID, model, task number, context, processing status, prompt progress, and decoded tokens.

Supports router mode: automatically detects server type and collects slots across all loaded models.

```bash
llama-manage ps                      # active tasks only
llama-manage ps -a                   # all slots
llama-manage ps -m gemma             # slots for a specific model
```

**Polling mode:**

```bash
llama-manage ps 5                    # every 5 seconds (Ctrl-C to exit)
llama-manage ps 5 10                 # every 5 seconds, 10 iterations
```

### `pull` — download a model

Downloads a model with a progress bar showing percentage, size, and speed.

```bash
llama-manage pull ggml-org/gemma-3-4b-it-GGUF:Q4_K_M
```

Progress is tracked via the `/models/sse` SSE endpoint.

### `rm` — delete a model

Removes a model from the server cache. Requires confirmation by default.

```bash
llama-manage rm ggml-org/gemma-3-4b-it-GGUF:Q4_K_M    # with confirmation
llama-manage rm -f ggml-org/gemma-3-4b-it-GGUF:Q4_K_M  # skip confirmation
```

### `run` — chat with a model

Interactive chat or single prompt via the OpenAI-compatible `/v1/chat/completions` endpoint.

```bash
llama-manage run gemma "Hello"          # single prompt (like curl)
llama-manage run gemma                  # interactive chat (REPL)
```

**Options:**

```bash
llama-manage run gemma --system "You are a helpful assistant"  # system prompt
llama-manage run gemma "Hello" --no-stream                     # disable streaming
```

**Usage:**

- **Single prompt:** outputs the response and exits
- **REPL:** enter messages line by line (Enter to send, Ctrl-D to exit)
- **Streaming:** enabled by default; tokens appear as they are generated
- **Reasoning models:** reasoning tokens are shown with `thinking>` prefix, content with `assistant>` prefix
- **Error handling:** if the model is not loaded, the command exits with a hint to run `load` first

**Example session:**

```
$ llama-manage run gemma

user> What is 2+2?

assistant> The answer is 4.

user> ^D  # exit
```

## Examples

```bash
# Set the URL once
export LLAMA_URL=http://localhost:8080/

# List loaded models
llama-manage ls

# Show extended info (modalities, args)
llama-manage ls -l

# Custom columns in expanded format
llama-manage ls -o ID,PATH,SIZE,INPUT -x

# Download and load a model
llama-manage pull unsloth/gemma-4-E2B-it-qat-GGUF:Q8_0
llama-manage load unsloth/gemma-4-E2B-it-qat-GGUF:Q8_0

# Monitor slots in real time
llama-manage ps 2

# Monitor models with a filter
llama-manage ls gemma 5

# Chat with a model
llama-manage run gemma "Explain quantum computing in simple terms"
llama-manage run gemma  # interactive REPL

# Clean up cache
llama-manage ls -a
llama-manage rm -f old-model-id
```
