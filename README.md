## Mellow Router

![Status: Beta](https://img.shields.io/badge/status-beta-orange)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![Dependencies: uv](https://img.shields.io/badge/deps-managed%20with%20uv-5C4EE5)

Layered routing assistant that triages each request, chooses a specialty workflow, and keeps a running memory of the conversation.

### Setup

1. Install dependencies (use `uv` for speed or fall back to `pip`):

   ```bash
   uv sync
   # or
   pip install -e .
   pip install -e .[dev]  # optional test extras
   ```

2. Provide your OpenAI API key (environment variable or `.env` file is fine):

   ```bash
   export OPENAI_API_KEY=sk-...
   echo 'OPENAI_API_KEY=sk-...' > .env
   ```

### Run it

Streams the assistant’s replies by node so you can watch the routing in real time.

```bash
uv run python main.py "Summarize this paragraph in one sentence" --thread user-123
uv run python main.py "Explain the backpropagation algorithm" --thread user-123
uv run python main.py "Coach me through preparing for a tough conversation" --thread user-123
```

`--thread` pins the memory checkpoint so follow-up turns reuse context.

### Visualize the flow

`main.py --save-graph` exports the compiled state machine to Mermaid/PNG. The script prefers `mmdc`; otherwise it falls back to Kroki.

![Router graph](graph.png)

```bash
# optional local renderer
npm i -g @mermaid-js/mermaid-cli

uv run python main.py --save-graph graph.png
uv run python main.py --save-graph graph.png --graph-theme dark
```

### Architecture at a glance

- **State** stores the running messages plus routing signals (`difficulty`, `clarity`, `intent`, `priority`, `notes`).
- **Classifier** produces those signals via structured output and sends the request to one of three top-level branches: clarification, human escalation, or deeper analysis.
- **Analysis layer** splits into follow-up, solution, or explanation tracks. Each track can plan, synthesize, and hand off to a common finalizer.
- **Finalizer** consolidates prior responses, reinforces next steps, and highlights any follow-up flags.
- **Memory** relies on a checkpoint store so each `thread` ID keeps its own history.

### Tests

```bash
uv run pytest
```
