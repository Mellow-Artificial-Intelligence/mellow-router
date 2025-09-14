from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel, Field

load_dotenv()


class ChatbotState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    route: str


class DifficultyResult(BaseModel):
    difficulty: Literal["low", "medium", "high"] = Field(
        description="Overall difficulty of the user request"
    )
    reason: str | None = Field(
        default=None, description="Optional brief rationale for the assignment"
    )


# Classifier (structured output)
_classifier_chain = ChatOpenAI(model="gpt-5-mini").with_structured_output(DifficultyResult)

# Response models by difficulty
_llm_low = ChatOpenAI(model="gpt-5-nano")
_llm_medium = ChatOpenAI(model="gpt-5-mini")
_llm_high = ChatOpenAI(model="gpt-5")


def router(state: ChatbotState) -> dict[str, str]:
    last = state["messages"][-1]
    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)
    result = _classifier_chain.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Classify the user's last message difficulty as one of: low, medium, high.\n"
                    "Definitions:\n"
                    "- low: straightforward questions or short factual lookups.\n"
                    "- medium: multi-step reasoning or moderate context required.\n"
                    "- high: complex reasoning, synthesis, or advanced domain knowledge."
                ),
            },
            {"role": "user", "content": text},
        ]
    )
    difficulty = getattr(result, "difficulty", None)
    if difficulty is None and isinstance(result, dict):
        difficulty = result.get("difficulty")
    if difficulty not in ("low", "medium", "high"):
        difficulty = "medium"
    return {"route": difficulty}


def _llm_reply_with_model(state: ChatbotState, system_instructions: str, model: ChatOpenAI) -> dict:
    response = model.invoke(
        [
            {"role": "system", "content": system_instructions},
            *state["messages"],
        ]
    )
    return {"messages": [response]}


def low(state: ChatbotState) -> dict:
    return _llm_reply_with_model(
        state,
        "You are a concise assistant. Provide a direct, brief answer.",
        _llm_low,
    )


def medium(state: ChatbotState) -> dict:
    return _llm_reply_with_model(
        state,
        "You are a helpful assistant. Provide clear, step-by-step reasoning briefly.",
        _llm_medium,
    )


def high(state: ChatbotState) -> dict:
    return _llm_reply_with_model(
        state,
        "You are an expert assistant. Provide precise, high-quality reasoning succinctly.",
        _llm_high,
    )


def build_app():
    graph = StateGraph(ChatbotState)

    graph.add_node("router", router)
    graph.add_node("low", low)
    graph.add_node("medium", medium)
    graph.add_node("high", high)

    graph.add_edge(START, "router")

    def select_route(state: ChatbotState) -> str:
        return state["route"]

    graph.add_conditional_edges(
        "router",
        select_route,
        {
            "low": "low",
            "medium": "medium",
            "high": "high",
        },
    )

    graph.add_edge("low", END)
    graph.add_edge("medium", END)
    graph.add_edge("high", END)

    memory = MemorySaver()
    app = graph.compile(checkpointer=memory)
    return app


def save_graph_png(
    app,
    output_path: str,
    theme: str | None = None,
    kroki_url: str = "https://kroki.io",
) -> None:
    """Render the compiled graph's Mermaid diagram to a PNG file.

    Tries mermaid-cli (mmdc) first; falls back to Kroki HTTP API if unavailable.
    """
    mermaid = app.get_graph().draw_mermaid()

    import os
    import shutil
    import subprocess
    import tempfile
    import urllib.error
    import urllib.request

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    mmdc = shutil.which("mmdc")
    if mmdc:
        with tempfile.TemporaryDirectory() as tmpdir:
            src = os.path.join(tmpdir, "graph.mmd")
            with open(src, "w", encoding="utf-8") as f:
                f.write(mermaid)
            cmd = [mmdc, "-i", src, "-o", output_path]
            if theme:
                cmd += ["-t", theme]
            try:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                return
            except Exception:
                pass

    try:
        url = f"{kroki_url.rstrip('/')}/mermaid/png" + (f"?theme={theme}" if theme else "")
        data = mermaid.encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "text/plain", "Accept": "image/png"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            content = resp.read()
        with open(output_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise RuntimeError(f"Failed to render mermaid graph to PNG: {e}") from e


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Mellow router chatbot")
    parser.add_argument(
        "message",
        type=str,
        nargs="?",
        default="Hello!",
        help="User message",
    )
    parser.add_argument(
        "--thread",
        dest="thread_id",
        type=str,
        default="default",
        help="Thread ID for memory",
    )
    parser.add_argument(
        "--save-graph",
        dest="save_graph",
        type=str,
        default=None,
        help="Save current graph PNG to this path and exit",
    )
    parser.add_argument(
        "--graph-theme",
        dest="graph_theme",
        type=str,
        default=None,
        help="Mermaid theme for rendering (e.g. neutral, default, dark, forest)",
    )
    args = parser.parse_args()

    # Ensure key is present; rely on standard env var
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("OPENAI_API_KEY_BETA"):
        print("Warning: OPENAI_API_KEY not set. Set it to use ChatOpenAI.")

    app = build_app()
    config = {"configurable": {"thread_id": args.thread_id}}

    if args.save_graph:
        save_graph_png(app, args.save_graph, theme=args.graph_theme)
        print(f"Saved graph PNG to {args.save_graph}")
        return

    state = {"messages": [{"role": "user", "content": args.message}], "route": ""}
    for event in app.stream(state, config):
        for node_name, value in event.items():
            msgs = value.get("messages")
            if msgs:
                last = msgs[-1]
                content = getattr(last, "content", last)
                print(f"{node_name}: {content}")


if __name__ == "__main__":
    main()
