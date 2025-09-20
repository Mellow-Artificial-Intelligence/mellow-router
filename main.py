from __future__ import annotations

from typing import Annotated, Literal, NotRequired, TypedDict

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
    difficulty: NotRequired[str]
    clarity: NotRequired[str]
    intent: NotRequired[str]
    priority: NotRequired[str]
    notes: NotRequired[list[str]]


class TriageResult(BaseModel):
    difficulty: Literal["low", "medium", "high"] = Field(
        description="Overall difficulty of the user request"
    )
    clarity: Literal["clear", "vague"] = Field(
        description="Whether the user request is sufficiently specific"
    )
    intent: Literal["informational", "problem_solving", "creative", "other"] = Field(
        description="Dominant intent of the user request"
    )
    priority: Literal["normal", "follow_up", "escalate"] = Field(
        description="Escalation level for additional review"
    )
    reason: str | None = Field(
        default=None, description="Optional brief rationale for the assignment"
    )


# Classifier (structured output)
_classifier_chain = ChatOpenAI(model="gpt-5-mini").with_structured_output(TriageResult)

# Response models by difficulty
_llm_low = ChatOpenAI(model="gpt-5-nano")
_llm_medium = ChatOpenAI(model="gpt-5-mini")
_llm_high = ChatOpenAI(model="gpt-5")


def router(state: ChatbotState) -> dict[str, object]:
    last = state["messages"][-1]
    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)
    result = _classifier_chain.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a triage assistant for an advanced LLM router.\n"
                    "Return JSON with fields: "
                    "difficulty ∈ {low, medium, high}; clarity ∈ {clear, vague}; "
                    "intent ∈ {informational, problem_solving, creative, other}; "
                    "priority ∈ {normal, follow_up, escalate}.\n"
                    "Label clarity as 'vague' if you need more detail.\n"
                    "Use priority='escalate' when the request should be reviewed by a human "
                    "operator.\n"
                    "Add an optional reason summarizing how you made the decision."
                ),
            },
            {"role": "user", "content": text},
        ]
    )
    difficulty = getattr(result, "difficulty", None)
    if difficulty is None and isinstance(result, dict):
        difficulty = result.get("difficulty")
    clarity = getattr(result, "clarity", None)
    if clarity is None and isinstance(result, dict):
        clarity = result.get("clarity")
    intent = getattr(result, "intent", None)
    if intent is None and isinstance(result, dict):
        intent = result.get("intent")
    priority = getattr(result, "priority", None)
    if priority is None and isinstance(result, dict):
        priority = result.get("priority")
    if difficulty not in ("low", "medium", "high"):
        difficulty = "medium"
    if clarity not in ("clear", "vague"):
        clarity = "clear"
    if intent not in ("informational", "problem_solving", "creative", "other"):
        intent = "other"
    if priority not in ("normal", "follow_up", "escalate"):
        priority = "normal"

    reason = getattr(result, "reason", None)
    if reason is None and isinstance(result, dict):
        reason = result.get("reason")

    notes = list(state.get("notes", []))
    if reason and reason not in notes:
        notes.append(reason)
    if priority == "follow_up":
        follow_note = "Marked for follow-up attention."
        if follow_note not in notes:
            notes.append(follow_note)
    if priority == "escalate":
        escalate_note = "Requires human review."
        if escalate_note not in notes:
            notes.append(escalate_note)

    if clarity == "vague":
        route = "clarify"
    elif priority == "escalate":
        route = "escalate"
    else:
        route = "analysis"

    payload: dict[str, object] = {
        "route": route,
        "difficulty": difficulty,
        "clarity": clarity,
        "intent": intent,
        "priority": priority,
    }

    if notes:
        payload["notes"] = notes

    return payload


def _llm_reply_with_model(state: ChatbotState, system_instructions: str, model: ChatOpenAI) -> dict:
    response = model.invoke(
        [
            {"role": "system", "content": system_instructions},
            *state["messages"],
        ]
    )
    return {"messages": [response]}


def analysis(state: ChatbotState) -> dict[str, object]:
    intent = state.get("intent", "informational")
    difficulty = state.get("difficulty", "medium")
    notes = list(state.get("notes", []))

    priority = state.get("priority", "normal")
    if priority == "follow_up":
        follow_note = "Track for follow-up context."
        if follow_note not in notes:
            notes.append(follow_note)

    if intent == "creative":
        route = "followup_response"
        rationale = f"Routing to follow-up flow; difficulty={difficulty}."
    elif intent == "problem_solving":
        route = "solution_entry"
        rationale = f"Routing to solution flow; difficulty={difficulty}."
    else:
        route = "analysis_entry"
        rationale = f"Routing to analysis flow; difficulty={difficulty}."

    if rationale not in notes:
        notes.append(rationale)

    payload: dict[str, object] = {"route": route}
    if notes:
        payload["notes"] = notes
    return payload


def followup_response(state: ChatbotState) -> dict:
    notes = state.get("notes", [])
    hints = " ".join(notes[-2:]) if notes else ""
    instructions = (
        "You are a collaborative assistant. Acknowledge the user's goal, surface helpful context, "
        "and suggest the most relevant next action."
    )
    if hints:
        instructions += f" Internal notes to reflect: {hints}."
    update = _llm_reply_with_model(state, instructions, _llm_medium)
    update["route"] = "finalize"
    return update


def solution_entry(state: ChatbotState) -> dict[str, object]:
    difficulty = state.get("difficulty", "medium")
    route = "solution_plan" if difficulty == "high" else "solution_direct"
    return {"route": route}


def solution_direct(state: ChatbotState) -> dict:
    difficulty = state.get("difficulty", "medium")
    model = _llm_low if difficulty == "low" else _llm_medium
    instructions = (
        "You are an applied problem solver. Provide a concise, step-by-step answer, "
        "call out key assumptions, and recommend a quick validation check."
    )
    if difficulty == "medium":
        instructions += " Include a short checklist for the user to follow."
    update = _llm_reply_with_model(state, instructions, model)
    update["route"] = "finalize"
    return update


def solution_plan(state: ChatbotState) -> dict:
    update = _llm_reply_with_model(
        state,
        (
            "You are a senior strategist. Draft a staged plan "
            "(assessment, strategy, execution, review) before producing the final solution."
        ),
        _llm_high,
    )
    update["route"] = "solution_synthesis"
    return update


def solution_synthesis(state: ChatbotState) -> dict:
    update = _llm_reply_with_model(
        state,
        (
            "You are a senior strategist. Use the plan to produce the final answer, "
            "reference each stage, and note potential risks or follow-up actions."
        ),
        _llm_high,
    )
    update["route"] = "finalize"
    return update


def analysis_entry(state: ChatbotState) -> dict[str, object]:
    difficulty = state.get("difficulty", "medium")
    route = "analysis_plan" if difficulty == "high" else "analysis_brief"
    return {"route": route}


def analysis_brief(state: ChatbotState) -> dict:
    difficulty = state.get("difficulty", "medium")
    model = _llm_low if difficulty == "low" else _llm_medium
    instructions = (
        "You are an analytical explainer. Summarize the core ideas, point to supporting context, "
        "and mention practical examples."
    )
    if difficulty == "medium":
        instructions += " Close with two follow-up questions the user could explore."
    update = _llm_reply_with_model(state, instructions, model)
    update["route"] = "finalize"
    return update


def analysis_plan(state: ChatbotState) -> dict:
    update = _llm_reply_with_model(
        state,
        (
            "You are an analytical explainer. Outline a structured plan for a deeper dive "
            "(sections, key questions, and evidence to gather) before drafting the final response."
        ),
        _llm_high,
    )
    update["route"] = "analysis_synthesis"
    return update


def analysis_synthesis(state: ChatbotState) -> dict:
    update = _llm_reply_with_model(
        state,
        (
            "You are an analytical explainer. Follow the plan to deliver the final response, "
            "organize it with numbered sections, and highlight open questions."
        ),
        _llm_high,
    )
    update["route"] = "finalize"
    return update


def escalate(state: ChatbotState) -> dict:
    message = (
        "This request needs a human to review it before the assistant can continue. "
        "A reviewer will examine the conversation and follow up."
    )
    if state.get("notes"):
        message += " " + state["notes"][-1]
    return {
        "messages": [
            {
                "role": "assistant",
                "content": message,
            }
        ]
    }


def clarify(state: ChatbotState) -> dict:
    hints = state.get("notes", [])
    reason = hints[-1] if hints else None
    message = (
        "I need a bit more detail to help. " + reason
        if reason
        else "Could you clarify the outcome, constraints, or examples you have in mind?"
    )
    return {
        "messages": [
            {
                "role": "assistant",
                "content": message,
            }
        ]
    }


def finalize(state: ChatbotState) -> dict:
    priority = state.get("priority", "normal")
    notes = state.get("notes", [])
    triage_summary = " ".join(notes[-3:]) if notes else ""
    instructions = (
        "You are a synthesizer. Recap the key points delivered so far, "
        "highlight recommended next steps, and mention any follow-up items."
    )
    if triage_summary:
        instructions += f" Internal notes to reflect: {triage_summary}."
    if priority == "follow_up":
        instructions += " Emphasize the items flagged for follow-up."
    return _llm_reply_with_model(state, instructions, _llm_medium)


def build_app():
    graph = StateGraph(ChatbotState)

    graph.add_node("router", router)
    graph.add_node("clarify", clarify)
    graph.add_node("escalate", escalate)
    graph.add_node("analysis", analysis)
    graph.add_node("followup_response", followup_response)
    graph.add_node("solution_entry", solution_entry)
    graph.add_node("solution_direct", solution_direct)
    graph.add_node("solution_plan", solution_plan)
    graph.add_node("solution_synthesis", solution_synthesis)
    graph.add_node("analysis_entry", analysis_entry)
    graph.add_node("analysis_brief", analysis_brief)
    graph.add_node("analysis_plan", analysis_plan)
    graph.add_node("analysis_synthesis", analysis_synthesis)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "router")

    def select_route(state: ChatbotState) -> str:
        return state.get("route", "analysis")

    graph.add_conditional_edges(
        "router",
        select_route,
        {
            "clarify": "clarify",
            "escalate": "escalate",
            "analysis": "analysis",
        },
    )

    graph.add_conditional_edges(
        "analysis",
        select_route,
        {
            "followup_response": "followup_response",
            "solution_entry": "solution_entry",
            "analysis_entry": "analysis_entry",
        },
    )

    graph.add_conditional_edges(
        "solution_entry",
        select_route,
        {
            "solution_direct": "solution_direct",
            "solution_plan": "solution_plan",
        },
    )

    graph.add_conditional_edges(
        "solution_plan",
        select_route,
        {
            "solution_synthesis": "solution_synthesis",
        },
    )

    graph.add_conditional_edges(
        "analysis_entry",
        select_route,
        {
            "analysis_brief": "analysis_brief",
            "analysis_plan": "analysis_plan",
        },
    )

    graph.add_conditional_edges(
        "analysis_plan",
        select_route,
        {
            "analysis_synthesis": "analysis_synthesis",
        },
    )

    graph.add_conditional_edges(
        "followup_response",
        select_route,
        {
            "finalize": "finalize",
        },
    )

    graph.add_conditional_edges(
        "solution_direct",
        select_route,
        {
            "finalize": "finalize",
        },
    )

    graph.add_conditional_edges(
        "solution_synthesis",
        select_route,
        {
            "finalize": "finalize",
        },
    )

    graph.add_conditional_edges(
        "analysis_brief",
        select_route,
        {
            "finalize": "finalize",
        },
    )

    graph.add_conditional_edges(
        "analysis_synthesis",
        select_route,
        {
            "finalize": "finalize",
        },
    )

    graph.add_edge("clarify", END)
    graph.add_edge("escalate", END)
    graph.add_edge("finalize", END)

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
