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
    risk: NotRequired[str]
    notes: NotRequired[list[str]]


class DifficultyResult(BaseModel):
    difficulty: Literal["low", "medium", "high"] = Field(
        description="Overall difficulty of the user request"
    )
    clarity: Literal["clear", "vague"] = Field(
        description="Whether the user request is sufficiently specific"
    )
    intent: Literal["information", "problem_solving", "emotional_support", "other"] = Field(
        description="Dominant intent of the user request"
    )
    risk: Literal["none", "watch", "critical"] = Field(
        description="Flags if the user expresses acute distress or safety concerns"
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


def router(state: ChatbotState) -> dict[str, object]:
    last = state["messages"][-1]
    content = getattr(last, "content", "")
    text = content if isinstance(content, str) else str(content)
    result = _classifier_chain.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You are a safety-aware triage assistant for a psychologist-backed chatbot.\n"
                    "Return JSON with fields: difficulty ∈ {low, medium, high}; clarity ∈ {clear, vague}; intent ∈ {information, problem_solving, emotional_support, other}; risk ∈ {none, watch, critical}.\n"
                    "Label clarity as 'vague' if more detail is required.\n"
                    "Use risk='critical' if the user mentions self-harm, harm to others, or acute crisis; use 'watch' for elevated distress.\n"
                    "Provide an optional reason summarizing your assessment grounded in psychological or scientific cues."
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
    risk = getattr(result, "risk", None)
    if risk is None and isinstance(result, dict):
        risk = result.get("risk")
    if difficulty not in ("low", "medium", "high"):
        difficulty = "medium"
    if clarity not in ("clear", "vague"):
        clarity = "clear"
    if intent not in ("information", "problem_solving", "emotional_support", "other"):
        intent = "other"
    if risk not in ("none", "watch", "critical"):
        risk = "none"

    reason = getattr(result, "reason", None)
    if reason is None and isinstance(result, dict):
        reason = result.get("reason")

    notes = list(state.get("notes", []))
    if reason and reason not in notes:
        notes.append(reason)
    if risk == "watch":
        watch_note = "Monitor affect; provide supportive resources and normalize emotions."
        if watch_note not in notes:
            notes.append(watch_note)
    if risk == "critical":
        crisis_note = "User may be in crisis—escalate to emergency guidance."
        if crisis_note not in notes:
            notes.append(crisis_note)

    if clarity == "vague":
        route = "clarify"
    elif risk == "critical":
        route = "safety"
    else:
        route = "analysis"

    payload: dict[str, object] = {
        "route": route,
        "difficulty": difficulty,
        "clarity": clarity,
        "intent": intent,
        "risk": risk,
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
    intent = state.get("intent", "information")
    difficulty = state.get("difficulty", "medium")
    notes = list(state.get("notes", []))

    if state.get("risk") == "watch":
        monitor_note = "Observe tone for distress; include grounding or support resources."
        if monitor_note not in notes:
            notes.append(monitor_note)

    if intent == "emotional_support":
        route = "support_response"
        rationale = f"Routing to emotional support with difficulty={difficulty}."
    elif intent == "problem_solving":
        route = "coaching_entry"
        rationale = f"Routing to structured coaching with difficulty={difficulty}."
    else:
        route = "evidence_entry"
        rationale = f"Routing to evidence synthesis with difficulty={difficulty}."

    if rationale not in notes:
        notes.append(rationale)

    payload: dict[str, object] = {"route": route}
    if notes:
        payload["notes"] = notes
    return payload


def support_response(state: ChatbotState) -> dict:
    notes = state.get("notes", [])
    hints = " ".join(notes[-2:]) if notes else ""
    risk = state.get("risk", "none")
    instructions = (
        "You are a licensed-psychologist-style support specialist. Offer empathic validation, summarize what you heard, and "
        "recommend evidence-based coping skills (CBT, DBT, ACT) tailored to the user's emotions."
    )
    if hints:
        instructions += f" Triage notes: {hints}."
    if risk == "watch":
        instructions += " Include optional professional resources and normalize seeking help."
    update = _llm_reply_with_model(state, instructions, _llm_medium)
    update["route"] = "finalize"
    return update


def coaching_entry(state: ChatbotState) -> dict[str, object]:
    difficulty = state.get("difficulty", "medium")
    route = "coaching_plan" if difficulty == "high" else "coaching_direct"
    return {"route": route}


def coaching_direct(state: ChatbotState) -> dict:
    difficulty = state.get("difficulty", "medium")
    model = _llm_low if difficulty == "low" else _llm_medium
    instructions = (
        "You are a structured problem-solving coach. Deliver a short, actionable intervention grounded in CBT coaching tools, SMART goals, and habit design."
    )
    if difficulty == "medium":
        instructions += " Include 2-3 numbered steps and a reflection question."
    update = _llm_reply_with_model(state, instructions, model)
    update["route"] = "finalize"
    return update


def coaching_plan(state: ChatbotState) -> dict:
    update = _llm_reply_with_model(
        state,
        (
            "You are an expert coach. Produce a staged plan (assessment, strategy, practice, follow-up) referencing validated frameworks like CBT, WOOP, or motivational interviewing."
        ),
        _llm_high,
    )
    update["route"] = "coaching_synthesis"
    return update


def coaching_synthesis(state: ChatbotState) -> dict:
    update = _llm_reply_with_model(
        state,
        (
            "You are an expert coach. Translate the approved plan into a compassionate guide, highlight success metrics, and cite the frameworks used."
        ),
        _llm_high,
    )
    update["route"] = "finalize"
    return update


def evidence_entry(state: ChatbotState) -> dict[str, object]:
    difficulty = state.get("difficulty", "medium")
    route = "evidence_plan" if difficulty == "high" else "evidence_brief"
    return {"route": route}


def evidence_brief(state: ChatbotState) -> dict:
    difficulty = state.get("difficulty", "medium")
    model = _llm_low if difficulty == "low" else _llm_medium
    instructions = (
        "You are a science-grounded analyst. Provide a concise synthesis citing high-quality sources (systematic reviews, guidelines)."
    )
    if difficulty == "medium":
        instructions += " Include a short methods note (e.g., study types, sample sizes)."
    update = _llm_reply_with_model(state, instructions, model)
    update["route"] = "finalize"
    return update


def evidence_plan(state: ChatbotState) -> dict:
    update = _llm_reply_with_model(
        state,
        (
            "You are a research strategist. Draft a bulleted plan detailing the evidence hierarchy, key comparison points, and evaluation criteria before composing the final answer."
        ),
        _llm_high,
    )
    update["route"] = "evidence_synthesis"
    return update


def evidence_synthesis(state: ChatbotState) -> dict:
    update = _llm_reply_with_model(
        state,
        (
            "You are a senior researcher. Execute the prior plan, organize the response with numbered sections, cite sources inline, and rate evidence strength."
        ),
        _llm_high,
    )
    update["route"] = "finalize"
    return update


def safety(state: ChatbotState) -> dict:
    message = (
        "I’m concerned you may be dealing with a crisis. I’m not equipped for emergency support. "
        "Please contact local emergency services or a crisis hotline right away (e.g., 988 in the U.S., or your country’s emergency number)."
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
        "To give a precise, evidence-based answer I need a bit more context. " + reason
        if reason
        else "Could you clarify the specific outcome, population, or constraints you're working with?"
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
    risk = state.get("risk", "none")
    notes = state.get("notes", [])
    triage_summary = " ".join(notes[-3:]) if notes else ""
    instructions = (
        "You are a clinical-quality facilitator. Deliver the final response by consolidating prior guidance into clear next steps, "
        "reinforcing evidence sources or frameworks mentioned, and setting expectations for follow-up."
    )
    if triage_summary:
        instructions += f" Internal notes to reflect: {triage_summary}."
    if risk == "watch":
        instructions += " Offer optional professional resources and encourage monitoring wellbeing."
    instructions += " Close with a professional disclaimer that this is not a substitute for licensed care."
    return _llm_reply_with_model(state, instructions, _llm_medium)


def build_app():
    graph = StateGraph(ChatbotState)

    graph.add_node("router", router)
    graph.add_node("clarify", clarify)
    graph.add_node("safety", safety)
    graph.add_node("analysis", analysis)
    graph.add_node("support_response", support_response)
    graph.add_node("coaching_entry", coaching_entry)
    graph.add_node("coaching_direct", coaching_direct)
    graph.add_node("coaching_plan", coaching_plan)
    graph.add_node("coaching_synthesis", coaching_synthesis)
    graph.add_node("evidence_entry", evidence_entry)
    graph.add_node("evidence_brief", evidence_brief)
    graph.add_node("evidence_plan", evidence_plan)
    graph.add_node("evidence_synthesis", evidence_synthesis)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "router")

    def select_route(state: ChatbotState) -> str:
        return state.get("route", "analysis")

    graph.add_conditional_edges(
        "router",
        select_route,
        {
            "clarify": "clarify",
            "safety": "safety",
            "analysis": "analysis",
        },
    )

    graph.add_conditional_edges(
        "analysis",
        select_route,
        {
            "support_response": "support_response",
            "coaching_entry": "coaching_entry",
            "evidence_entry": "evidence_entry",
        },
    )

    graph.add_conditional_edges(
        "coaching_entry",
        select_route,
        {
            "coaching_direct": "coaching_direct",
            "coaching_plan": "coaching_plan",
        },
    )

    graph.add_conditional_edges(
        "coaching_plan",
        select_route,
        {
            "coaching_synthesis": "coaching_synthesis",
        },
    )

    graph.add_conditional_edges(
        "evidence_entry",
        select_route,
        {
            "evidence_brief": "evidence_brief",
            "evidence_plan": "evidence_plan",
        },
    )

    graph.add_conditional_edges(
        "evidence_plan",
        select_route,
        {
            "evidence_synthesis": "evidence_synthesis",
        },
    )

    graph.add_conditional_edges(
        "support_response",
        select_route,
        {
            "finalize": "finalize",
        },
    )

    graph.add_conditional_edges(
        "coaching_direct",
        select_route,
        {
            "finalize": "finalize",
        },
    )

    graph.add_conditional_edges(
        "coaching_synthesis",
        select_route,
        {
            "finalize": "finalize",
        },
    )

    graph.add_conditional_edges(
        "evidence_brief",
        select_route,
        {
            "finalize": "finalize",
        },
    )

    graph.add_conditional_edges(
        "evidence_synthesis",
        select_route,
        {
            "finalize": "finalize",
        },
    )

    graph.add_edge("clarify", END)
    graph.add_edge("safety", END)
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
