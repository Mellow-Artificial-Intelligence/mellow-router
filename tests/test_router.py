import types

import main as appmod


class DummyMessage(dict):
    def __init__(self, role: str, content: str):
        super().__init__({"role": role, "content": content})
        self.role = role
        self.content = content


def make_state(text: str):
    return {"messages": [DummyMessage("user", text)], "route": ""}


def test_build_app_compiles():
    app = appmod.build_app()
    assert app is not None


def test_router_returns_metadata(monkeypatch):
    class Result:
        def __init__(self):
            self.difficulty = "medium"
            self.clarity = "clear"
            self.intent = "informational"
            self.priority = "normal"
            self.reason = "basic lookup"

    monkeypatch.setattr(
        appmod,
        "_classifier_chain",
        types.SimpleNamespace(invoke=lambda _: Result()),
    )

    state = make_state("Explain merge sort")
    triage = appmod.router(state)

    assert triage["route"] == "analysis"
    assert triage["intent"] == "informational"
    assert triage["priority"] == "normal"
    assert "notes" in triage and "basic lookup" in triage["notes"]


def test_router_handles_clarify_and_escalate(monkeypatch):
    class ClarifyResult:
        def __init__(self):
            self.difficulty = "low"
            self.clarity = "vague"
            self.intent = "other"
            self.priority = "normal"
            self.reason = None

    class EscalateResult:
        def __init__(self):
            self.difficulty = "high"
            self.clarity = "clear"
            self.intent = "problem_solving"
            self.priority = "escalate"
            self.reason = "needs review"

    monkeypatch.setattr(
        appmod,
        "_classifier_chain",
        types.SimpleNamespace(invoke=lambda _: ClarifyResult()),
    )
    clarify_state = make_state("???")
    clarify = appmod.router(clarify_state)
    assert clarify["route"] == "clarify"

    monkeypatch.setattr(
        appmod,
        "_classifier_chain",
        types.SimpleNamespace(invoke=lambda _: EscalateResult()),
    )
    escalate_state = make_state("Sensitive data request")
    escalate = appmod.router(escalate_state)
    assert escalate["route"] == "escalate"
    assert "Requires human review." in escalate.get("notes", [])


def test_analysis_routes_by_intent():
    state = {
        "messages": [DummyMessage("user", "help me brainstorm")],
        "route": "",
        "intent": "creative",
        "difficulty": "medium",
        "notes": [],
        "priority": "normal",
    }
    result = appmod.analysis(state)
    assert result["route"] == "followup_response"
