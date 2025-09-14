import types

import main as appmod


class DummyMessage(dict):
    # Simple message shim so we don't need LangChain message classes for tests
    def __init__(self, role: str, content: str):
        super().__init__({"role": role, "content": content})
        self.role = role
        self.content = content


def make_state(text: str):
    return {"messages": [DummyMessage("user", text)], "route": ""}


def test_build_app_compiles():
    app = appmod.build_app()
    assert app is not None


def test_router_sets_route_field(monkeypatch):
    # monkeypatch classifier to return deterministic difficulty
    class Result:
        def __init__(self, difficulty):
            self.difficulty = difficulty

    called = {}

    def fake_invoke(messages):
        called["messages"] = messages
        return Result("medium")

    monkeypatch.setattr(appmod, "_classifier_chain", types.SimpleNamespace(invoke=fake_invoke))

    state = make_state("Explain merge sort")
    route_dict = appmod.router(state)
    assert route_dict == {"route": "medium"}


def test_low_medium_high_nodes_call_correct_models(monkeypatch):
    outputs = []

    def make_fake_model(name):
        def _invoke(messages):
            outputs.append(name)
            # Return a message-like dict that the app expects
            return {"role": "assistant", "content": f"ok from {name}"}
        return types.SimpleNamespace(invoke=_invoke)

    monkeypatch.setattr(appmod, "_llm_low", make_fake_model("low"))
    monkeypatch.setattr(appmod, "_llm_medium", make_fake_model("medium"))
    monkeypatch.setattr(appmod, "_llm_high", make_fake_model("high"))

    state = make_state("simple question")
    appmod.low(state)
    appmod.medium(state)
    appmod.high(state)

    assert outputs == ["low", "medium", "high"]
