import types

import main as appmod


def test_save_graph_png_with_mmdc(monkeypatch, tmp_path):
    app = appmod.build_app()

    # Fake which returns a path
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/mmdc")

    # Intercept subprocess.run to avoid calling real mmdc
    calls = {}

    def fake_run(cmd, check, stdout, stderr):
        calls["cmd"] = cmd
        # create a fake output file since mmdc would do so
        out_index = cmd.index("-o") + 1
        out_path = cmd[out_index]
        with open(out_path, "wb") as f:
            f.write(b"PNGDATA")
        return types.SimpleNamespace(returncode=0)

    monkeypatch.setattr("subprocess.run", fake_run)

    out_file = tmp_path / "graph.png"
    appmod.save_graph_png(app, str(out_file))

    assert out_file.exists()
    assert calls["cmd"][0].endswith("mmdc")


def test_save_graph_png_with_kroki(monkeypatch, tmp_path):
    app = appmod.build_app()

    # No mmdc available
    monkeypatch.setattr("shutil.which", lambda name: None)

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(req):
        return FakeResponse(b"PNGDATA")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    out_file = tmp_path / "graph.png"
    appmod.save_graph_png(app, str(out_file))

    assert out_file.exists()
    assert out_file.read_bytes() == b"PNGDATA"
