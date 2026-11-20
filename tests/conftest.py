"""Fixtures for every test: no test may touch the real data folders, the user's process or real keys."""
from __future__ import annotations

import pytest

import interviewer.process as P
import interviewer.session as S
from tests.helpers import FakeLLM, FakeServer, make_process


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Sessions, weak spots and sample data go to a temp folder; no provider keys or process from the env."""
    data = tmp_path / "data"
    monkeypatch.setattr(S, "WEAK_FILE", data / "weak_spots.json")
    monkeypatch.setattr(S, "SESS_DIR", data / "sessions")
    monkeypatch.setattr(P, "DATA_ROOT", tmp_path / "xdg" / "ainterviewer")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    for var in ("AINTERVIEWER_PROCESS", "INTERVIEWER_DATA_DIR", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                "OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    return data


@pytest.fixture
def proc_dir(tmp_path):
    return make_process(tmp_path / "proc")


@pytest.fixture
def cfg(proc_dir):
    return P.load(proc_dir)


@pytest.fixture
def bank(cfg):
    return P.bank(cfg)


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def server():
    srv = FakeServer()
    yield srv
    srv.close()
