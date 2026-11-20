"""The dry run (interviewer.selftest) and the CLI's per-run LLM settings."""
import types

import pytest

import interviewer.process as P
import interviewer.session as S
from interviewer.__main__ import llm_config
from interviewer.selftest import MAX_TURNS, dry_run
from tests.helpers import make_process


class TestDryRun:
    @pytest.mark.parametrize("folder", [P.SAMPLES / "behavioural", P.TEMPLATE])
    def test_shipped_processes(self, folder):
        cfg = P.load(folder)
        assert dry_run(cfg, P.bank(cfg), P.profile(cfg)).startswith("dry run ok")

    def test_custom_process(self, cfg, bank):
        assert "questions" in dry_run(cfg, bank, "Name: Sam.")

    def test_does_not_touch_data_or_the_inputs(self, cfg, bank, isolated):
        before_cfg, before_bank = repr(cfg), repr(bank)
        weak, sess = S.WEAK_FILE, S.SESS_DIR
        dry_run(cfg, bank, "")
        assert repr(cfg) == before_cfg and repr(bank) == before_bank
        assert (S.WEAK_FILE, S.SESS_DIR) == (weak, sess) and not isolated.exists()

    def test_prints_nothing(self, cfg, bank, capsys):
        dry_run(cfg, bank, "")
        assert capsys.readouterr().out == ""

    def test_catches_an_interview_that_never_ends(self, cfg, bank, monkeypatch):
        monkeypatch.setattr(S.Interview, "run", lambda self: [self.io.listen() for _ in range(MAX_TURNS + 2)])
        with pytest.raises(AssertionError, match="did not end"):
            dry_run(cfg, bank, "")

    def test_catches_a_broken_prompt(self, cfg, bank, monkeypatch):
        monkeypatch.setattr(S, "SYSTEM", "{nonexistent_field}")
        with pytest.raises(KeyError):
            dry_run(cfg, bank, "")


def args(provider=None, model=None):
    return types.SimpleNamespace(provider=provider, model=model)


class TestLLMConfig:
    base = {"provider": "auto", "model": "claude-opus-5-5", "debrief_model": "anthropic/claude-opus-5-5",
            "feedback_model": "claude-opus-5-5"}

    def test_no_flags_changes_nothing(self):
        assert llm_config(self.base, args()) == self.base

    def test_model_flag_sets_every_job(self):
        c = llm_config(self.base, args(model="qwen3:8b"))
        assert c["model"] == c["debrief_model"] == c["feedback_model"] == "qwen3:8b"

    def test_switching_to_a_local_provider_drops_claude_models(self):
        c = llm_config(self.base, args(provider="ollama"))
        assert c["provider"] == "ollama" and c["model"] == c["feedback_model"] == "auto"
        assert c["debrief_model"] == "anthropic/claude-opus-5-5"      # prefixed: it names its own provider

    def test_switching_between_claude_providers_keeps_models(self):
        c = llm_config(self.base, args(provider="claude-cli"))
        assert c["model"] == "claude-opus-5-5"

    def test_url_provider(self):
        c = llm_config(self.base, args(provider="http://127.0.0.1:8000/v1"))
        assert c["provider"] == "http://127.0.0.1:8000/v1" and c["model"] == "auto"

    def test_does_not_modify_the_input(self):
        base = dict(self.base)
        llm_config(base, args(provider="ollama", model="x"))
        assert base == self.base
