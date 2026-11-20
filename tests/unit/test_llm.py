"""interviewer.llm: provider routing, the OpenAI-compatible backend (against a stand-in server), the claude CLI
backend (against a fake `claude`), the Anthropic backend (stubbed client), think-stripping and continuation."""
import json
import random
import types

import pytest

import interviewer.llm as L
from tests.helpers import FakeServer, fake_claude

BASE = {"model": "auto", "debrief_model": "auto", "feedback_model": "auto", "max_tokens": 160, "effort": "low"}


def llm_for(url, **cfg):
    return L.LLM({**BASE, "provider": url, **cfg})


# --- routing -------------------------------------------------------------------------------------

class TestRouting:
    def test_url_provider_picks_first_model(self, server):
        llm = llm_for(server.url)
        assert llm.resolve("auto") == (server.url, "test-model") and llm.is_local()

    def test_url_provider_remote_host_is_not_local(self):
        llm = L.LLM.__new__(L.LLM)
        llm.cfg, llm.providers = {}, {k: dict(v) for k, v in L.PRESETS.items()}
        assert llm._auto("https://llm.example.com/v1") == "https://llm.example.com/v1"
        assert not llm.providers["https://llm.example.com/v1"]["local"]

    def test_explicit_model_is_kept(self, server):
        assert llm_for(server.url, model="my-model").resolve("my-model") == (server.url, "my-model")

    def test_claude_model_name_on_local_server_means_auto(self, server):
        assert llm_for(server.url).resolve("claude-opus-5-5")[1] == "test-model"

    def test_prefixed_model_uses_its_provider(self, server):
        llm = L.LLM({**BASE, "provider": server.url, "providers": {"mine": {"base_url": server.url, "local": True}},
                     "debrief_model": "mine/big-model"})
        assert llm.jobs["debrief"] == ("mine", "big-model") and llm.jobs["interviewer"][0] == server.url
        assert "mine: big-model (debrief)" in llm.describe()

    def test_unprefixed_slash_model_stays_whole(self, server):
        llm = llm_for(server.url, model="meta-llama/llama-3")
        assert llm.resolve("meta-llama/llama-3") == (server.url, "meta-llama/llama-3")

    def test_unknown_provider(self):
        with pytest.raises(SystemExit, match="Unknown LLM provider 'nope'"):
            L.LLM({**BASE, "provider": "nope"})

    def test_cloud_provider_without_key(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        with pytest.raises(SystemExit, match="GROQ_API_KEY"):
            L.LLM({**BASE, "provider": "groq", "model": "llama-3.3-70b"})

    def test_cloud_provider_needs_a_model_name(self, server, monkeypatch):
        monkeypatch.setenv("TEST_KEY", "k")
        with pytest.raises(SystemExit, match="set a model"):
            L.LLM({**BASE, "provider": "cloud", "providers": {"cloud": {"base_url": server.url, "key_env": "TEST_KEY"}}})

    def test_local_server_not_running(self):
        with pytest.raises(SystemExit, match="Can't reach lmstudio.*LM Studio"):
            L.LLM({**BASE, "provider": "lmstudio", "providers": {"lmstudio": {"base_url": "http://127.0.0.1:9/v1"}}})

    def test_provider_without_base_url(self):
        with pytest.raises(SystemExit, match="needs a base_url"):
            L.LLM({**BASE, "provider": "x", "providers": {"x": {"kind": "openai"}}})

    def test_auto_prefers_anthropic_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        llm = L.LLM.__new__(L.LLM)
        llm.cfg, llm.providers = {}, {k: dict(v) for k, v in L.PRESETS.items()}
        assert llm._auto("auto") == "anthropic" and llm._auto("api") == "anthropic"

    def test_auto_uses_claude_cli_then_ollama(self, monkeypatch, tmp_path):
        monkeypatch.setattr(L.shutil, "which", lambda name: "/usr/bin/claude" if name == "claude" else None)
        llm = L.LLM.__new__(L.LLM)
        llm.cfg, llm.providers = {}, {k: dict(v) for k, v in L.PRESETS.items()}
        assert llm._auto("auto") == "claude-cli"
        monkeypatch.setattr(L.shutil, "which", lambda name: None)
        monkeypatch.setattr(L, "_reachable", lambda url, timeout=0.5: True)
        assert llm._auto("auto") == "ollama"
        monkeypatch.setattr(L, "_reachable", lambda url, timeout=0.5: False)
        with pytest.raises(SystemExit, match="No LLM provider found"):
            llm._auto("auto")

    def test_provider_names(self):
        names = L.provider_names({"providers": {"work": {}}})
        assert {"anthropic", "claude-cli", "ollama", "vllm", "openrouter", "work"} <= set(names)


# --- OpenAI-compatible backend ---------------------------------------------------------------------------

class TestOpenAICompat:
    def test_streams_text_and_records_usage(self, server):
        server.reply = lambda body: "Hello there, how are you?"
        llm = llm_for(server.url)
        assert "".join(llm.stream("sys", [{"role": "user", "content": "hi"}], label="t")) == "Hello there, how are you?"
        assert llm.usage == [("t", 100, len("Hello there, how are you?") // 4)]

    def test_request_shape(self, server):
        llm = llm_for(server.url)
        llm.complete("SYSTEM", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                                {"role": "user", "content": "c"}], label="t", max_tokens=50)
        body = server.requests[-1]
        assert body["model"] == "test-model" and body["stream"] is True
        assert [m["role"] for m in body["messages"]] == ["system", "user", "assistant", "user"]
        assert body["messages"][0]["content"] == "SYSTEM"
        assert body["max_tokens"] == L.LOCAL_MIN_MAX_TOKENS        # local: raised so thinking models have room
        assert body["stream_options"] == {"include_usage": True} and "reasoning_effort" not in body

    def test_think_blocks_are_never_returned(self, server):
        server.reply = lambda body: "<think>The answer lacks numbers</think>[F] What was the result?"
        server.chunk = 3
        assert llm_for(server.url).complete("s", [{"role": "user", "content": "x"}], label="t") == "[F] What was the result?"

    def test_server_rejecting_stream_options(self):
        srv = FakeServer(reply=lambda b: "Fine.", reject_stream_options=True)
        try:
            llm = llm_for(srv.url)
            assert llm.complete("s", [{"role": "user", "content": "x"}], label="t") == "Fine."
            assert llm.complete("s", [{"role": "user", "content": "x"}], label="t") == "Fine."
            assert [("stream_options" in r) for r in srv.requests] == [True, False, False]
            assert llm.usage and llm.usage[0][2] > 0                 # estimated when the server sends none
        finally:
            srv.close()

    @pytest.mark.parametrize("status, expect", [(429, L.APOLOGY_RATE), (500, L.APOLOGY_RATE), (503, L.APOLOGY_RATE)])
    def test_transient_errors_become_a_spoken_apology(self, server, status, expect):
        server.reply = lambda body: (status, {"error": {"message": "busy"}})
        assert llm_for(server.url).complete("s", [{"role": "user", "content": "x"}], label="t") == expect

    @pytest.mark.parametrize("status", [400, 401, 404])
    def test_configuration_errors_stop_with_the_server_message(self, server, status):
        server.reply = lambda body: (status, {"error": {"message": "model not loaded"}})
        with pytest.raises(SystemExit, match=f"HTTP {status} model not loaded"):
            llm_for(server.url).complete("s", [{"role": "user", "content": "x"}], label="t")

    def test_local_server_dying_mid_session(self):
        srv = FakeServer()
        llm = llm_for(srv.url)
        srv.close()
        with pytest.raises(SystemExit, match="Can't reach"):
            llm.complete("s", [{"role": "user", "content": "x"}], label="t")

    def test_api_key_and_options(self, server, monkeypatch):
        monkeypatch.setenv("MY_KEY", "secret")
        llm = L.LLM({**BASE, "model": "m", "debrief_model": "m", "feedback_model": "m", "provider": "c", "providers": {"c": {
            "base_url": server.url, "key_env": "MY_KEY", "send_effort": True, "extra_body": {"temperature": 0.3},
            "max_tokens_field": "max_completion_tokens"}}})
        llm.complete("s", [{"role": "user", "content": "x"}], label="t", effort="xhigh", max_tokens=99)
        body, headers = server.requests[-1], server.headers[-1]
        assert headers["Authorization"] == "Bearer secret"
        assert body["reasoning_effort"] == "high" and body["temperature"] == 0.3
        assert body["max_completion_tokens"] == 99 and "max_tokens" not in body

    def test_no_usage_from_server_is_estimated(self):
        srv = FakeServer(reply=lambda b: "x" * 80, usage=False)
        try:
            llm = llm_for(srv.url)
            llm.complete("s", [{"role": "user", "content": "y" * 40}], label="t")
            assert llm.usage[-1][2] == 20
        finally:
            srv.close()


# --- think stripping --------------------------------------------------------------------------------------

@pytest.mark.parametrize("src, want", [
    ("<think>a\nb</think>Hello there.", "Hello there."),
    ("Hi <thinking>x</thinking>you", "Hi you"),
    ("[F] Is 3 < 4 true?", "[F] Is 3 < 4 true?"),
    ("Keep <b>bold</b> text.", "Keep <b>bold</b> text."),
    ("<think>never closed", ""),
    ("A<think>1</think>B<think>2</think>C", "ABC"),
])
def test_think_stripper_any_chunking(src, want):
    rnd = random.Random(0)
    for _ in range(200):
        s, out, i = L.ThinkStripper(), "", 0
        while i < len(src):
            n = rnd.randint(1, 6)
            out += "".join(s.feed(src[i:i + n]))
            i += n
        assert out + s.flush() == want


# --- continuation --------------------------------------------------------------------------------------------

class TestCompleteUntil:
    def test_continues_until_done(self, server):
        parts = iter(["First part, sec", "ond part END"])
        server.reply = lambda body: next(parts)
        out = llm_for(server.url).complete_until("s", [{"role": "user", "content": "x"}], lambda t: "END" in t, label="t")
        assert out == "First part, second part END"          # joined exactly, even mid-word
        assert server.requests[-1]["messages"][-1]["content"].startswith("Your reply was cut off")

    def test_stops_when_the_model_starts_over(self, server):
        text = "# Report\nThe candidate gave a clear overview of their background and their two biggest results."
        server.reply = lambda body: text
        out = llm_for(server.url).complete_until("s", [{"role": "user", "content": "x"}], lambda t: False, label="t")
        assert out == text and len(server.requests) == 2

    def test_round_limit(self, server):
        server.reply = lambda body: f"part {len(server.requests)} "
        llm_for(server.url).complete_until("s", [{"role": "user", "content": "x"}], lambda t: False, label="t", rounds=2)
        assert len(server.requests) == 3


# --- claude CLI ----------------------------------------------------------------------------------------------

class TestClaudeCLI:
    def test_streams_and_counts_cache_tokens(self, tmp_path, monkeypatch):
        log = fake_claude(tmp_path / "bin", text="[F] How did you measure that?")
        monkeypatch.setenv("PATH", f"{tmp_path / 'bin'}:{__import__('os').environ['PATH']}")
        llm = L.LLM({**BASE, "provider": "claude-cli", "model": "claude-opus-5-5", "debrief_model": "claude-opus-5-5",
                     "feedback_model": "claude-opus-5-5"})
        out = llm.complete("SYS", [{"role": "user", "content": "hello"}], label="t", effort="low")
        assert out == "[F] How did you measure that?" and llm.usage == [("t", 15, 7)]
        call = json.loads(log.read_text())
        argv = call["argv"]
        assert argv[argv.index("--model") + 1] == "claude-opus-5-5" and argv[argv.index("--system-prompt") + 1] == "SYS"
        assert "--no-session-persistence" in argv and argv[argv.index("--tools") + 1] == ""
        assert call["stdin"] == "hello"

    def test_missing_cli(self, monkeypatch):
        monkeypatch.setattr(L.shutil, "which", lambda name: None)
        with pytest.raises(SystemExit, match="claude"):
            L.LLM({**BASE, "provider": "claude-cli", "model": "claude-opus-5-5", "debrief_model": "x", "feedback_model": "x"})

    def test_flatten_conversation(self):
        assert L._flatten([{"role": "user", "content": "only"}]) == "only"
        text = L._flatten([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                           {"role": "user", "content": "c"}])
        assert "Candidate: a" in text and "You: b" in text and "Latest candidate turn: c" in text


# --- Anthropic API (stubbed client) ------------------------------------------------------------------------

class _Stream:
    def __init__(self, texts, stop="end_turn"):
        self.text_stream, self.stop = texts, stop

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_final_message(self):
        return types.SimpleNamespace(usage=types.SimpleNamespace(input_tokens=11, output_tokens=5), stop_reason=self.stop)


def anthropic_backend(behaviour):
    errors = types.SimpleNamespace(RateLimitError=type("RateLimitError", (Exception,), {}),
                                   APIConnectionError=type("APIConnectionError", (Exception,), {}),
                                   AuthenticationError=type("AuthenticationError", (Exception,), {}))
    calls = []

    def stream(**kw):
        calls.append(kw)
        return behaviour(errors, kw)
    client = types.SimpleNamespace(messages=types.SimpleNamespace(stream=lambda **kw: stream(beta=False, **kw)),
                                   beta=types.SimpleNamespace(messages=types.SimpleNamespace(
                                       stream=lambda **kw: stream(beta=True, **kw))))
    b = L.AnthropicBackend.__new__(L.AnthropicBackend)
    b.anthropic, b.client = errors, client
    return b, calls, errors


class TestAnthropic:
    def run(self, b, model="claude-opus-5-5"):
        return list(b.stream("SYS", [{"role": "user", "content": "x"}], model, 160, "low"))

    def test_opus_uses_server_side_fallback(self):
        b, calls, _ = anthropic_backend(lambda e, kw: _Stream(["Hi", " there"]))
        out = self.run(b)
        assert out == ["Hi", " there", (11, 5)]
        assert calls[0]["beta"] and calls[0]["fallbacks"] == "default" and calls[0]["betas"] == [L.FALLBACK_BETA]
        assert calls[0]["output_config"] == {"effort": "low"} and calls[0]["system"] == "SYS"

    def test_haiku_uses_plain_endpoint(self):
        b, calls, _ = anthropic_backend(lambda e, kw: _Stream(["Hi"]))
        self.run(b, "claude-haiku-5-5")
        assert not calls[0]["beta"] and "fallbacks" not in calls[0]

    def test_refusal(self):
        b, _, _ = anthropic_backend(lambda e, kw: _Stream([], stop="refusal"))
        assert self.run(b)[-1] == " Let's move on."

    def test_rate_limit_and_connection_errors_are_spoken(self):
        def boom(kind):
            def behaviour(e, kw):
                raise getattr(e, kind)("x")
            return behaviour
        assert self.run(anthropic_backend(boom("RateLimitError"))[0]) == [L.APOLOGY_RATE]
        assert self.run(anthropic_backend(boom("APIConnectionError"))[0]) == [L.APOLOGY_CONN]

    def test_bad_key_stops_with_a_message(self):
        def behaviour(e, kw):
            raise e.AuthenticationError("invalid x-api-key")
        b, _, _ = anthropic_backend(behaviour)
        with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
            self.run(b)
