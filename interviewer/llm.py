"""LLM providers behind one interface: LLM.stream(system, messages, ...) -> iterator of text chunks.

Providers:
  anthropic        Anthropic API (ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / `ant auth login` profile)
  claude-cli       `claude -p` with your Claude subscription
  OpenAI-compatible chat-completions endpoints, cloud or local:
    openai, openrouter, groq, gemini, mistral, deepseek, together      (API key from an env var)
    ollama, lmstudio, llamacpp, vllm                                    (local, no key)
    anything else under llm.providers with a base_url

A model can name its provider with a prefix, so each job can use a different one:
  model: ollama/qwen3:14b            debrief_model: anthropic/claude-opus-5-5
An unprefixed model uses llm.provider. "auto" as a model (or a claude-* model on a non-Claude provider)
picks the first model a local server offers.

Token usage of every call is appended to self.usage as (label, input_tokens, output_tokens).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from typing import Iterator

# Presets for OpenAI-compatible providers. `local`: no key needed, models can be auto-picked, and
# max_tokens is raised to `min_max_tokens` so thinking models have room (unused tokens cost nothing).
PRESETS: dict[str, dict] = {
    "anthropic": {"kind": "anthropic"},
    "claude-cli": {"kind": "claude-cli"},
    "openai": {"kind": "openai", "base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY",
               "max_tokens_field": "max_completion_tokens"},
    "openrouter": {"kind": "openai", "base_url": "https://openrouter.ai/api/v1", "key_env": "OPENROUTER_API_KEY"},
    "groq": {"kind": "openai", "base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "gemini": {"kind": "openai", "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "key_env": "GEMINI_API_KEY"},
    "mistral": {"kind": "openai", "base_url": "https://api.mistral.ai/v1", "key_env": "MISTRAL_API_KEY"},
    "deepseek": {"kind": "openai", "base_url": "https://api.deepseek.com/v1", "key_env": "DEEPSEEK_API_KEY"},
    "together": {"kind": "openai", "base_url": "https://api.together.xyz/v1", "key_env": "TOGETHER_API_KEY"},
    "ollama": {"kind": "openai", "base_url": "http://localhost:11434/v1", "local": True,
               "hint": "is `ollama serve` running? Pull a model with `ollama pull <model>`."},
    "lmstudio": {"kind": "openai", "base_url": "http://localhost:1234/v1", "local": True,
                 "hint": "start the server in LM Studio (Developer tab) and load a model."},
    "llamacpp": {"kind": "openai", "base_url": "http://localhost:8080/v1", "local": True,
                 "hint": "start it with `llama-server -m <model.gguf>`."},
    "vllm": {"kind": "openai", "base_url": "http://localhost:8000/v1", "local": True,
             "hint": "start it with `vllm serve <model>`."},
}
LOCAL_MIN_MAX_TOKENS = 2048

# Claude models that take the server-side refusal fallback (`fallbacks: "default"`).
FALLBACK_MODELS = ("claude-opus-5-5", "claude-opus-5", "claude-fable-5-2", "claude-fable-5-1", "claude-sonnet-5-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"

APOLOGY_RATE = " Sorry, give me a second, I lost my notes. Could you repeat that?"
APOLOGY_CONN = " Sorry, my connection dropped. Could you say that again?"


class LLMError(SystemExit):
    """A configuration problem (unknown provider, missing key, server not running): stops the program
    with a readable message instead of a traceback."""


def provider_names(cfg: dict | None = None) -> list[str]:
    return sorted(set(PRESETS) | set((cfg or {}).get("providers") or {}))


class LLM:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.usage: list[tuple[str, int, int]] = []
        self.providers = {k: dict(v) for k, v in PRESETS.items()}
        for name, over in (cfg.get("providers") or {}).items():
            self.providers[name] = {**self.providers.get(name, {"kind": "openai"}), **(over or {})}
        self.default = self._auto(cfg.get("provider") or cfg.get("backend") or "auto")
        self._backends: dict[str, object] = {}
        self._resolved: dict[str, tuple[str, str]] = {}
        # resolve every job's model now, so a missing key or a stopped local server fails before the call
        self.jobs = {job: self.resolve(cfg.get(key) or cfg["model"])
                     for job, key in (("interviewer", "model"), ("debrief", "debrief_model"),
                                      ("feedback", "feedback_model"))}
        for prov, _ in self.jobs.values():
            self._backend(prov)          # set up every provider now: a missing CLI or key fails at startup
        self.backend = self.describe()

    # --- provider and model resolution ---------------------------------------------------
    def _auto(self, name: str) -> str:
        if name == "api":                       # old config value
            return "anthropic"
        if name.startswith(("http://", "https://")):   # --provider http://127.0.0.1:8000/v1: any OpenAI-compatible server
            from urllib.parse import urlparse
            host = urlparse(name).hostname or ""
            self.providers[name] = {"kind": "openai", "base_url": name,
                                    "local": host in ("localhost", "127.0.0.1", "::1") or host.endswith(".local")}
            return name
        if name != "auto":
            if name not in self.providers:
                raise LLMError(f"Unknown LLM provider '{name}'. Known: {', '.join(provider_names(self.cfg))}, "
                               f"or add one under llm.providers with a base_url.")
            return name
        if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            return "anthropic"
        if shutil.which("claude"):
            return "claude-cli"
        if _reachable(self.providers["ollama"]["base_url"]):
            return "ollama"
        raise LLMError("No LLM provider found: set ANTHROPIC_API_KEY, install the `claude` CLI, start Ollama, "
                       "or pick one with --provider / llm.provider (" + ", ".join(provider_names(self.cfg)) + ").")

    def resolve(self, spec: str) -> tuple[str, str]:
        """'ollama/qwen3:14b' -> ('ollama', 'qwen3:14b'); unprefixed -> the default provider."""
        if spec in self._resolved:
            return self._resolved[spec]
        provider, model = self.default, spec
        head, sep, rest = spec.partition("/")
        if sep and head in self.providers:
            provider, model = head, rest
        p = self.providers[provider]
        if p["kind"] == "openai" and (not model or model == "auto" or model.startswith("claude-")):
            model = self._backend(provider).pick_model(model)
        self._resolved[spec] = (provider, model)
        return provider, model

    def _backend(self, provider: str):
        if provider not in self._backends:
            p = self.providers[provider]
            kind = p.get("kind", "openai")
            if kind == "anthropic":
                self._backends[provider] = AnthropicBackend()
            elif kind == "claude-cli":
                self._backends[provider] = ClaudeCLIBackend()
            elif kind == "openai":
                self._backends[provider] = OpenAICompatBackend(provider, p)
            else:
                raise LLMError(f"llm.providers.{provider}: unknown kind '{kind}' (anthropic | claude-cli | openai)")
        return self._backends[provider]

    def describe(self) -> str:
        seen = {}
        for job, (prov, model) in self.jobs.items():
            seen.setdefault(f"{prov}: {model}", []).append(job)
        if len(seen) == 1:
            return next(iter(seen))
        return " · ".join(f"{k} ({', '.join(v)})" for k, v in seen.items())

    def is_local(self, spec: str | None = None) -> bool:
        prov, _ = self.resolve(spec or self.cfg["model"])
        return bool(self.providers[prov].get("local"))

    # --- calls ---------------------------------------------------------------------------
    def totals(self) -> tuple[int, int]:
        return sum(u[1] for u in self.usage), sum(u[2] for u in self.usage)

    def stream(self, system: str, messages: list[dict], *, label: str, max_tokens: int | None = None,
               model: str | None = None, effort: str | None = None) -> Iterator[str]:
        provider, model_id = self.resolve(model or self.cfg["model"])
        max_tokens = max_tokens or self.cfg["max_tokens"]
        effort = effort or self.cfg.get("effort", "low")
        backend = self._backend(provider)
        for item in backend.stream(system, messages, model_id, max_tokens, effort):
            if isinstance(item, tuple):                      # (input_tokens, output_tokens) at the end
                self.usage.append((label, item[0], item[1]))
            else:
                yield item

    def complete(self, *args, **kwargs) -> str:
        return "".join(self.stream(*args, **kwargs))

    def complete_until(self, system: str, messages: list[dict], done, *, rounds: int = 5, **kwargs) -> str:
        """complete(), continued while `done(text)` is false: long outputs (the debrief, init) can be cut by a
        server-side output cap (one local server stops every reply at 2,047 tokens and still reports
        finish_reason "stop"), so the model is asked to carry on from where it stopped."""
        text = self.complete(system, messages, **kwargs)
        for i in range(rounds):
            if done(text) or not text.strip():
                break
            print(f"\033[2m[the reply stopped early; asking the model to continue ({i + 1}/{rounds})]\033[0m", flush=True)
            more = self.complete(system, messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": "Your reply was cut off. Continue exactly where it stopped, mid-sentence "
                                            "if needed. Don't repeat anything or start over."}], **kwargs)
            if not more.strip():
                break
            head = re.sub(r"\W+", " ", more[:160]).strip().lower()
            if len(head) > 40 and head in re.sub(r"\W+", " ", text).lower():
                print("\033[2m[the model started over instead of continuing; keeping the first reply]\033[0m",
                      flush=True)
                break
            text += more                     # asked to continue mid-sentence, the model brings its own spacing
        return text


# --- Anthropic API -----------------------------------------------------------------------------

class AnthropicBackend:
    def __init__(self):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()

    def stream(self, system, messages, model, max_tokens, effort):
        anthropic = self.anthropic
        kwargs = dict(model=model, max_tokens=max_tokens, system=system, messages=messages,
                      output_config={"effort": effort})
        try:
            if model in FALLBACK_MODELS:
                # on a policy decline the API re-runs the request on a fallback model inside the same call
                ctx = self.client.beta.messages.stream(**kwargs, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                ctx = self.client.messages.stream(**kwargs)
            with ctx as stream:
                for text in stream.text_stream:
                    yield text
                final = stream.get_final_message()
        except anthropic.RateLimitError:
            yield APOLOGY_RATE
            return
        except anthropic.APIConnectionError:
            yield APOLOGY_CONN
            return
        except anthropic.AuthenticationError as e:
            raise LLMError(f"Anthropic API: authentication failed ({e}). Check ANTHROPIC_API_KEY.")
        yield (final.usage.input_tokens, final.usage.output_tokens)
        if final.stop_reason == "refusal":
            yield " Let's move on."


# --- claude -p (subscription auth; more tokens per turn) ---------------------------------------

class ClaudeCLIBackend:
    def __init__(self):
        if not shutil.which("claude"):
            raise LLMError("Provider claude-cli: the `claude` CLI is not installed or not on PATH.")

    def stream(self, system, messages, model, max_tokens, effort):
        cmd = [
            "claude", "-p", "--model", model, "--effort", effort, "--system-prompt", system,
            "--tools", "", "--strict-mcp-config", "--setting-sources", "",
            "--no-session-persistence", "--output-format", "stream-json",
            "--include-partial-messages", "--verbose",
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, text=True, cwd=os.path.expanduser("~"),
                                # long outputs (the debrief) would otherwise hit the CLI's default output cap
                                env={**os.environ, "CLAUDE_CODE_MAX_OUTPUT_TOKENS": str(max(max_tokens, 32000))})
        proc.stdin.write(_flatten(messages))
        proc.stdin.close()
        got_text = False
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("type") == "stream_event":
                e = ev.get("event", {})
                if e.get("type") == "content_block_delta" and e.get("delta", {}).get("type") == "text_delta":
                    got_text = True
                    yield e["delta"]["text"]
            elif ev.get("type") == "result":
                u = ev.get("usage", {})
                inp = u.get("input_tokens", 0) + u.get("cache_read_input_tokens", 0) + u.get("cache_creation_input_tokens", 0)
                if not got_text and ev.get("result"):
                    yield ev["result"]
                yield (inp, u.get("output_tokens", 0))
        proc.wait()


def _flatten(messages: list[dict]) -> str:
    """Render a message list as one prompt for the stateless CLI backend."""
    if len(messages) == 1:
        return messages[0]["content"]
    lines = ["Conversation so far (you are the interviewer):"]
    for m in messages[:-1]:
        who = "You" if m["role"] == "assistant" else "Candidate"
        lines.append(f"{who}: {m['content']}")
    lines.append("")
    lines.append(f"Latest candidate turn: {messages[-1]['content']}")
    lines.append("Write only your next spoken reply.")
    return "\n".join(lines)


# --- OpenAI-compatible chat completions (cloud and local) --------------------------------------

def _reachable(base_url: str, timeout: float = 0.5) -> bool:
    try:
        urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout).close()
        return True
    except Exception:
        return False


class OpenAICompatBackend:
    """Streams /chat/completions over plain HTTP (no extra dependency). Options per provider, under
    llm.providers.<name>: base_url, key_env (env var with the key), local, max_tokens_field,
    min_max_tokens, send_effort (map effort to `reasoning_effort`), extra_body (merged into requests)."""

    def __init__(self, name: str, p: dict):
        self.name, self.p = name, p
        if not p.get("base_url"):
            raise LLMError(f"llm.providers.{name}: needs a base_url, e.g. http://localhost:8000/v1")
        self.base = p["base_url"].rstrip("/")
        self.key = os.environ.get(p["key_env"], "") if p.get("key_env") else ""
        if p.get("key_env") and not self.key and not p.get("local"):
            raise LLMError(f"Provider {name}: set the {p['key_env']} environment variable to your API key.")
        self.no_usage = False                        # server rejected stream_options once: stop sending it

    def _req(self, path: str, body: dict | None = None, timeout: float = 600):
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        if self.name == "openrouter":
            headers["X-Title"] = "AInterviewer"
        data = json.dumps(body).encode() if body is not None else None
        return urllib.request.urlopen(urllib.request.Request(self.base + path, data=data, headers=headers),
                                      timeout=timeout)

    def _unreachable(self, e) -> LLMError:
        hint = self.p.get("hint", "check llm.providers." + self.name + ".base_url")
        return LLMError(f"Can't reach {self.name} at {self.base} ({getattr(e, 'reason', e)}): {hint}")

    def models(self) -> list[str]:
        try:
            with self._req("/models", timeout=5) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise LLMError(f"{self.name}: listing models failed: HTTP {e.code} {_err_text(e)}")
        except (urllib.error.URLError, OSError) as e:
            raise self._unreachable(e)
        return [m.get("id") for m in data.get("data", []) if m.get("id")]

    def pick_model(self, wanted: str) -> str:
        ids = self.models()
        if not ids:
            raise LLMError(f"{self.name} has no models available. {self.p.get('hint', '')}")
        if self.p.get("local"):
            print(f"\033[2m[{self.name}: using model {ids[0]}; set llm.model to choose another: "
                  f"{', '.join(ids[:8])}{' ...' if len(ids) > 8 else ''}]\033[0m", flush=True)
            return ids[0]
        raise LLMError(f"Provider {self.name}: set a model with --model or llm.model "
                       f"(e.g. {', '.join(ids[:10])}{' ...' if len(ids) > 10 else ''}).")

    def stream(self, system, messages, model, max_tokens, effort):
        if self.p.get("local") or self.p.get("min_max_tokens"):
            max_tokens = max(max_tokens, int(self.p.get("min_max_tokens", LOCAL_MIN_MAX_TOKENS)))
        body = {"model": model, "stream": True,
                "messages": [{"role": "system", "content": system}] + [
                    {"role": m["role"], "content": m["content"]} for m in messages],
                self.p.get("max_tokens_field", "max_tokens"): max_tokens}
        if not self.no_usage:
            body["stream_options"] = {"include_usage": True}
        if self.p.get("send_effort") and effort:
            body["reasoning_effort"] = {"max": "high", "xhigh": "high"}.get(effort, effort)
        body.update(self.p.get("extra_body") or {})
        try:
            resp = self._req("/chat/completions", body)
        except urllib.error.HTTPError as e:
            if e.code == 400 and "stream_options" in body:   # some servers reject it: retry once without
                self.no_usage = True
                yield from self.stream(system, messages, model, max_tokens, effort)
                return
            if e.code == 429 or e.code >= 500:
                yield APOLOGY_RATE
                return
            raise LLMError(f"{self.name} ({model}): HTTP {e.code} {_err_text(e)}")
        except (urllib.error.URLError, OSError) as e:
            if self.p.get("local"):
                raise self._unreachable(e)
            yield APOLOGY_CONN
            return
        usage, out_chars = None, 0
        strip = ThinkStripper()
        with resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    ev = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    text = (ch.get("delta") or {}).get("content")   # reasoning_content is never spoken
                    if text:
                        out_chars += len(text)
                        for piece in strip.feed(text):
                            yield piece
        tail = strip.flush()
        if tail:
            yield tail
        if usage:
            yield (usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0))
        else:   # no usage reported: rough estimate so the session footer isn't zero
            yield (sum(len(m["content"]) for m in body["messages"]) // 4, out_chars // 4)


def _err_text(e: urllib.error.HTTPError) -> str:
    try:
        body = json.loads(e.read())
        err = body.get("error", body)
        return (err.get("message") if isinstance(err, dict) else str(err))[:300]
    except Exception:
        return str(e.reason)


class ThinkStripper:
    """Drops <think>...</think> (and <thinking>) blocks that local reasoning models write into the reply,
    even when a tag is split across stream chunks, so the voice never reads the model's reasoning."""

    OPEN = re.compile(r"<think(?:ing)?>")
    CLOSE = re.compile(r"</think(?:ing)?>")
    HOLD = 11          # longest tag, "</thinking>"

    def __init__(self):
        self.buf, self.inside = "", False

    def feed(self, text: str):
        self.buf += text
        while True:
            if self.inside:
                m = self.CLOSE.search(self.buf)
                if not m:
                    self.buf = self.buf[-self.HOLD:]
                    return
                self.buf, self.inside = self.buf[m.end():].lstrip(), False
            else:
                m = self.OPEN.search(self.buf)
                if m:
                    if m.start():
                        yield self.buf[:m.start()]
                    self.buf, self.inside = self.buf[m.end():], True
                    continue
                cut = self.buf.rfind("<")
                keep = cut if cut >= 0 and len(self.buf) - cut < self.HOLD else len(self.buf)
                if keep:
                    yield self.buf[:keep]
                self.buf = self.buf[keep:]
                return

    def flush(self) -> str:
        out = "" if self.inside else self.buf
        self.buf = ""
        return out
