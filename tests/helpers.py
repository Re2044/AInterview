"""Shared test doubles: a scriptable fake LLM, a scripted candidate on a fake clock, a stand-in
OpenAI-compatible server, a fake `claude` CLI, and builders for process folders."""
from __future__ import annotations

import json
import os
import stat
import textwrap
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

import interviewer.session as S

ROOT = Path(__file__).resolve().parent.parent
CLI = ROOT / "ainterviewer"


# --- process folders ------------------------------------------------------------------------------

BASE_PROCESS = {
    "company": "Northwind",
    "role": "Backend Engineer",
    "round": "Hiring Manager",
    "round_description": "behavioural",
    "team": "the Payments team",
    "team_facts": "Payments builds the card platform.",
    "candidate_level": "Mid-level engineer, about 4 years.",
    "closing": "say the recruiter will be in touch about next steps",
    "panel": [{"role": "Engineering Manager", "opens": True,
               "people": [{"name": "Alex", "voice": "am_michael"}]}],
    "values": {"Ownership": {"brief": "own outcomes", "probe": "What did you do yourself?",
                             "red_flags": "blames others"}},
    "vocabulary": ["Northwind", "Kafka"],
    "profile": "profile.md",
    "notes": "",
    "questions": "questions.yaml",
    "interview": {"minutes": 30, "intro_skip_chance": 0},
}
BASE_QUESTIONS = {
    "extends_default": True,
    "questions": [
        {"id": "why_northwind", "theme": "motivation", "core": True, "q": ["Why Northwind?"]},
        {"id": "own_it", "theme": "ownership", "value": "Ownership", "core": True,
         "q": ["Tell me about something you owned end to end at your last job."]},
    ],
}
PROFILE = "Name: Sam.\nBackend engineer, 4 years. Cut checkout latency from 900 ms to 250 ms.\n"


def make_process(folder: Path, process: dict | None = None, questions_doc: dict | str | None = None,
                 profile_text: str | None = PROFILE, **overrides) -> Path:
    """Write a process folder: process.yaml (`process` replaces the base; keyword arguments patch any of its
    keys, e.g. questions="nope.yaml"), questions.yaml from `questions_doc` and profile.md from `profile_text`."""
    folder.mkdir(parents=True, exist_ok=True)
    proc = json.loads(json.dumps(process if process is not None else BASE_PROCESS))
    proc.update(overrides)
    (folder / "process.yaml").write_text(yaml.safe_dump(proc, sort_keys=False, allow_unicode=True))
    q = questions_doc if questions_doc is not None else BASE_QUESTIONS
    (folder / "questions.yaml").write_text(q if isinstance(q, str) else yaml.safe_dump(q, sort_keys=False))
    if profile_text is not None:
        (folder / "profile.md").write_text(profile_text)
    return folder


# --- fake LLM ----------------------------------------------------------------------------------

def default_reply(label: str, system: str, messages: list[dict]) -> str:
    """Replies in the shapes the real models produce, chosen from what the request asks for."""
    last = messages[-1]["content"] if messages else ""
    if label == "intro":                      # a well-behaved model asks the requested question word for word
        asked = last.split('ask exactly: "', 1)[1].split('"', 1)[0] if 'ask exactly: "' in last else "Tell me about yourself."
        return f"Hi, I'm Alex from Payments. About thirty minutes. {asked}"
    if label.startswith("q:"):
        return "[N] Next." if "Move on now [N]" in last else "[F] What was the result?"
    if label == "cand_q":
        return "[F] We ship weekly. Another question?" if "another question" in last else "[E] Thanks, goodbye."
    if label.startswith("drill"):
        return "Clear story. Add a number. Lead with the result.\nSCORE: 4"
    if label == "debrief":
        return ("SPOKEN:\nHire. Lead with results.\n\nREPORT:\n# Interviewer and system issues\nNone\n\n"
                "# Overview\n|q|s|\n\n# Answer by answer\n## 1. Q\nok\n\n# Patterns across the interview\nNone\n\n"
                'SCORES:\n{"intro": 4, "ownership": 3, "invented_theme": 5}')
    if label.startswith("debrief:"):
        if label == "debrief:1":
            return "SPOKEN:\nHire.\n\nREPORT:\n# Interviewer and system issues\nNone\n\n# Overview\n|a|"
        if label.startswith("debrief:q"):
            n = label[len("debrief:q"):]
            return f"## {n}. Question\nfeedback {n}"
        if label == "debrief:rest1":
            return "# Patterns across the interview\nNone"
        if label == "debrief:rest2":
            return '# Top 3 to practise\n1. x\n\nSCORES:\n{"intro": 3}'
        return '{"intro": 3}'
    return "OK."


class FakeLLM:
    """Stands in for interviewer.llm.LLM. `script(label, system, messages)` returns the reply text (or a list
    of chunks); every call is recorded in `calls`."""

    backend = "fake"

    def __init__(self, script=default_reply, local: bool = False):
        self.script, self.local = script, local
        self.calls: list[dict] = []
        self.usage: list[tuple[str, int, int]] = []

    def totals(self):
        return sum(u[1] for u in self.usage), sum(u[2] for u in self.usage)

    def is_local(self, spec=None):
        return self.local

    def stream(self, system, messages, *, label="", **kw):
        self.calls.append({"label": label, "system": system, "messages": [dict(m) for m in messages], **kw})
        out = self.script(label, system, messages)
        chunks = out if isinstance(out, list) else [out[i:i + 7] for i in range(0, len(out), 7)]
        self.usage.append((label, 10, len("".join(chunks)) // 4))
        yield from chunks

    def complete(self, *a, **k):
        return "".join(self.stream(*a, **k))

    def complete_until(self, system, messages, done, *, rounds=5, **k):
        return self.complete(system, messages, **k)

    def labels(self):
        return [c["label"] for c in self.calls]


# --- scripted candidate ------------------------------------------------------------------------

class ScriptedIO(S.TextIO):
    """A candidate on a fake clock. `answers` is a list of str or S.Turn (or a callable(n) -> str|Turn);
    after the list runs out it keeps giving `default`. Everything the interviewer says is in `said`."""

    can_resume = False

    def __init__(self, answers=None, seconds: float = 90, default="I built it myself and cut latency by 70%.",
                 clock: list[float] | None = None, max_turns: int = 500):
        super().__init__(None)
        self.answers, self.seconds, self.default = list(answers or []), seconds, default
        self.clock = clock if clock is not None else [0.0]
        self.said: list[tuple[str, str]] = []
        self.turns, self.max_turns = 0, max_turns
        self.listen_calls: list[dict] = []

    def say(self, chunks, after_turn=None, who=None):
        text = "".join(chunks)
        name = (who() if callable(who) else who or {}).get("name", "Interviewer")
        self.said.append((name, text))
        return text

    def listen(self, max_seconds=None, silence_end=None, clock="", wait_seconds=None):
        self.turns += 1
        self.listen_calls.append({"silence_end": silence_end, "wait_seconds": wait_seconds})
        if self.turns > self.max_turns:
            raise AssertionError("the interview did not end")
        self.clock[0] += self.seconds
        a = self.answers.pop(0) if self.answers else self.default
        if callable(a):
            a = a(self.turns)
        if isinstance(a, S.Turn):
            return a
        return S.Turn(text=a, speech_seconds=min(self.seconds, 80), ended_at=self.clock[0])


def run_interview(cfg, bank, profile="", llm=None, io=None, panel=None, seed=1):
    """Run a whole interview on a fake clock; returns (interview, io, llm)."""
    import random
    random.seed(seed)
    llm = llm or FakeLLM()
    io = io or ScriptedIO()
    iv = S.Interview(cfg, io, llm, bank, profile, panel or S.pick_panel(cfg))
    iv.clock = lambda: io.clock[0]
    iv.run()
    return iv, io, llm


# --- stand-in OpenAI-compatible server -----------------------------------------------------------

class FakeServer:
    """An OpenAI-compatible /v1 server on 127.0.0.1. `reply(body) -> str | (status, dict)` decides each
    chat completion; `requests` keeps every request body. `cap_chars` cuts every reply (like a server-side
    output cap that still reports finish_reason "stop"); `reject_stream_options` answers 400 to it."""

    def __init__(self, reply=None, models=("test-model",), chunk=5, cap_chars: int | None = None,
                 reject_stream_options=False, usage=True):
        self.reply = reply or (lambda body: "Hello.")
        self.models, self.chunk, self.cap = list(models), chunk, cap_chars
        self.reject_stream_options, self.usage = reject_stream_options, usage
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, status, obj):
                data = json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path.endswith("/models"):
                    self._json(200, {"data": [{"id": m} for m in outer.models]})
                else:
                    self._json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                outer.headers.append(dict(self.headers))
                if outer.reject_stream_options and "stream_options" in body:
                    return self._json(400, {"error": {"message": "unknown field stream_options"}})
                try:
                    out = outer.reply(body)
                except Exception as e:                 # a test script that ran out: answer, don't hang up
                    return self._json(500, {"error": {"message": f"test server: {type(e).__name__}: {e}"}})
                if isinstance(out, tuple):
                    return self._json(*out)
                if outer.cap is not None:
                    out = out[:outer.cap]
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                for i in range(0, len(out), outer.chunk):
                    ev = {"choices": [{"index": 0, "delta": {"content": out[i:i + outer.chunk]}}]}
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                fin = {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                if outer.usage and "stream_options" in body:
                    fin["usage"] = {"prompt_tokens": 100, "completion_tokens": len(out) // 4}
                self.wfile.write(f"data: {json.dumps(fin)}\n\ndata: [DONE]\n\n".encode())

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


INIT_OUTPUT = textwrap.dedent("""\
    === process.yaml ===
    company: Northwind
    role: Backend Engineer
    round: Hiring Manager
    team: the Payments team
    # TODO: ask the recruiter how long the round is
    team_facts: >-
      Payments builds the card platform.
    panel:
      - role: Engineering Manager
        opens: true
        people: [{name: Alex, voice: am_michael}, {name: Sara, voice: am_michael}]
    vocabulary: [Northwind, Kafka]
    profile: profile.md
    notes: ""
    questions: questions.yaml
    interview:
      minutes: 30
    === questions.yaml ===
    extends_default: true
    questions:
      - id: why_northwind
        theme: motivation
        core: true
        q: ["Why Northwind?", "What made you apply to Northwind?"]
    === profile.md ===
    Name: Sam.
    Backend engineer, 4 years. Cut checkout latency from 900 ms to 250 ms.
    === summary ===
    - Used the role and team. Missing: the round length.
    """)


def scripted_server_reply(body: dict) -> str:
    """What the stand-in server answers, chosen like default_reply but from the HTTP request alone."""
    system = body["messages"][0]["content"]
    last = body["messages"][-1]["content"]
    if "recruitment-process folder" in system:
        return INIT_OUTPUT
    if "expert interview coach" in system:
        if "Write exactly two sections" in last:
            return "SPOKEN:\nHire. Lead with the result.\n\nREPORT:\n# Interviewer and system issues\nNone\n\n# Overview\n|a|"
        if "Write only the feedback for main question" in last:
            n = last.split("main question ", 1)[1].split(",", 1)[0]
            return f"## {n}. Question\nfeedback"
        if "then the SCORES line" in last:
            return '# Top 3 to practise\n1. x\n\nSCORES:\n{"intro": 3, "motivation": 4}'
        if "Write only these parts" in last:
            return "# Patterns across the interview\nNone"
        return ('SPOKEN:\nHire.\n\nREPORT:\n# Overview\n|a|\n\nSCORES:\n{"intro": 3}')
    if "sharp interview coach" in system:
        return "<think>judge it</think>Clear story. Add a number.\nSCORE: 4"
    if "(call starts)" in last:
        asked = last.split('ask exactly: "', 1)[1].split('"', 1)[0] if 'ask exactly: "' in last else "Tell me about yourself."
        return f"Hi, I'm Alex from Payments. About thirty minutes. {asked}"
    if "answers the candidate's question" in last:
        return "[F] We ship weekly. Another question?" if "another question" in last else "[E] Thanks, goodbye."
    if "Move on now [N]" in last:
        return "[N] Moving on."
    return "<think>no number given</think>[F] What was the result?"


# --- fake `claude` CLI ---------------------------------------------------------------------------

def fake_claude(bin_dir: Path, text: str = "[F] How did you measure that?", log: Path | None = None) -> Path:
    """An executable `claude` that prints stream-json like `claude -p --output-format stream-json`."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = bin_dir / "claude"
    log = log or bin_dir / "claude-args.json"
    events = [{"type": "stream_event", "event": {"type": "content_block_delta",
                                                 "delta": {"type": "text_delta", "text": text[i:i + 6]}}}
              for i in range(0, len(text), 6)]
    events.append({"type": "result", "result": text, "usage": {"input_tokens": 12, "output_tokens": 7,
                                                               "cache_read_input_tokens": 3}})
    script.write_text("#!/usr/bin/env python3\nimport json, sys\n"
                      f"json.dump({{'argv': sys.argv[1:], 'stdin': sys.stdin.read()}}, open({str(log)!r}, 'w'))\n"
                      + "".join(f"print({json.dumps(json.dumps(e))})\n" for e in events))
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return log


def env_without_llm_keys() -> dict:
    return {k: v for k, v in os.environ.items() if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}
