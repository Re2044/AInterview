"""Dry run of a process folder: a whole interview, a drill round and both debrief layouts, with a scripted
candidate, a fake clock and a fake model. No tokens, no audio, nothing written to the real data folders.

It proves a process is technically sound (it loads, every prompt can be built, the interview reaches its
end in a bounded number of turns, the debrief parses), not that its content is good. `init` runs it before
writing anything; `ainterviewer check` runs it on any folder.
"""
from __future__ import annotations

import contextlib
import copy
import io
import tempfile
from pathlib import Path

from . import session as S

MAX_TURNS = 400              # a 4-hour interview with every limit maxed is far below this


class _FakeLLM:
    """Answers in the shapes the real interviewer and coach produce, alternating follow-ups and move-ons."""
    backend = "dry-run"

    def __init__(self):
        self.usage, self.n = [], 0

    def totals(self):
        return 0, 0

    def is_local(self, spec=None):
        return False

    def stream(self, system, messages, *, label="", **kw):
        if not system or not messages or not all(isinstance(m.get("content"), str) for m in messages):
            raise AssertionError(f"malformed request for {label}")
        self.n += 1
        if label == "intro":
            yield "Hello, I'm the interviewer. We have some time today. Tell me about yourself."
        elif label.startswith("q:"):
            yield "[F] What was the result?" if self.n % 3 else "[N] Next question."
        elif label == "cand_q":
            yield "[F] Good question; it depends on the quarter. Anything else?" if self.n % 2 else "[E] Thanks, goodbye."
        elif label.startswith("drill"):
            yield "Clear story. Add a number. Lead with the result.\nSCORE: 3"
        elif label.startswith("debrief"):
            yield ('SPOKEN:\nHire.\n\nREPORT:\n# Interviewer and system issues\nNone\n\n## 1. Question\nok\n\n'
                   '# Patterns across the interview\nNone\n\nSCORES:\n{"intro": 3}')
        else:
            yield "OK."

    def complete(self, *a, **k):
        return "".join(x for x in self.stream(*a, **k) if isinstance(x, str))


class _ScriptedIO(S.TextIO):
    """A candidate who answers everything in 90 s, asks two questions at the end and says "stop" to the drill."""

    def __init__(self, clock: list[float]):
        super().__init__(None)
        self.clock, self.turns = clock, 0

    def listen(self, max_seconds=None, silence_end=None, clock="", wait_seconds=None):
        self.turns += 1
        if self.turns > MAX_TURNS:
            raise AssertionError(f"the interview did not end within {MAX_TURNS} turns")
        self.clock[0] += 90
        if silence_end:                                       # drill: "again, next or stop?"
            return S.Turn(text="stop", ended_at=self.clock[0])
        return S.Turn(text="I built the service myself and cut latency from 900 ms to 250 ms; we shipped it in "
                           "two weeks.", speech_seconds=80, ended_at=self.clock[0])


def dry_run(cfg: dict, bank: dict, profile: str, notes: str = "") -> str:
    """Raises (with the reason) if the process can't run; returns a one-line summary."""
    from . import process as P
    saved = (S.WEAK_FILE, S.SESS_DIR)
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        S.WEAK_FILE, S.SESS_DIR = Path(tmp) / "weak_spots.json", Path(tmp) / "sessions"
        try:
            clock = [0.0]
            cfg = copy.deepcopy(cfg)
            bank = copy.deepcopy(bank)
            panel = S.pick_panel(cfg)
            iv = S.Interview(cfg, _ScriptedIO(clock), _FakeLLM(), bank, profile, panel)
            iv.clock = lambda: clock[0]
            iv.run()
            ended = clock[0]
            answers = sum(e["who"] == "you" for e in iv.transcript)
            if answers < 2:
                raise AssertionError("the interview ended before any real question was answered")
            for mode in ("single", "sections"):
                c = dict(cfg, llm=dict(cfg["llm"], debrief_mode=mode))
                spoken, report, scores = S.debrief(c, _FakeLLM(), iv.transcript, bank, profile, notes)
                if not (spoken and report):
                    raise AssertionError(f"the debrief ({mode}) could not be parsed")
            log: list[dict] = []
            S.run_drill(cfg, _ScriptedIO(clock), _FakeLLM(), bank, profile, log)
            S.drill_system(cfg, profile)
            S.save_session("interview", iv.transcript, report, _FakeLLM())
            P.notes(cfg)
        finally:
            S.WEAK_FILE, S.SESS_DIR = saved
    mains = len({e.get("q") for e in iv.transcript if e.get("q")})
    return f"dry run ok: {mains} questions, {answers} answers, ended at {S._fmt_clock(ended)}"
