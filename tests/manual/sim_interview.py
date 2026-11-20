"""Simulates a full interview on a fake clock (each answer advances it 60-150 s) with the real LLM,
to check timing, panel turn-taking and question variety.
Run: AINTERVIEWER_PROCESS=<dir> INTERVIEWER_DATA_DIR=<tmp> .venv/bin/python -m tests.manual.sim_interview
The canned answers below are generic; they only need to keep the interview moving."""
import random, time
import interviewer.session as S
from interviewer import process as P
from interviewer.audio import Turn
from interviewer.llm import LLM

clock = [1000.0]
S.time.monotonic = lambda: clock[0]
cfg = P.load(P.find_dir(None)); PROFILE = P.profile(cfg); BANK = S.load_bank(cfg)
ANS = ["Sure. I am a backend engineer, two years on identity for five million users; I cut account creation from five minutes to two seconds.",
       "I led the change myself: I designed it, wrote the code and deployed it on AWS, and latency went from five minutes to two seconds.",
       "Honestly I'd say I pushed for it because nobody else would, and two other teams adopted it afterwards.",
       "I disagreed with the senior engineer, argued with numbers, lost one point and committed fully."]

class FakeIO(S.TextIO):
    def listen(self, max_seconds=None, silence_end=None, clock=""):
        secs = random.randint(60, 150)
        clock_now = S._fmt_clock(clock_[0] - 1000)
        clock_[0] += secs
        txt = random.choice(ANS) if self.phase_q() else "How do you measure whether your team's changes actually help customers?"
        print(f"\033[2m[{clock_now}] candidate answers for {secs}s\033[0m")
        return Turn(text=txt, speech_seconds=secs, ended_at=clock_[0])
clock_ = clock
iv = None
import re as _re
HANDOVER = _re.compile(r"questions? for (us|me)|any questions|what questions|you'd like to ask", _re.I)
FakeIO.phase_q = lambda self: not any(HANDOVER.search(e["text"]) for e in iv.transcript[len(panel):] if e["who"] != "you")
llm = LLM(cfg["llm"])
panel = S.pick_panel(cfg); print("PANEL:", [m["name"] for m in panel])
iv = S.Interview(cfg, FakeIO(), llm, BANK, PROFILE, panel)
iv.weak = {"themes": {}, "asked": {}}
iv.run()
print(f"\nEnded at {S._fmt_clock(clock[0]-1000)}; tokens {llm.totals()} over {len(llm.usage)} calls")
import os as _os
if _os.environ.get("SIM_DEBRIEF"):   # also run the real debrief call (uses llm.debrief_model)
    spoken, report, scores = S.debrief(cfg, llm, iv.transcript, BANK, PROFILE, "")
    print(f"\n--- DEBRIEF ---\n{spoken}\n\n{report}\n\nSCORES {scores}")
    print(f"debrief tokens: {llm.usage[-1]}")
