"""Measures end-of-answer -> first interviewer audio, with real LLM + TTS but playback stubbed.
Run: AINTERVIEWER_PROCESS=<dir> .venv/bin/python -m tests.manual.latency"""
import os, time
import sounddevice as sd
sd.play = lambda a, sr: setattr(sd, "_dur", len(a) / sr)
sd.wait = lambda: time.sleep(getattr(sd, "_dur", 0))
from interviewer.audio import Speaker, Turn
from interviewer.llm import LLM
from interviewer.session import Interview, VoiceIO, load_bank, pick_panel, ROOT
from interviewer import process as P

cfg = P.load(P.find_dir(None)); profile = P.profile(cfg); bank = load_bank(cfg)
os.chdir(ROOT)
panel = pick_panel(cfg)
sp = Speaker(cfg["tts"], voices=[m["voice"] for m in panel]); llm = LLM(cfg["llm"])
io = VoiceIO(None, sp)
iv = Interview(cfg, io, llm, bank, profile, panel)
thread = [{"role": "user", "content": "(call starts)"},
          {"role": "assistant", "content": "Tell me about a time you disagreed with a more senior engineer."},
          {"role": "user", "content": "A senior wrote an exponential search for free usernames. I proposed binary search, it shipped."}]
for _ in range(2):
    turn = Turn(text="x", ended_at=time.monotonic()); io.ack(panel[0]["voice"])
    iv._turn(thread, "Ask ONE follow-up [F] or move on [N]: ask \"How do you prioritise?\"", "lat", {"F": panel[-1], "N": panel[0]}, after=turn)
    print(io.last_latency)
