"""Feeds a synthesized spoken answer (instead of the mic) through VAD + incremental Whisper.
Run: AINTERVIEWER_PROCESS=<dir> .venv/bin/python -m tests.manual.sim_listen"""
import os, threading, time
import numpy as np
from interviewer.audio import Listener, VAD, STT, FRAME, SR
from kokoro_onnx import Kokoro
import queue
from concurrent.futures import ThreadPoolExecutor

from interviewer import process as P
from interviewer.session import ROOT
cfg = P.load(P.find_dir(None)); os.chdir(ROOT)
k = Kokoro(cfg["tts"]["model"], cfg["tts"]["voices"])
ANSWER = ("So, the achievement I'm most proud of is rewriting our account provisioning. "
          "It used to take about five minutes. I made it synchronous, and now it takes about two seconds.|"
          "I owned the design, the code and the AWS deployment, which nobody on the team had done before.")
parts = []
t = time.monotonic()
for i, s in enumerate(ANSWER.split("|")):
    a, sr = k.create(s, voice="am_michael", speed=1.0, lang="en-us")
    parts.append(np.interp(np.arange(0, len(a), sr / SR), np.arange(len(a)), a).astype(np.float32))
    parts.append(np.zeros(int(SR * 1.0), np.float32))  # 1 s thinking pause mid-answer
print(f"TTS for {sum(len(p) for p in parts)/SR:.1f}s of audio took {time.monotonic()-t:.2f}s")
audio = np.concatenate([np.zeros(SR//2, np.float32)] + parts + [np.zeros(SR*4, np.float32)])

L = Listener.__new__(Listener)
L.cfg, L.keys, L.vad = cfg["stt"], None, VAD()
t = time.monotonic(); L.stt = STT(cfg["stt"]); print(f"Whisper load {time.monotonic()-t:.2f}s")
L.pool, L.q, L.active = ThreadPoolExecutor(1), queue.Queue(), False
L.warmup()

def feed():
    while not L.active: time.sleep(0.01)
    for i in range(0, len(audio) - FRAME, FRAME):
        L.q.put(audio[i:i+FRAME]); time.sleep(FRAME / SR)  # real time
threading.Thread(target=feed, daemon=True).start()
turn = L.listen()
print("TEXT:", turn.text)
print(f"speech_seconds={turn.speech_seconds:.1f} stt_tail={turn.stt_tail:.2f}s cut_off={turn.cut_off}")
