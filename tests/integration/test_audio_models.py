"""The real speech models, no microphone or speakers: Kokoro voices, Whisper accuracy on synthesized speech,
and the full listening pipeline (Silero VAD + chunked Whisper) on a spoken answer with a thinking pause.

Run: .venv/bin/python -m pytest -m audio      (needs models/ and downloads the Whisper model on first run)"""
import queue
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import yaml

import interviewer.audio as A
import interviewer.process as P

pytestmark = pytest.mark.audio

CFG = yaml.safe_load((P.ROOT / "config.yaml").read_text())


def words(text):
    return re.findall(r"[a-z0-9]+", text.lower())


@pytest.fixture(scope="module")
def kokoro():
    from kokoro_onnx import Kokoro
    tts = CFG["tts"]
    if not (P.ROOT / tts["model"]).exists():
        pytest.skip("Kokoro model files are missing (see README: Setup from scratch)")
    return Kokoro(str(P.ROOT / tts["model"]), str(P.ROOT / tts["voices"]))


def speak(kokoro, text, voice="am_michael"):
    a, sr = kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
    return np.interp(np.arange(0, len(a), sr / A.SR), np.arange(len(a)), a).astype(np.float32)


@pytest.fixture(scope="module")
def stt():
    return A.STT({**CFG["stt"], "vocabulary": ["Kafka", "Northwind"]})


@pytest.mark.parametrize("voice", P.VOICES)
def test_every_panel_voice_synthesizes(kokoro, voice):
    audio = speak(kokoro, "Tell me about a time you owned something end to end.", voice)
    secs = len(audio) / A.SR
    assert 1.5 < secs < 8 and np.abs(audio).max() > 0.01


def test_whisper_transcribes_synthesized_speech(kokoro, stt):
    said = "I moved the inventory check off the hot path and latency dropped from 900 to 250 milliseconds."
    heard = stt(speak(kokoro, said))
    overlap = len(set(words(said)) & set(words(heard))) / len(set(words(said)))
    assert overlap >= 0.8, heard


def test_vocabulary_words_are_spelled_right(kokoro, stt):
    heard = stt(speak(kokoro, "We migrated the Northwind payments queue to Kafka."))
    assert "Kafka" in heard and "Northwind" in heard, heard


@pytest.mark.parametrize("level", [0.0005, 0.001, 0.003])
def test_near_silence_is_not_a_hallucinated_answer(stt, level):
    assert stt(np.random.default_rng(0).normal(0, level, A.SR * 2).astype(np.float32)) == ""


def test_quiet_but_real_speech_is_still_heard(kokoro, stt):
    audio = speak(kokoro, "I owned the migration myself.") * 0.15          # a quiet speaker
    assert "migration" in stt(audio).lower()


def test_listening_pipeline_end_to_end(kokoro):
    """Speech, a 1.2 s thinking pause, more speech: one turn, both parts transcribed, ended by silence."""
    parts = [speak(kokoro, "So the achievement I'm proudest of is rewriting our account provisioning."),
             np.zeros(int(A.SR * 1.2), np.float32),
             speak(kokoro, "Account creation went from five minutes to about two seconds."),
             np.zeros(A.SR * 4, np.float32)]
    audio = np.concatenate([np.zeros(A.SR // 2, np.float32)] + parts)
    L = A.Listener.__new__(A.Listener)
    L.cfg = {**CFG["stt"], "silence_end_seconds": 2.0}
    L.keys, L.vad, L.stt = None, A.VAD(), A.STT(CFG["stt"])
    L.pool, L.q, L.active = ThreadPoolExecutor(1), queue.Queue(), False

    def feed():
        while not L.active:
            time.sleep(0.01)
        for i in range(0, len(audio) - A.FRAME, A.FRAME):
            L.q.put(audio[i:i + A.FRAME])
            time.sleep(A.FRAME / A.SR / 4)          # 4x real time
    threading.Thread(target=feed, daemon=True).start()
    turn = L.listen()
    w = words(turn.text)
    assert "provisioning" in w and ("seconds" in w or "two" in w), turn.text
    assert 6 < turn.speech_seconds < 14 and not turn.cut_off
