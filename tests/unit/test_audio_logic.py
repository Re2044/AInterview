"""interviewer.audio without models or devices: text cleanup, the STT prompt, the speaker's sentence streaming
(fake Kokoro, stubbed playback) and the listener's turn-taking (fake VAD and STT fed synthetic frames)."""
import queue
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import interviewer.audio as A


@pytest.mark.parametrize("text, want", [("**Bold** and _it_", "Bold and it"), ("# Heading", "Heading"),
                                        ("Wait — then", "Wait ,  then"), ("Emoji 😀 gone", "Emoji  gone"),
                                        ("Café naïve", "Café naïve"), ("[F] tag", "F tag")])
def test_clean_for_speech(text, want):
    assert A.clean_for_speech(text) == want


def test_stt_prompt_has_fillers_and_vocabulary():
    assert A.stt_prompt(["Revolut", "ECS"]) == A.STT_FILLERS + " Revolut, ECS."
    assert A.stt_prompt([]) == A.STT_FILLERS and A.stt_prompt(None) == A.STT_FILLERS


# --- speaker -----------------------------------------------------------------------------------------------

class FakeKokoro:
    def __init__(self, *a, **k):
        self.made = []

    def create(self, text, voice, speed, lang):
        self.made.append((text, voice))
        return np.zeros(240, np.float32), 24000


@pytest.fixture
def speaker(monkeypatch):
    played = []
    sd = types.SimpleNamespace(play=lambda a, sr: played.append(len(a)), wait=lambda: None)
    monkeypatch.setitem(sys.modules, "sounddevice", sd)
    monkeypatch.setitem(sys.modules, "kokoro_onnx", types.SimpleNamespace(Kokoro=FakeKokoro))
    monkeypatch.setattr(A.time, "sleep", lambda s: None)
    sp = A.Speaker({"model": "m", "voices": "v", "voice": "af_heart", "speed": 1.0, "acknowledgements": ["Mm-hm."]},
                   voices=["bm_george"])
    sp.played = played
    return sp


class TestSpeaker:
    def test_acknowledgements_are_prepared_per_voice(self, speaker):
        assert set(speaker.acks) == {"bm_george", "af_heart"}

    def test_speaks_sentence_by_sentence_in_order(self, speaker):
        speaker.tts.made.clear()
        text = speaker.say_stream(["Hello there. How are", " you today? Fine."], voice="bm_george")
        assert text == "Hello there. How are you today? Fine."
        assert [t for t, _ in speaker.tts.made] == ["Hello there.", "How are you today?", "Fine."]
        assert {v for _, v in speaker.tts.made} == {"bm_george"}

    def test_voice_can_be_decided_late(self, speaker):
        speaker.tts.made.clear()
        chosen = {"v": None}
        chunks = iter(["One sentence. ", "Two."])

        def gen():
            chosen["v"] = "bm_george"
            yield from chunks
        speaker.say_stream(gen(), voice=lambda: chosen["v"])
        assert {v for _, v in speaker.tts.made} == {"bm_george"}

    def test_markdown_is_not_read_aloud(self, speaker):
        speaker.tts.made.clear()
        speaker.say("**Great** question.")
        assert speaker.tts.made == [("Great question.", "af_heart")]

    def test_echo_receives_every_chunk(self, speaker):
        seen = []
        speaker.say_stream(["a ", "b."], echo=seen.append)
        assert seen == ["a ", "b."]

    def test_empty_text_returns_without_hanging(self, speaker):
        assert speaker.say_stream([]) == ""


# --- listener ----------------------------------------------------------------------------------------------

SPEECH, SILENCE = 1.0, 0.0


def listener(cfg=None, keys=None):
    L = A.Listener.__new__(A.Listener)
    L.cfg = {"silence_end_seconds": 1.0, "chunk_pause_seconds": 0.3, "vad_threshold": 0.5,
             "start_silence_seconds": 2.0, "min_speech_seconds": 1.0, "end_turn": "space", **(cfg or {})}
    L.keys, L.q, L.active = keys, queue.Queue(), False
    L.vad = lambda f: float(f[0])                         # frames carry their own "speech" flag
    L.vad.reset = lambda: None
    L.stt = lambda audio: f"<{len(audio) // A.FRAME} frames>"
    L.pool = ThreadPoolExecutor(1)
    return L


def frames(*segments):
    """segments: (seconds, speech?) -> list of frames"""
    out = []
    for secs, value in segments:
        out += [np.full(A.FRAME, value, np.float32)] * int(secs * A.SR / A.FRAME)
    return out


def feed(L, fs, then=None):
    def run():
        while not L.active:
            time.sleep(0.001)
        for f in fs:
            L.q.put(f)
        if then:
            then()
    threading.Thread(target=run, daemon=True).start()


class TestListener:
    def test_ends_on_silence_without_keyboard(self):
        L = listener(keys=None)
        feed(L, frames((0.5, SILENCE), (2.0, SPEECH), (1.5, SILENCE)))
        t = L.listen()
        assert t.text and 1.8 <= t.speech_seconds <= 2.2 and not t.cut_off and not t.by_key

    def test_space_mode_ignores_long_pauses_until_space(self):
        keys = types.SimpleNamespace(space=threading.Event())
        L = listener(keys=keys)

        def press_when_heard():
            while not L.q.empty():
                time.sleep(0.001)
            keys.space.set()
        feed(L, frames((1.0, SPEECH), (3.0, SILENCE), (1.0, SPEECH), (0.5, SILENCE)), then=press_when_heard)
        t = L.listen()
        assert t.by_key and t.speech_seconds > 4

    def test_space_keeps_audio_still_queued(self):
        keys = types.SimpleNamespace(space=threading.Event())
        L = listener(keys=keys)
        feed(L, frames((0.5, SPEECH), (1.0, SPEECH)), then=keys.space.set)   # Space while frames are queued
        t = L.listen()
        assert t.by_key and t.speech_seconds > 1.4

    def test_short_start_is_not_taken_as_the_whole_answer(self):
        L = listener(keys=None)
        # 0.3 s of "So," then a 1.5 s pause (shorter than start_silence) then the real answer
        feed(L, frames((0.3, SPEECH), (1.5, SILENCE), (2.0, SPEECH), (1.5, SILENCE)))
        t = L.listen()
        assert t.speech_seconds > 3

    def test_time_limit_cuts_off(self):
        L = listener(keys=None)
        feed(L, frames((5.0, SPEECH)))
        t = L.listen(max_seconds=2)
        assert t.cut_off and t.speech_seconds <= 2.2

    def test_wait_seconds_gives_up_without_speech(self):
        L = listener(keys=None)
        feed(L, frames((0.2, SILENCE)))
        t = L.listen(wait_seconds=0.5)
        assert t.text == "" and t.speech_seconds == 0

    def test_command_listening_ends_on_short_pause_even_in_space_mode(self):
        keys = types.SimpleNamespace(space=threading.Event())
        L = listener(keys=keys)
        feed(L, frames((0.6, SPEECH), (1.0, SILENCE)))
        t = L.listen(silence_end=0.8)
        assert t.text and not t.by_key

    def test_long_answers_are_transcribed_in_chunks(self):
        L = listener(keys=None)
        feed(L, frames((2.0, SPEECH), (0.5, SILENCE), (2.0, SPEECH), (1.5, SILENCE)))
        t = L.listen()
        assert t.text.count("frames>") >= 2
