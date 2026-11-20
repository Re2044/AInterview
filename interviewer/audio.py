"""Local audio: microphone + VAD turn detection + incremental Whisper, and Kokoro TTS playback.

No tokens are spent here. Half-duplex: the mic is ignored while the interviewer speaks.
"""
from __future__ import annotations

import os
import queue
import random
import re
import select
import sys
import termios
import threading
import time
import tty
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

SR = 16000
FRAME = 512  # 32 ms at 16 kHz, Silero's frame size
MIN_SPEECH_RMS = 0.004   # quieter than this is room noise (speech through a headset mic is ~0.02-0.2)

# Whisper drops fillers unless the prompt contains them; domain words (the process's `vocabulary`)
# improve accuracy.
STT_FILLERS = "Umm, so, uh, let me think, like, hmm."


def stt_prompt(vocabulary: list[str] | None = None) -> str:
    return STT_FILLERS + (" " + ", ".join(vocabulary) + "." if vocabulary else "")


def setup_devices(name: str = "auto"):
    """Route audio through PipeWire/Pulse (resamples, follows system default mic/speakers).
    Raw ALSA hw devices usually reject 16 kHz."""
    import sounddevice as sd

    names = [d["name"] for d in sd.query_devices()]
    if name and name != "auto":
        sd.default.device = name
        return
    for cand in ("pipewire", "pulse", "default"):
        if cand in names:
            sd.default.device = cand
            return


@dataclass
class Turn:
    text: str
    speech_seconds: float = 0.0     # first speech to last speech
    cut_off: bool = False           # hit max_answer_seconds
    by_key: bool = False            # ended with Space (the candidate says they're done)
    stt_tail: float = 0.0           # seconds from end of turn to final transcript
    ended_at: float = field(default_factory=time.monotonic)


class Keys:
    """Background reader for single keypresses (Space ends a turn). Ctrl-C still works."""

    def __init__(self):
        self.space = threading.Event()
        self._old = None
        if sys.stdin.isatty():
            fd = sys.stdin.fileno()
            self._old = termios.tcgetattr(fd)
            tty.setcbreak(fd)
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if r:
                ch = sys.stdin.read(1)
                if ch == " ":
                    self.space.set()

    def restore(self):
        if self._old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)
            self._old = None


class VAD:
    """Streaming Silero VAD (the ONNX model bundled with faster-whisper)."""

    def __init__(self):
        import faster_whisper
        import onnxruntime as ort

        path = os.path.join(os.path.dirname(faster_whisper.__file__), "assets", "silero_vad_v6.onnx")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = opts.intra_op_num_threads = 1
        opts.log_severity_level = 4
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=opts)
        self.reset()

    def reset(self):
        self.h = np.zeros((1, 1, 128), np.float32)
        self.c = np.zeros((1, 1, 128), np.float32)
        self.ctx = np.zeros(64, np.float32)

    def __call__(self, frame: np.ndarray) -> float:
        x = np.concatenate([self.ctx, frame])[None, :].astype(np.float32)
        p, self.h, self.c = self.sess.run(None, {"input": x, "h": self.h, "c": self.c})
        self.ctx = frame[-64:]
        return float(np.asarray(p).reshape(-1)[0])


class STT:
    def __init__(self, cfg: dict):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(cfg["model"], device="cpu", compute_type=cfg.get("compute_type", "int8"),
                                  cpu_threads=max(2, (os.cpu_count() or 4) - 2))
        self.prompt = stt_prompt(cfg.get("vocabulary"))

    def __call__(self, audio: np.ndarray) -> str:
        if len(audio) < SR * 0.3:
            return ""
        # Whisper invents words ("Yeah.", "Thank you.") for near-silence: don't transcribe what is barely there
        if float(np.sqrt(np.mean(np.square(audio)))) < MIN_SPEECH_RMS:
            return ""
        segs, _ = self.model.transcribe(audio, language="en", beam_size=1, vad_filter=False,
                                        condition_on_previous_text=False, initial_prompt=self.prompt,
                                        without_timestamps=True)
        # drop segments Whisper itself thinks are not speech and is unsure about
        text = " ".join(s.text.strip() for s in segs
                        if not (s.no_speech_prob > 0.6 and s.avg_logprob < -0.7)).strip()
        # Whisper hallucinates these on near-silence
        if text.lower().strip(" .!") in {"you", "thank you", "thanks for watching", "bye"}:
            return ""
        return text


class Listener:
    """Captures one spoken turn. Audio is transcribed in chunks at short pauses while you are
    still talking, so only the last few seconds remain to transcribe when you stop."""

    def __init__(self, cfg: dict, keys: Keys | None):
        import sounddevice as sd

        self.cfg = cfg
        self.keys = keys
        self.vad = VAD()
        self.stt = STT(cfg)
        self.pool = ThreadPoolExecutor(max_workers=1)  # ordered transcription
        self.q: queue.Queue = queue.Queue()
        self.active = False
        self.stream = sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=FRAME,
                                     callback=self._cb)
        self.stream.start()

    def _cb(self, indata, frames, t, status):
        if self.active:
            self.q.put(indata[:, 0].copy())

    def warmup(self):
        self.stt(np.zeros(SR, np.float32))

    def close(self):
        self.active = False
        self.stream.stop()
        self.stream.close()
        self.pool.shutdown(wait=False, cancel_futures=True)

    def mic_level(self, seconds: float = 0.6) -> float:
        """Peak input level over a short window; exactly 0 means a muted or dead mic."""
        while not self.q.empty():
            self.q.get_nowait()
        self.active = True
        time.sleep(seconds)
        self.active = False
        frames = []
        while not self.q.empty():
            frames.append(self.q.get_nowait())
        return float(np.abs(np.concatenate(frames)).max()) if frames else 0.0

    def listen(self, max_seconds: float | None = None, silence_end: float | None = None,
               on_start=None, wait_seconds: float | None = None) -> Turn:
        """Record one turn. `wait_seconds`: give up (empty Turn) if no speech starts within it."""
        answer = silence_end is None        # a custom silence_end means a short command ("next", "stop")
        silence_end = silence_end or self.cfg["silence_end_seconds"]
        chunk_pause = self.cfg["chunk_pause_seconds"]
        thr = self.cfg.get("vad_threshold", 0.5)
        start_silence = self.cfg.get("start_silence_seconds", 6.0) if answer else 0.0
        min_speech = self.cfg.get("min_speech_seconds", 4.0)
        # end_turn: space -> an answer ends only on Space (or the time limit), never on a pause.
        # Short commands (drill "next"/"stop") and setups without a keyboard still end on silence.
        manual = answer and self.keys is not None and self.cfg.get("end_turn", "space") == "space"
        fsec = FRAME / SR

        while not self.q.empty():
            self.q.get_nowait()
        if self.keys:
            self.keys.space.clear()
        self.vad.reset()
        self.active = True

        frames: list[np.ndarray] = []
        futures = []
        cut = 0                     # index of first frame not yet sent to STT
        started = None              # frame index of first speech
        last_speech = None          # frame index of last speech
        silence = 0.0
        cut_off = by_key = False
        pre_roll = int(0.3 / fsec)  # keep 300 ms before speech onset
        opened = time.monotonic()

        try:
            while True:
                try:
                    f = self.q.get(timeout=0.1)
                except queue.Empty:
                    f = None
                if self.keys and self.keys.space.is_set() and started is not None:
                    by_key = True
                    # keep the audio that arrived before Space but wasn't processed yet (the last words)
                    pending = ([f] if f is not None else []) + [self.q.get_nowait() for _ in range(self.q.qsize())]
                    for g in pending:
                        frames.append(g)
                        if self.vad(g) >= thr:
                            last_speech = len(frames) - 1
                    break
                if started is None and wait_seconds and time.monotonic() - opened >= wait_seconds:
                    break
                if f is None:
                    continue
                frames.append(f)
                i = len(frames) - 1
                speech = self.vad(f) >= thr
                if speech:
                    if started is None:
                        started = i
                        cut = max(0, i - pre_roll)
                        if on_start:
                            on_start()
                    last_speech = i
                    silence = 0.0
                elif started is not None:
                    silence += fsec
                    # send finished phrase for background transcription
                    if silence >= chunk_pause and last_speech >= cut and i - cut > int(1.0 / fsec):
                        futures.append(self.pool.submit(self.stt, np.concatenate(frames[cut:i + 1])))
                        cut = i + 1
                    # a pause right after "So," or "I-" is thinking, not the end of the answer:
                    # until there are a few seconds of speech, wait longer before ending the turn
                    spoken = (last_speech - started + 1) * fsec
                    if not manual and silence >= (silence_end if spoken >= min_speech
                                                  else max(silence_end, start_silence)):
                        break
                else:
                    # discard leading silence beyond the pre-roll
                    if len(frames) > pre_roll * 4:
                        drop = len(frames) - pre_roll
                        frames = frames[drop:]
                if started is not None and max_seconds and (i - started) * fsec >= max_seconds:
                    cut_off = True
                    break
        finally:
            self.active = False

        end_t = time.monotonic()
        if started is not None and cut < len(frames) and last_speech is not None and last_speech >= cut:
            futures.append(self.pool.submit(self.stt, np.concatenate(frames[cut:])))
        text = " ".join(t for t in (fu.result() for fu in futures) if t).strip()
        span = ((last_speech or 0) - (started or 0) + 1) * fsec if started is not None else 0.0
        return Turn(text=text, speech_seconds=span, cut_off=cut_off, by_key=by_key,
                    stt_tail=time.monotonic() - end_t)


# --- Text to speech ---------------------------------------------------------------------

_SENT_END = re.compile(r"(?<=[.!?])\s+")


def clean_for_speech(s: str) -> str:
    s = re.sub(r"[*_#`>\[\]]", "", s)
    s = s.replace("—", ", ").replace("–", ", ")
    s = re.sub(r"[^\x00-\u024F\u2018-\u201D\u2026]", "", s)   # drop stray non-Latin characters
    return s.strip()


class Speaker:
    """Kokoro TTS with several voices. Sentences are synthesized and played as the LLM streams them."""

    def __init__(self, cfg: dict, voices: list[str] | None = None):
        import sounddevice as sd
        from kokoro_onnx import Kokoro

        self.sd = sd
        self.cfg = cfg
        self.tts = Kokoro(cfg["model"], cfg["voices"])
        self.voice = cfg["voice"]                      # default voice (drill coach)
        self.speed = cfg.get("speed", 1.0)
        self.play_q: queue.Queue = queue.Queue()
        self.synth_q: queue.Queue = queue.Queue()
        self.first_audio_at: float | None = None
        self._idle = threading.Event()
        self._idle.set()
        self.acks = {v: [self._synth(a, v) for a in cfg.get("acknowledgements", [])]
                     for v in dict.fromkeys((voices or []) + [self.voice])}
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def _synth(self, text: str, voice: str):
        samples, sr = self.tts.create(text, voice=voice, speed=self.speed, lang="en-us")
        return np.asarray(samples, np.float32), sr

    def _synth_loop(self):
        while True:
            item = self.synth_q.get()
            if item is None:
                self.play_q.put(None)
                continue
            text, voice = item
            text = clean_for_speech(text)
            if text:
                self.play_q.put(("speech", *self._synth(text, voice)))

    def _play_loop(self):
        while True:
            item = self.play_q.get()
            if item is None:            # end-of-utterance marker
                self._idle.set()
                continue
            kind, audio, sr = item
            if kind == "speech" and self.first_audio_at is None:
                self.first_audio_at = time.monotonic()
            self.sd.play(audio, sr)
            self.sd.wait()

    def ack(self, voice: str | None = None):
        acks = self.acks.get(voice or self.voice) or self.acks.get(self.voice)
        if acks:
            self._idle.clear()
            audio, sr = random.choice(acks)
            self.play_q.put(("ack", audio, sr))

    def say_stream(self, chunks, echo=None, voice=None) -> str:
        """Speak streamed text sentence by sentence. Blocks until playback finishes.
        `voice` may be a callable, resolved when the first sentence is queued (so a reply's
        speaker can depend on the tag at the start of the stream)."""
        self._idle.clear()
        self.first_audio_at = None
        v = lambda: (voice() if callable(voice) else voice) or self.voice
        buf, full = "", ""
        for ch in chunks:
            full += ch
            buf += ch
            if echo:
                echo(ch)
            parts = _SENT_END.split(buf)
            for sent in parts[:-1]:
                self.synth_q.put((sent, v()))
            buf = parts[-1]
            # get the first words out fast: speak the first clause without waiting for the full sentence
            if full.strip() and full.count(" ") < 25 and len(buf) >= 25 and ", " in buf[12:] \
                    and self.synth_q.empty() and self.play_q.qsize() <= 1 and self.first_audio_at is None:
                head, buf = buf.rsplit(", ", 1)
                self.synth_q.put((head + ",", v()))
        if buf.strip():
            self.synth_q.put((buf, v()))
        self.synth_q.put(None)
        self._idle.wait()
        time.sleep(0.15)  # let the room echo die before the mic opens
        return full

    def say(self, text: str, echo=None, voice=None) -> str:
        return self.say_stream([text], echo, voice)
