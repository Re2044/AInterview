"""Interview and drill flows. The code drives the structure (which question, how many
follow-ups, timing); the model only phrases the next 1-2 sentences."""
from __future__ import annotations

import datetime as dt
import json
import os
import random
import re
import time
from collections import Counter
from pathlib import Path


from .audio import Turn
from .metrics import answer_metrics, format_metrics

ROOT = Path(__file__).resolve().parent.parent
# Sessions and weak spots live in the process folder (set by use_data_dir); tests point
# INTERVIEWER_DATA_DIR elsewhere so they never touch real data.
WEAK_FILE = ROOT / "weak_spots.json"
SESS_DIR = ROOT / "sessions"


def use_data_dir(path: Path) -> None:
    global WEAK_FILE, SESS_DIR
    data = Path(os.environ.get("INTERVIEWER_DATA_DIR") or path)
    WEAK_FILE, SESS_DIR = data / "weak_spots.json", data / "sessions"


if os.environ.get("INTERVIEWER_DATA_DIR"):
    use_data_dir(ROOT)


# --- I/O adapters --------------------------------------------------------------------------

def _resolve(who):
    return (who() if callable(who) else who) or {}


def _speak(speaker, chunks, after_turn, who, io) -> str:
    """Print and (optionally) speak a reply. `who` is a panel member dict or a callable returning
    one; it is resolved at the first chunk, after the [F]/[N]/[E] tag has been read."""
    t0 = time.monotonic()
    first_tok = [None]

    def echo(ch):
        if first_tok[0] is None:
            first_tok[0] = time.monotonic()
            name = _resolve(who).get("name", "Interviewer")
            print(f"\n\033[1;36m{name}:\033[0m ", end="", flush=True)
        print(ch, end="", flush=True)

    if speaker:
        text = speaker.say_stream(chunks, echo=echo, voice=lambda: _resolve(who).get("voice"))
    else:
        text = ""
        for c in chunks:
            text += c
            echo(c)
    print()
    if speaker and after_turn is not None and first_tok[0] and speaker.first_audio_at:
        io.last_latency = {
            "stt_tail": round(after_turn.stt_tail, 2),
            "llm_first_token": round(first_tok[0] - t0, 2),
            "first_audio_after_you_stopped": round(speaker.first_audio_at - after_turn.ended_at, 2),
        }
    return text


class VoiceIO:
    can_resume = True       # can keep listening after a turn that ended mid-sentence

    def __init__(self, listener, speaker):
        self.listener, self.speaker = listener, speaker
        self.last_latency: dict = {}

    def say(self, chunks, after_turn: Turn | None = None, who=None) -> str:
        return _speak(self.speaker, chunks, after_turn, who, self)

    def listen(self, max_seconds=None, silence_end=None, clock: str = "", wait_seconds=None) -> Turn:
        clock = f"{clock} · " if clock else ""
        print(f"\033[2m[{clock}{'still ' if wait_seconds else ''}listening... press Space when you've finished]\033[0m", flush=True)
        turn = self.listener.listen(max_seconds=max_seconds, silence_end=silence_end, wait_seconds=wait_seconds,
                                    on_start=lambda: print("\033[2m[recording]\033[0m", flush=True))
        if turn.text:
            print(f"\033[1;33mYou:\033[0m {turn.text}")
        return turn

    def ack(self, voice=None):
        self.speaker.ack(voice)


class TextIO:
    """Type your answers. Optionally still speaks the interviewer's lines."""

    def __init__(self, speaker=None):
        self.speaker = speaker
        self.last_latency: dict = {}

    def say(self, chunks, after_turn=None, who=None) -> str:
        return _speak(self.speaker, chunks, after_turn, who, self)

    def listen(self, max_seconds=None, silence_end=None, clock: str = "", wait_seconds=None) -> Turn:
        t0 = time.monotonic()
        clock = f"\033[2m[{clock}]\033[0m " if clock else ""
        text = input(f"{clock}\033[1;33mYou:\033[0m ").strip()
        # typing time is not speaking time; estimate speech at 140 wpm
        return Turn(text=text, speech_seconds=max(1.0, len(text.split()) / 140 * 60),
                    ended_at=time.monotonic(), stt_tail=0.0) if text else Turn(text="", ended_at=t0)

    def ack(self, voice=None):
        pass


# --- helpers -------------------------------------------------------------------------------

TAG = re.compile(r"^\s*\[([FNE])\]\s*")
LATE_TAG = re.compile(r"\[([FNE])\]\s*")
# Speaker labels the model sometimes adds despite the prompt: "Kate:", "Kate here.", "Emma here, ..."
LABEL = re.compile(r"^\s*(?:[A-Z][a-z]+(?:,[^:.]{0,40})?:\s+|[A-Z][a-z]+ here[.,]?\s+)")
# An opening evaluation ("That sounds efficient.", "Great.") that smaller models add despite the prompt.
# Closings ("That's all from my side", "Thank you for your time") are not evaluations and stay.
PRAISE = re.compile(r"^(?!(?:that's|that is) all\b|thanks? (?:you )?for (?:your|the) time)"
                    r"(?:(?:that|this|it)(?:'s| is| was| sounds| seems| must have been)\s[^.!?]{0,50}"
                    r"|(?:great|good|nice|excellent|impressive|interesting|wonderful|awesome|perfect|fantastic)"
                    r"\b[^.!?]{0,40}|(?:thanks?|thank you)(?: for (?:sharing|that|explaining)[^.!?]{0,30})?)[.!]\s+",
                    re.I)


class Tagged:
    """Wraps an LLM stream: strips the leading [F]/[N]/[E] tag (remembered in .tag), any speaker label and
    an opening evaluation, buffering only the first ~60 characters. `replace` maps a tag to fixed text: once
    the model has chosen that tag, the fixed text is spoken instead of the model's wording (the next
    question is always asked exactly as written, whatever the model would have said)."""

    def __init__(self, chunks, name: str = "", replace: dict | None = None):
        self.chunks, self.tag, self.replace = chunks, "F", replace or {}
        # also drop a leading "<candidate name>, " (models overuse it); "Thanks, Sam" stays
        self.name_re = re.compile(rf"^\s*{re.escape(name)},\s*") if name else None

    def _clean(self, s: str) -> str:
        s = LABEL.sub("", s.lstrip(), count=1)
        if self.name_re and self.name_re.match(s):
            s = self.name_re.sub("", s, count=1)
            s = s[:1].upper() + s[1:]
        rest = PRAISE.sub("", s, count=1)
        if rest.strip() and rest != s:
            s = rest[:1].upper() + rest[1:]
        return s

    def __iter__(self):
        buf, done, found = "", False, False
        tail = ""                                   # after the first tag: held back while a later tag may be arriving
        for ch in self.chunks:
            if done:
                # A later tag ("...answer. [E] Thanks, goodbye.") is never spoken; the last tag decides the action.
                tail += ch
                for m in LATE_TAG.finditer(tail):
                    self.tag = m.group(1)
                tail = LATE_TAG.sub("", tail)
                cut = tail.rfind("[")
                keep = len(tail) - cut if cut >= 0 and len(tail) - cut < 3 else 0
                if len(tail) > keep:
                    yield tail[:len(tail) - keep]
                    tail = tail[len(tail) - keep:]
                continue
            buf += ch
            if not found:
                m = TAG.match(buf)
                if m:
                    found, self.tag, buf = True, m.group(1), buf[m.end():]
                elif buf.lstrip().startswith("[") and len(buf) < 6:
                    continue  # tag still arriving
                else:
                    # The model sometimes writes a note to itself before the tag ("The answer was
                    # general, so ask for an example. [F] Can you..."): never speak that part.
                    late = LATE_TAG.search(buf)
                    if late:
                        found, self.tag, buf = True, late.group(1), buf[late.end():]
                    elif len(buf) < 300:
                        continue  # no tag yet: hold back a little longer in case one follows
            if found and self.tag in self.replace:
                if hasattr(self.chunks, "close"):
                    self.chunks.close()                 # stop generating: its wording is not used
                yield self.replace[self.tag]
                return
            if len(buf) >= 60 or (not found and len(buf) >= 300):
                done = True
                for m in LATE_TAG.finditer(buf):        # a second tag already in the first part
                    self.tag = m.group(1)
                buf = LATE_TAG.sub("", buf)
                cut = buf.rfind("[")
                if cut >= 0 and len(buf) - cut < 3:      # a tag may be arriving: hold its start back
                    buf, tail = buf[:cut], buf[cut:]
                yield self._clean(buf)
        if done and tail:
            yield tail
        if not done and buf:
            for m in LATE_TAG.finditer(buf):           # a short reply with a tag after the first one
                self.tag = m.group(1)
            yield self._clean(LATE_TAG.sub("", buf))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


_STOP = set("a an the you your yours me my i we our us to of in on at for with about tell what how why when "
            "did do does have has had was were is are be that this it can could would time give".split())


def asks_same(question: str, text: str) -> bool:
    """True when `text` asks `question` in other words ("Do you have a different example of a project that
    didn't go well?" vs "Tell me about a project that didn't go well.")."""
    a = {w for w in _norm(question).split() if w not in _STOP and len(w) > 2}
    b = {w for w in _norm(text).split() if w not in _STOP and len(w) > 2}
    shared = a & b
    # most of the next question's own content words, and at least all of them up to three
    return len(a) >= 2 and len(shared) >= min(3, len(a)) and len(shared) / len(a) >= 0.6


def load_weak() -> dict:
    try:
        return json.loads(WEAK_FILE.read_text())
    except (OSError, json.JSONDecodeError):
        return {"themes": {}, "asked": {}}


def save_weak(w: dict):
    WEAK_FILE.parent.mkdir(parents=True, exist_ok=True)
    WEAK_FILE.write_text(json.dumps(w, indent=2))


def update_weak(w: dict, theme: str, score: float):
    """Moving average of 1-5 scores per theme."""
    t = w["themes"].get(theme)
    w["themes"][theme] = round(score if t is None else 0.6 * t + 0.4 * score, 2)


def weight(q: dict, weak: dict) -> float:
    score = weak["themes"].get(q["theme"], 3.0)          # unknown = average
    asked = weak["asked"].get(q["id"], 0)
    return (1.0 + max(0.0, 4.0 - score) * 1.5) / (1 + asked)


def pick_questions(bank: list[dict], n: int, weak: dict, extra_chance: float = 0.4) -> list[dict]:
    """Core topics first, one question per theme, weak themes more likely to come early. Themes with
    process-specific questions (`own`, from the process's questions.yaml) come before the generic bank's,
    and within a theme its own questions win. One motivation question goes in at a random early point;
    sometimes one off-script extra."""
    w = lambda qs: [weight(q, weak) for q in qs]
    if not any(q.get("core") for q in bank):          # a bank without core questions: every question is core
        bank = [dict(q, core=True) for q in bank]
    themes: dict[str, list] = {}
    for q in bank:
        if q.get("core") and q["theme"] != "motivation":
            themes.setdefault(q["theme"], []).append(q)
    own = [t for t in themes if any(q.get("own") for q in themes[t])]
    order = []
    for pool in (own, [t for t in themes if t not in own]):
        while pool:
            t = random.choices(pool, [max(w(themes[x])) for x in pool])[0]
            pool.remove(t)
            cands = [q for q in themes[t] if q.get("own")] or themes[t]
            order.append(random.choices(cands, w(cands))[0])
    motiv = [q for q in bank if q["theme"] == "motivation" and q.get("core")]
    motiv = [q for q in motiv if q.get("own")] or motiv
    if motiv:
        order.insert(random.randint(0, 3), random.choices(motiv, w(motiv))[0])
    extras = [q for q in bank if not q.get("core") and q["theme"] != "motivation"]
    if extras and random.random() < extra_chance:
        # favour a value the core questions that will fit in time barely touch (usually Dream Team / Think Deeper)
        seen = Counter(q.get("value") for q in order[:6])
        ew = [x * (3 if seen[q.get("value")] == 0 else 1) for q, x in zip(extras, w(extras))]
        order.insert(random.randint(1, 4), random.choices(extras, ew)[0])
    return order[:n]


def values_text(cfg: dict, key: str = "brief", only: str | None = None) -> str:
    """The company's values from the process, one per line, with the chosen field (brief / probe / red_flags).
    Empty when the process lists no values."""
    vals = cfg.get("values") or {}
    if isinstance(vals, list):                      # names only
        return ", ".join(vals)
    return "\n".join(f"- {name}: {(v or {}).get(key, '')}".rstrip(": ") for name, v in vals.items()
                     if only is None or name == only)


def level_text(cfg: dict) -> str:
    p = cfg.get("process", {})
    return p.get("candidate_level") or " ".join(str(cfg["interview"].get("candidate_level", "")).split()) \
        or "experienced hire"


def value_field(cfg: dict, name: str, key: str) -> str:
    v = (cfg.get("values") or {}).get(name, {}) if isinstance(cfg.get("values"), dict) else {}
    return (v or {}).get(key, "")


def round_name(p: dict) -> str:
    """'Acme Hiring Manager interview (behavioural)'."""
    desc = f" ({p['round_description']})" if p.get("round_description") else ""
    return f"{p['company']} {p['round']} interview{desc}"


# --- Interview -----------------------------------------------------------------------------

def wording(q: dict) -> str:
    """One of the question's natural wordings, picked at random (asked as written)."""
    v = q["q"]
    return random.choice(v) if isinstance(v, list) else v


def _condense(text: str, words: int) -> str:
    """Keep the start (situation, action) and the end (result) of a long answer."""
    w = text.split()
    if len(w) <= words:
        return text
    head = int(words * 0.65)
    return " ".join(w[:head]) + " ... " + " ".join(w[-(words - head):])


# Said before about half of the main questions (the code picks; the rest start straight away).
BRIDGES = ["Let's move on to something different.", "Let me ask you about something else.",
           "Okay, changing topic a bit.", "Thanks. Next question.", "Alright, a different area now."]

# Words a sentence doesn't end on: a short turn ending like this is a thinking pause, not an answer.
_DANGLING = {"and", "but", "so", "because", "or", "uh", "um", "erm", "the", "a", "an", "to", "of", "that",
             "which", "with", "for", "like", "then", "when", "my", "we", "i"}


def unfinished(text: str, secs: float) -> bool:
    """True when voice detection most likely ended the turn mid-sentence ("..., and, uh...")."""
    t = text.strip()
    if not t or secs > 20:
        return False
    last = re.findall(r"[A-Za-z']+", t.lower())[-1:] or [""]
    return t.endswith(("...", "…", ",", "-")) or last[0] in _DANGLING

# Follow-ups that put real pressure on an answer, as a real interviewer says them: short and plain.
CURVEBALLS = [
    "What did you do yourself, as opposed to the team?",
    "How do you know it was your change that made the difference?",
    "Looking back, what would you do differently?",
    "Why didn't you raise it earlier?",
    "What did your lead think about that?",
    "Is that number measured, or an estimate?",
    "What would you improve next?",
    "What was the risk, and how did you handle it?",
    "Were you happy with the result?",
]

SYSTEM = """You play the interviewer in the {round_name} on a video call. The candidate is applying for: {role}. {who_line}
{panel}
About the team: {facts}

Candidate profile:
{profile}

Role level: {level}
Pitch your questions and follow-ups at this level; never ask about things the role doesn't involve.

How this round runs: {format}

How you speak: everything you write is read aloud by a voice engine. Plain spoken English, the way a friendly but neutral engineer talks on a call. No lists, markdown, emojis, stage directions, speaker labels or quotation marks. Speak in the first person as the person the note names; say your own name only when introducing yourself, and the candidate's name only when greeting and saying goodbye. A short acknowledgement ("Mm-hm") has already been played, so don't open with filler like "Great", "Okay" or "Thanks for sharing" unless the note gives you a bridge. Never praise, criticise or evaluate an answer ("good answer", "that was incomplete"); stay neutral.

How you ask follow-ups: ONE short, plain question, usually under 15 words, about something specific the candidate just said. For example: "How did you measure that?", "What was your part in it?", "What happened in the end?", "Why that approach?", "Can you give me a specific example?". Never join two questions. Never invent hypothetical scenarios or add conditions ("imagine...", "what if you had no one to ask..."). No interview jargon ("bold target", "10x", "business value", "STAR"). What a real interviewer does:
- a general answer ("I always...", "usually I...") -> ask for one specific time it happened;
- no result or number -> ask what the result was, or how they knew;
- "we" all the way through -> ask what they did themselves;
- the same story they already told for an earlier question, or an answer to a different question -> just ask for another example ("Do you have a different example?"), without commenting on their answer;
- a rambling answer -> politely ask what the outcome was;
- something unclear -> ask about it plainly.
Refer only to what the candidate actually said (including the condensed summary in notes); never invent facts, people or events in their story. Never ask a question that contradicts the profile. {rules}Every question must make sense to someone who heard only this conversation.

What you assess, silently: {assess}; and whether answers are {good_answers}. Never mention, name or quote what you assess, the company's values or its "bar".{values}

Text in <note> tags comes from the interview system, not the candidate: follow it and never mention it.
Start every reply with a tag: [F] when you ask a follow-up or answer a candidate question, [N] when you ask the next question from the note, [E] when you close the interview."""


NO_PROFILE = ("not given: you know nothing about the candidate beyond what they say in this call (no name, "
              "CV or employer), so never assume any")


def profile_text(profile: str) -> str:
    return profile.strip() or NO_PROFILE


def close_note(cfg: dict, name: str = "x") -> str:
    closing = cfg.get("process", {}).get("closing") or "say the recruiter will be in touch about next steps"
    bye = "say goodbye by name" if name else "say goodbye"
    return (f"Then thank them for their time, {closing.rstrip('.')}, and {bye}. "
            "Two short sentences, no evaluation. Tag [E].")


def panel_desc(cfg: dict) -> str:
    """'one interviewer, the Lead Platform Python Engineer ...' style description for the coach and debrief prompts."""
    roles = [r["role"] + (" (observer)" if r.get("observer") else "") for r in cfg["panel"]]
    return (f"one interviewer, the {roles[0]}" if len(roles) == 1
            else f"a panel of {len(roles)}: " + "; ".join(roles))


def pick_panel(cfg: dict) -> list[dict]:
    """One person per panel role (names and voices vary between sessions). `opens: true` marks who
    greets and runs the call (the old `key: head` still works); `technical: true` (or the old
    `key: python`) marks who asks the technical-curiosity follow-ups; `observer: true` (a shadow or note-taker)
    is introduced but never speaks."""
    return [{"key": r.get("key", ""), "role": r["role"],
             "opens": bool(r.get("opens") or r.get("key") == "head"),
             "technical": bool(r.get("technical") or r.get("key") == "python"),
             "observer": bool(r.get("observer")),
             **random.choice(r["people"])} for r in cfg["panel"]]


def _fmt_clock(secs: float) -> str:
    secs = max(0, int(secs))
    return f"{secs // 60}:{secs % 60:02d}"


def interviewer_system(cfg: dict, panel: list[dict], profile: str) -> str:
    p = cfg["process"]
    members = "\n".join(f"- {m['name']}, {m['role']}"
                        + (" (only observes and takes notes: never speaks, never write lines for them)"
                           if m.get("observer") else "") for m in panel)
    talking = [m for m in panel if not m.get("observer")]
    who_line = (f"One person from {p['interviewer_from']}" if len(talking) == 1 else
                f"{len(talking)} people from {p['interviewer_from']}") + \
               (", with an observer:" if len(talking) < len(panel) else ":")
    vals = values_text(cfg)
    return SYSTEM.format(
        round_name=round_name(p), role=p["role"], who_line=who_line, panel=members,
        facts=p["team_facts"] or "nothing beyond the basics; if asked, answer plausibly and briefly",
        profile=profile_text(profile), level=level_text(cfg), format=p["format"],
        rules=(p["interviewer_rules"].rstrip() + " ") if p["interviewer_rules"] else "",
        assess=p["assess"], good_answers=p["good_answers"],
        values=f" {p['company']} values:\n{vals}" if vals else "")


class Interview:
    def __init__(self, cfg: dict, io, llm, bank: dict, profile: str, panel: list[dict]):
        self.cfg, self.io, self.llm, self.bank, self.panel = cfg, io, llm, bank, panel
        self.active = [m for m in panel if not m.get("observer")] or panel[:1]    # who speaks
        self.observers = [m for m in panel if m.get("observer") and m not in self.active]
        self.system = interviewer_system(cfg, panel, profile)
        self.transcript: list[dict] = []     # {"who","text", optional "q", "metrics", "latency"}
        self.weak = load_weak()
        m = re.search(r"^Name:\s*([A-Za-z'-]+)", profile, re.M)
        self.cand_name = m.group(1) if m else ""

    def _turn(self, thread: list[dict], note: str, label: str, voices: dict, after: Turn | None = None,
              max_tokens: int | None = None, replace: dict | None = None) -> tuple[str, str, dict]:
        """One model reply. `voices` maps tag -> panel member, so the speaker follows the model's choice."""
        msgs = [dict(m) for m in thread]
        msgs[-1]["content"] = f"{msgs[-1]['content']}\n<note>{note}</note>".strip()
        tagged = Tagged(self.llm.stream(self.system, msgs, label=label, max_tokens=max_tokens), self.cand_name,
                        replace)
        who = lambda: voices.get(tagged.tag) or voices.get("F") or self.active[0]
        text = self.io.say(tagged, after_turn=after, who=who).strip()
        member = who()
        self.transcript.append({"who": member["name"], "text": text})
        return tagged.tag, text, member

    def _log_answer(self, q: dict | None, turn: Turn):
        m = answer_metrics(turn.text, turn.speech_seconds)
        entry = {"who": "you", "text": turn.text, "metrics": m}
        if q:
            entry["q"] = q["id"]
        self.transcript.append(entry)
        if getattr(self.io, "last_latency", None):
            entry["latency"] = dict(self.io.last_latency)

    def run(self) -> None:
        ic = self.cfg["interview"]
        total = ic["minutes"] * 60
        cand_secs = ic.get("candidate_minutes", 5) * 60
        now = getattr(self, "clock", time.monotonic)        # the dry run (selftest) injects a fake clock
        start = now()
        remaining = lambda: total - (now() - start)
        panel = self.active                 # observers never ask, follow up or answer
        lead = panel[0]
        other = lambda m: next((x for x in panel if x is not m), m)

        plan = pick_questions(self.bank["questions"], ic.get("max_questions", 10), self.weak,
                              ic.get("extra_question_chance", 0.4))
        head = next((m for m in panel if m.get("opens")), panel[0])
        lead = head
        speakers = [head] + [m for m in panel if m is not head]       # intro order
        skip_intro = random.random() < ic.get("intro_skip_chance", 0.3) and bool(plan)   # never an empty interview
        queue_ = [] if skip_intro else [{"id": "intro", "theme": "intro", "value": "",
                                          "q": self.bank["intro"], "asker": speakers[-1]}]
        for i, q in enumerate(plan):   # alternate askers; the last introducer asks first
            queue_.append({**q, "asker": speakers[(len(speakers) - 1 + i + (0 if skip_intro else 1)) % len(speakers)]})
        covered: list[str] = []
        tech_asked: list[str] = []      # technical follow-ups so far (the thread is reset per question)
        memory: list[str] = []          # condensed answers to earlier questions
        q_answers: list[str] = []       # answers to the current question

        # Panel introductions: one call, one part per person, each spoken in their own voice.
        fq = queue_[0]
        first_q = wording(fq)
        ending = (f"ask exactly: \"{first_q}\"" if not skip_intro else
                  f"say you'll go straight into questions and ask exactly: \"{first_q}\"")
        parts_desc = []
        for i, m in enumerate(speakers):
            if i == 0:
                d = (f"{m['name']} ({m['role']}): greet the candidate{' by name' if self.cand_name else ' (you do not know their name yet)'}; introduce yourself and your role in "
                     f"a sentence; say in a sentence or two what {self.cfg['process']['team']} does"
                     + (" and what you personally work on" if len(speakers) == 1 else "")
                     + (f"; mention that {' and '.join(o['name'] + ' (' + o['role'] + ')' for o in self.observers)}"
                        f" {'is' if len(self.observers) == 1 else 'are'} also on the call, only to observe"
                        if self.observers else "")
                     + f" (at most one number); then one sentence on the format: about {ic['minutes']} minutes, "
                       "mostly about their past experience, with time for their questions at the end. "
                       "Conversational, 60-100 words")
            else:
                d = f"{m['name']} ({m['role']}): introduce yourself and what you work on, 40-70 words"
            if i == len(speakers) - 1:
                d += f", then {ending}"
            parts_desc.append(f"Part {i + 1}, {d}.")
        if len(speakers) == 1:
            note = f"The candidate just joined. Write your opening, about a minute: {parts_desc[0][8:]}"
        else:
            note = ("The candidate just joined. Write the panel's opening (about a minute each), as "
                    f"{len(speakers)} parts separated by a line containing only ##. " + " ".join(parts_desc))
        print(f"\033[2m[{'the interviewer is' if len(speakers) == 1 else 'the panel are'} joining...]\033[0m", flush=True)
        raw = self.llm.complete(self.system, [{"role": "user", "content": f"(call starts)\n<note>{note}</note>"}],
                                label="intro", max_tokens=500)
        parts = [p for p in re.split(r"\n?\s*#{2,}\s*\n?", raw) if p.strip()]
        spoken = []
        for i, part in enumerate(parts):
            m = speakers[min(i, len(speakers) - 1)]
            part = LABEL.sub("", TAG.sub("", part.strip()), count=1).strip()
            self.io.say([part], who=m)
            self.transcript.append({"who": m["name"], "text": part})
            spoken.append(part)
        # The candidate must hear the first question before answering: a model may greet and stop, or say nothing.
        said = " ".join(spoken)
        if not (_norm(first_q)[:40] in _norm(said) or asks_same(first_q, said)):
            m = speakers[-1]
            line = (first_q if spoken else
                    f"Hi{', ' + self.cand_name if self.cand_name else ''}, I'm {m['name']}, {m['role']}. {first_q}")
            self.io.say([line], who=m)
            self.transcript.append({"who": m["name"], "text": line})
            spoken.append(line)
        thread = [{"role": "user", "content": "(call starts)"}, {"role": "assistant", "content": " ".join(spoken)}]

        idx, followups, phase, cand_qs = 0, 0, "questions", 0
        last_speaker = speakers[-1]
        while True:
            cur = queue_[idx] if phase == "questions" else None
            clock = f"{_fmt_clock(total - remaining())} / {_fmt_clock(total)}"
            turn = self.io.listen(max_seconds=ic["max_answer_seconds"], clock=clock)
            if not turn.text:
                continue
            # Voice detection sometimes ends the turn on a thinking pause ("..., and, uh..."). A real
            # interviewer waits, so keep listening for a few seconds and join the parts.
            for _ in range(2):
                if turn.by_key or not (getattr(self.io, "can_resume", False) and unfinished(turn.text, turn.speech_seconds)):
                    break
                more = self.io.listen(max_seconds=ic["max_answer_seconds"], clock=clock, wait_seconds=6.0)
                if not more.text:
                    break
                turn = Turn(text=f"{turn.text} {more.text}", speech_seconds=turn.speech_seconds + more.speech_seconds,
                            cut_off=more.cut_off, stt_tail=more.stt_tail, ended_at=more.ended_at)
            self.io.ack(last_speaker.get("voice"))
            self._log_answer(cur, turn)
            thread.append({"role": "user", "content": turn.text})
            q_answers.append(turn.text)
            secs = int(turn.speech_seconds)
            rem = remaining()

            if phase == "questions":
                has_next = idx + 1 < len(queue_) and rem > cand_secs + 60  # no new question in the last minute
                # hand over on time: no more digging when only the closing is left and it's close
                time_up = rem <= cand_secs or (not has_next and rem <= cand_secs + 120)
                max_f = 1 if cur["id"] == "intro" else ic["max_followups"]
                nxt_q = ""
                if has_next:
                    nq = queue_[idx + 1]
                    nxt_by = nq["asker"]
                    bridge = random.choice(BRIDGES) + " " if random.random() < 0.5 else ""
                    nxt_q = wording(nq)
                    move_on = f"{nxt_by['name']} says exactly: \"{bridge}{nxt_q}\""
                else:
                    nxt_by = lead
                    mins = max(1, round(rem / 60))
                    move_on = (f"{lead['name']} says that's all from {'their' if len(panel) == 1 else 'the panel'}"
                               f" side and asks what questions the candidate "
                               f"has for you (about {mins} minutes are left; don't say time is up).")
                fu_by = cur["asker"]
                if len(panel) > 1 and random.random() < ic.get("cross_followup_chance", 0.3):
                    fu_by = other(fu_by)
                # technical-curiosity follow-ups only when someone on the panel is technical
                r = random.random()
                tech = ic.get("technical_chance", 0.2) if any(m.get("technical") for m in panel) else 0.0
                if r < tech:   # the technical interviewer gets curious about how something actually worked
                    fu_by = next((m for m in panel if m.get("technical")), fu_by)
                    fu_how = (": out of technical curiosity, a short question about how something they described "
                              "actually worked, e.g. \"How did you deploy it?\", \"What happened when it failed?\", "
                              "\"Why did you pick that tool?\", \"How did you test it?\""
                              + ("; do NOT repeat an earlier technical question: " + " | ".join(tech_asked)
                                 if tech_asked else ""))
                elif r < tech + ic.get("curveball_chance", 0.25):
                    fu_how = (f": a harder one, like \"{random.choice(CURVEBALLS)}\" (reword it to fit what they "
                              f"said; if it doesn't fit, ask another plain, pointed question)")
                else:
                    fu_how = ""
                forced = followups >= max_f or time_up
                if turn.cut_off:
                    then = ("moves on [N]" if forced else
                            "either asks for the result in one sentence [F] or moves on [N]")
                    note = (f"The candidate has talked for over {secs // 60} minutes. {cur['asker']['name']} cuts in "
                            f"politely, then {then}: {move_on}")
                    voices = {"F": cur["asker"], "N": cur["asker"]}
                elif forced:
                    note = f"Answer took {secs}s. Move on now [N]: {move_on}"
                    voices = {"N": nxt_by, "F": nxt_by}
                else:
                    dig = ("Real interviewers almost always dig in once, so" if followups == 0 else
                           "If something important is still missing (or they only spoke in general terms),")
                    probe = value_field(self.cfg, cur.get("value", ""), "probe")
                    if probe and not fu_how:
                        fu_how = f" about whatever their answer was missing, e.g. {probe}"
                    note = (f"Answer took {secs}s. {dig} {fu_by['name']} asks ONE follow-up [F]{fu_how}. "
                            f"Move on [N] only if the answer already had a clear personal action, a measurable "
                            f"result and its business value: {move_on}")
                    voices = {"F": fu_by, "N": nxt_by}
                handover = (f"That's all from {'my' if len(panel) == 1 else 'our'} side. "
                            "What questions do you have for me?")
                if forced and followups > max_f and not turn.cut_off:
                    # The model already ignored one move-on note (weaker local models do): ask the next
                    # question directly, so the interview can't get stuck on one topic.
                    line = f"{bridge}{nxt_q}" if has_next else handover
                    self.io.say([line], after_turn=turn, who=nxt_by)
                    self.transcript.append({"who": nxt_by["name"], "text": line})
                    tag, text, last_speaker = "N", line, nxt_by
                else:
                    # Moving on means asking the next question exactly as written, or handing over with a fixed
                    # line (a cut-in keeps the model's wording, since it starts with the polite interruption).
                    # Closing [E] isn't the model's call while questions remain: it becomes the move-on line.
                    move_line = f"{bridge}{nxt_q}" if has_next else handover
                    fixed = {"E": move_line} if turn.cut_off else {"N": move_line, "E": move_line}
                    tag, text, last_speaker = self._turn(thread, note, f"q:{cur['id']}", voices, after=turn,
                                                         replace=fixed)
                if r < tech and tag == "F":
                    tech_asked.append(text[:90])
                # Advance only when the turn really moved on. Trusting the follow-up limit alone once logged
                # a cut-in follow-up as the next question, so that question was silently skipped.
                moved = tag in ("N", "E") or (bool(nxt_q) and (_norm(nxt_q)[:40] in _norm(text)
                                                               or asks_same(nxt_q, text)))
                if moved:
                    covered.append(cur["id"] if cur["id"] == "intro" else f"{cur['id']} ({cur['theme']})")
                    memory.append(f"- {cur['id']}: {_condense(' '.join(q_answers), 90 if cur['id'] == 'intro' else 70)}")
                    q_answers = []
                    if cur["id"] != "intro":
                        self.weak["asked"][cur["id"]] = self.weak["asked"].get(cur["id"], 0) + 1
                    idx, followups = idx + 1, 0
                    if not has_next:
                        phase = "candidate"
                    # The thread is reset per question to save tokens; a condensed memory of every earlier
                    # answer keeps references to "what you said earlier" grounded in what was actually said.
                    ctx = ("What the candidate has said so far (condensed, in order):\n" + "\n".join(memory) + "\n"
                           + ("You just asked the next question." if has_next else "The candidate's questions now."))
                    thread = [{"role": "user", "content": f"<note>{ctx}</note>"}, {"role": "assistant", "content": text}]
                else:
                    followups += 1
                    thread.append({"role": "assistant", "content": text})
            else:  # the candidate asks questions
                cand_qs += 1
                last = cand_qs >= ic.get("max_candidate_questions", 4) or rem < 60
                by = random.choice(panel)
                note = (f"{by['name']} answers the candidate's question from their own role, using what you know "
                        f"about the team, realistically and briefly (up to 70 words). "
                        + (close_note(self.cfg, self.cand_name) if last else
                           "Then ask if they have another question. Tag [F]. If they have no more questions, "
                           + close_note(self.cfg, self.cand_name)))
                tag, text, last_speaker = self._turn(thread, note, "cand_q", {"F": by, "E": by}, after=turn,
                                                     max_tokens=220)
                thread.append({"role": "assistant", "content": text})
                if tag == "E" or last:
                    break


# --- Drill ---------------------------------------------------------------------------------

DRILL_SYSTEM = """You are a sharp interview coach preparing a candidate for the {round_name} with {team} ({panel}). The role: {role}.{values}
Role level: {level}
The interviewers assess {assess}. They want answers that are {good_answers}. Judge at the role level above. Your feedback is read aloud: plain spoken English, no lists or markdown.
Candidate profile:
{profile}"""


def drill_system(cfg: dict, profile: str) -> str:
    p, vals = cfg["process"], values_text(cfg)
    return DRILL_SYSTEM.format(round_name=round_name(p), team=p["team"], role=p["role"], panel=panel_desc(cfg),
                               values=f" {p['company']} values:\n{vals}" if vals else "",
                               level=level_text(cfg), assess=p["assess"], good_answers=p["good_answers"],
                               profile=profile_text(profile))


def run_drill(cfg, io, llm, bank, profile, log: list[dict]):
    weak = load_weak()
    system = drill_system(cfg, profile)
    q, asked = None, ""
    while True:
        if q is None:
            pool = bank["questions"]
            # core topics (what the panel said they ask) 3x as likely as extras
            q = random.choices(pool, [weight(x, weak) * (3 if x.get("core") else 1) for x in pool])[0]
            asked = io.say([wording(q)]).strip()
        else:
            io.say([asked])
        log.append({"who": "interviewer", "text": asked})
        turn = io.listen(max_seconds=cfg["interview"]["max_answer_seconds"])
        if not turn.text:
            continue
        io.ack()
        m = answer_metrics(turn.text, turn.speech_seconds)
        log.append({"who": "you", "text": turn.text, "q": q["id"], "metrics": m})
        val = q.get("value", "")
        flags = value_field(cfg, val, "red_flags")
        prompt = (f"Question asked: {asked}\n(Theme {q['theme']}" + (f", tests '{val}'" if val else "") + ".)\n"
                  + (f"Red flags for '{val}': {flags}\n" if flags else "")
                  + f"Answer (speech-to-text): {turn.text}\nDelivery: {format_metrics(m)}\n"
                  "Give exactly 3 spoken sentences, each under 30 words: what worked, the biggest gap (including how "
                  "well it showed what the question tests, or any red flag), one concrete fix "
                  "(quote their words if useful). Then on a new line: SCORE: <1-5>")
        out = []

        def spoken(chunks):
            """Yield the feedback text, never the trailing SCORE line (hold back 6 chars until sure)."""
            emitted = 0
            for ch in chunks:
                out.append(ch)
                full = "".join(out)
                end = full.find("SCORE")
                safe = end if end >= 0 else max(emitted, len(full) - 6)
                if safe > emitted:
                    yield full[emitted:safe]
                    emitted = safe
            full = "".join(out)
            if "SCORE" not in full and len(full) > emitted:
                yield full[emitted:]
        io.say(spoken(llm.stream(system, [{"role": "user", "content": prompt}], label=f"drill:{q['id']}",
                                 max_tokens=300, model=cfg["llm"].get("feedback_model"),
                                 effort=cfg["llm"].get("feedback_effort"))), after_turn=turn, who={"name": "Coach"})
        full = "".join(out)
        log.append({"who": "coach", "text": full.split("SCORE")[0].strip()})
        sm = re.search(r"SCORE:\s*([1-5])", full)
        if sm:
            update_weak(weak, q["theme"], int(sm.group(1)))
            log[-2]["score"] = int(sm.group(1))
        weak["asked"][q["id"]] = weak["asked"].get(q["id"], 0) + 1
        save_weak(weak)

        io.say(["Again, next, or stop?"], who={"name": "Coach"})
        cmd = io.listen(silence_end=1.2).text.lower()
        if any(w in cmd for w in ("stop", "quit", "exit", "done", "enough")):
            return
        if not any(w in cmd for w in ("again", "retry", "repeat", "same", "one more time")):
            q = None


# --- Debrief and saving --------------------------------------------------------------------

DEBRIEF_SYSTEM = """You are an expert interview coach. You review a mock {round_name} (~{minutes} min, {team}, {panel}). The role: {role}. The interviewer side assesses {assess}, and wants answers that are {good_answers}.
Role level: {level}
Calibrate every score, the verdict and the model answers to this level: compare the candidate with strong candidates for the same role, not with more senior people. For numbers, credit any honest figure the candidate could realistically have had at this level (before/after times, scale, counts, honest estimates). Model answers must sound like a strong candidate at this level. Candidates commonly fail this round on: {failures}{values}
Answers were transcribed from speech, so ignore transcription glitches. The candidate's prepared notes are reference material, not the truth: never score them down for details or numbers that differ from the notes; judge consistency only within the interview itself. Be blunt, specific and useful."""

# The debrief format in parts: sent together (one call), or in several smaller calls for models or servers
# with a low output cap (llm.debrief_mode; "auto" = sections when the debrief model is local).
FMT_SPOKEN = """SPOKEN:
3-4 plain spoken sentences for text-to-speech: overall verdict (strong hire / hire / lean no / no) and the two most important fixes."""

FMT_ISSUES = """1. "# Interviewer and system issues". The interviewer is an AI and the answers pass through voice detection, so both can fail. List:
- interviewer questions that made no sense, referred to things the candidate never said, repeated an answered question or were unfair, and any interviewer text that is clearly a note to itself rather than speech;
- candidate turns that are only a word or a fragment (a few seconds). These are almost always the voice detection ending the turn while the candidate paused to think, not the candidate going silent;
- answers cut off mid-sentence at the time limit.
Never penalise the candidate for these. Mark such turns "n/a (glitch)" and judge the follow-up exchange on its merits; if they handled a bad question well, credit it. Write "None" if there were no issues. Judge each turn by the question actually asked, not by its q= tag (tags can be wrong)."""

FMT_OVERVIEW = """2. "# Overview": one table with a row per candidate turn (main answers and follow-up replies): question (short), seconds, verdict (✅ satisfied / ⚠️ partly / ❌ not satisfied), then scores 1-5 for structure (STAR + result), conciseness, concrete numbers (any honest number counts at this level), value for users or the business, individual contribution ("I" vs "we"), fit with {fit_with}, and length (target {lo}-{hi}s for main answers; short follow-up replies are fine when crisp). The verdict answers one question: would the interviewer tick this off and move on happy?"""

FMT_ANSWER_ITEMS = """- **What they were testing:** the value and the signal behind the question, in one or two sentences.
- **How satisfying it was:** the verdict (✅/⚠️/❌), then two or three sentences written as the interviewer's private notes after hearing it (e.g. "Clear ownership, but I had to drag the result out of him"), and what would have made it a clear ✅.
- **What worked:** 1-3 bullets, each quoting the candidate. Say "Nothing usable" if so.
- **What to do better:** 2-5 bullets. Each bullet quotes what the candidate actually said, explains in a few words how it lands with the interviewer, and gives the exact sentence to say instead.
- **Best story:** the prepared story from the notes that fits best, by name, and why it beats the one used (or "you picked right"). If no story fits, say which real experience from the transcript or profile to use.
- **Points you should have made:** 4-6 bullets: the specific facts, numbers, your own actions, the result and the lesson, taken from the notes, profile or what the candidate said. Mark each one "said", "missing" or "garbled".
- **Say it like this:** a complete model answer in the candidate's first-person voice, 120-200 words (about 60-90 s spoken), STAR order, opening with the headline result, plain spoken English the candidate could actually say. Use only facts from the transcript, notes or profile; never invent numbers, names, team sizes, durations or events. Where a number is needed but unknown, write [your number]; otherwise be concrete, never a placeholder like "[your example]" when the transcript or notes hold a usable one. The notes are prepared material, not the truth: new details or numbers the candidate gave in the interview are fine and are not a weakness; prefer them. Keep every model answer consistent with what the candidate said elsewhere in this interview.
- **Follow-ups:** for each follow-up actually asked, a sub-bullet with the follow-up quoted, what it was after, the verdict on the reply (✅/⚠️/❌) with the reason in one sentence quoting the reply, and a 1-3 sentence model reply."""

FMT_ANSWERS = """3. "# Answer by answer": the most important section, so be thorough. For EVERY main question asked (skip glitch-only turns), in order, a "## N. <the question as asked>" heading, then:
""" + FMT_ANSWER_ITEMS

FMT_REST = """4. "# Patterns across the interview": blame, minimising failures, lukewarm motivation, or the same weakness showing in several answers, each with quotes; "None" if there were none.

5. {values_section}

6. "# Self-contradictions": places where the candidate contradicted themselves within this interview (e.g. claiming estimates are always accurate after telling a story about a wrong estimate, or two different numbers for the same thing). Quote both statements, say why an interviewer would notice, and give the exact sentence that would have kept them consistent. Do not list differences from the notes as contradictions. Write "None" if there were none.

7. "# Story choice": whether the right prepared story was used overall; flag a story told twice and which questions should have used a different one.

8. "# Your questions at the end": a verdict (✅/⚠️/❌) for each question the candidate asked, with the reason, then 2-3 sharper, informed ones ({question_ideas}).

9. "# Delivery": patterns from the metrics (speed, length, fillers, restarts, "I" vs "we").

10. "# Top 3 to practise": concrete drills for the next session."""

FMT_SCORES = """SCORES:
One line of JSON mapping each of these themes to an overall 1-5 score, using exactly these keys: {themes}"""

DEBRIEF_FORMAT = ("Write exactly three sections:\n\n" + FMT_SPOKEN + "\n\nREPORT:\nMarkdown, in this order.\n\n"
                  + "\n\n".join([FMT_ISSUES, FMT_OVERVIEW, FMT_ANSWERS, FMT_REST]) + "\n\n" + FMT_SCORES)


def _label(name: str) -> str:
    """A section label, as written or as a markdown heading ("SPOKEN:", "## REPORT:", "### SPOKEN")."""
    # the label alone on its line, or followed by a colon; never the first word of a sentence ("Scores across...")
    return rf"^[ \t#*]*{name}(?:[ \t]*\*{{0,2}}[ \t]*:\*{{0,2}}|\*{{0,2}}[ \t]*(?=\n|$))[ \t]*"


_FLAGS = re.S | re.M | re.I


def _section(text: str, name: str, until: str | None = None) -> str:
    m = re.search(_label(name) + r"\s*(.*?)(?:" + (_label(until) if until else r"(?!x)x") + r"|\Z)", text, _FLAGS)
    return m.group(1).strip() if m else ""


def _scores(text: str) -> dict:
    m = re.search(_label("SCORES") + r"\s*(\{.*?\})", text, _FLAGS)
    try:
        return json.loads(m.group(1)) if m else {}
    except json.JSONDecodeError:
        return {}


def _question_of(line: str) -> str:
    """The question in an interviewer line: its last sentence ending in "?", else its last sentence (an
    opening paragraph that ends "...tell me about your biggest result.")."""
    sents = [x.strip() for x in re.split(r"(?<=[.!?])\s+", line.strip()) if x.strip()]
    qs = [x for x in sents if x.endswith("?")]
    return (qs[-1] if qs else sents[-1] if sents else line)[:300]


def _main_questions(transcript: list[dict], bank: dict | None = None) -> list[tuple[str, int]]:
    """The main questions (not follow-ups) as (question as asked, index of the interviewer line). With question
    ids: the interviewer line before the first answer to each id. Without (older saved sessions): the lines
    that contain a wording from the question bank, since main questions are always asked exactly as written;
    the first question of the interview always counts."""
    out, seen, last = [], set(), -1
    if any(e.get("q") for e in transcript):
        for i, e in enumerate(transcript):
            if e["who"] != "you":
                last = i
            elif e.get("q") and e["q"] not in seen and last >= 0:
                seen.add(e["q"])
                out.append((_question_of(transcript[last]["text"]), last))
        return out
    if not bank:
        return []
    wordings = [_norm(w) for q in bank["questions"] for w in (q["q"] if isinstance(q["q"], list) else [q["q"]])]
    wordings += [_norm(w) for w in (bank.get("intro") or [])]
    for i, e in enumerate(transcript):
        if e["who"] != "you" and i + 1 < len(transcript) and transcript[i + 1]["who"] == "you":
            if not out or any(w and w in _norm(e["text"]) for w in wordings):
                out.append((_question_of(e["text"]), i))
    return out


REPORT_SECTIONS = ["Interviewer and system issues", "Overview", "Answer by answer", "Patterns across the interview",
                   "Values", "What they assess", "Self-contradictions", "Story choice", "Your questions at the end",
                   "Delivery", "Top 3 to practise"]


def _tidy_headings(report: str) -> str:
    """Report sections as top-level headings whatever the model wrote: '# # Self-contradictions',
    '## 1. # Overview', '## 2. "# Overview"', '## 7. Story Choice', '**Delivery**' -> '# Overview' etc."""
    report = re.sub(r'(?m)^#{1,6}[ \t]+(?:\d+\.[ \t]*)?"?#+[ \t]*([^"\n]+?)"?[ \t]*$', r"# \1", report)
    # the prompt's own notation copied literally, title and content on one line: 7. "# Story choice": text
    report = re.sub(r'(?m)^[ \t]*\d+\.[ \t]*"#+[ \t]*([^"\n]+)"[ \t]*:?[ \t]*(.*)$',
                    lambda m: f"# {m.group(1).strip()}" + (f"\n\n{m.group(2)}" if m.group(2).strip() else ""), report)
    for name in REPORT_SECTIONS:                      # "9. Delivery: text" / "9. **Delivery**" -> "# Delivery"
        pat = re.escape(name).replace(r"\ ", r"[ \t]+")
        report = re.sub(rf'(?mi)^[ \t]*\d+\.[ \t]*\*{{0,2}}{pat}\*{{0,2}}[ \t]*:?\*{{0,2}}[ \t]*(.*)$',
                        lambda m, name=name: f"# {name}" + (f"\n\n{m.group(1)}" if m.group(1).strip() else ""), report)
    for name in REPORT_SECTIONS:
        pat = re.escape(name).replace(r"\ ", r"[ \t]+")
        report = re.sub(rf'(?mi)^(?:#{{1,6}}[ \t]*|\*\*)(?:\d+\.[ \t]*)?"?#?[ \t]*{pat}"?:?(?:\*\*)?:?[ \t]*$',
                        f"# {name}", report)
    return report


def _debrief_sections(llm, system: str, context: str, fmt: dict, transcript: list[dict], lines: list[str],
                      bank: dict, **kw) -> str:
    """The debrief in several small calls, each well under any output cap (one local server stops every reply
    at 2,047 tokens): verdict + issues + overview; one call per main question, given only its part of the
    transcript; the closing sections in two calls; the scores. Returns the single-call layout."""
    ask = lambda prompt, label, ctx=context: llm.complete(system, [{"role": "user", "content": ctx + prompt}],
                                                          label=label, **kw)
    mains = _main_questions(transcript, bank)
    first = ask("Write exactly two sections:\n\n" + FMT_SPOKEN + "\n\nREPORT:\nMarkdown with only these two "
                "parts:\n\n" + FMT_ISSUES + "\n\n" + FMT_OVERVIEW.format(**fmt), "debrief:1")
    spoken = _section(first, "SPOKEN", "REPORT")
    report = [_section(first, "REPORT")]
    m = re.search(r"^\s*#", spoken, re.M)                 # REPORT label skipped: split at the first heading
    if m:
        report, spoken = [spoken[m.start():].strip()], spoken[:m.start()].strip()
    answers = []
    for n, (question, start) in enumerate(mains, 1):
        end = mains[n][1] if n < len(mains) else len(lines)
        part = ask(f"Write only the feedback for main question {n}, nothing else: no other questions, no other "
                   f"sections. Start with the heading \"## {n}. {question}\", then:\n{FMT_ANSWER_ITEMS}\n\n"
                   f"This question's part of the transcript (the question, the answer and its follow-ups):\n"
                   + "\n".join(lines[start:end]), f"debrief:q{n}")
        answers.append(re.sub(r"^\s*#\s*Answer by answer\s*\n", "", part.strip()))
    rest_a, rest_b = FMT_REST.split("\n\n7. ", 1)
    rest = [ask("Write only these parts of the report, in this order:\n\n" + rest_a.format(**fmt), "debrief:rest1"),
            ask("Write only these parts of the report, in this order, then the SCORES line:\n\n7. "
                + rest_b.format(**fmt) + "\n\n" + FMT_SCORES.format(**fmt), "debrief:rest2")]
    scores = _scores(rest[1])
    if not scores:
        scores = _scores("SCORES: " + ask("Output only one line of JSON, nothing else, mapping each of these themes "
                                          f"to an overall 1-5 score: {fmt['themes']}", "debrief:scores")
                         .strip().removeprefix("SCORES:").strip())
    report += ["# Answer by answer\n\n" + "\n\n".join(answers)]
    report += [re.split(_label("SCORES"), r, maxsplit=1, flags=_FLAGS)[0].strip() for r in rest]
    return (f"SPOKEN:\n{spoken}\n\nREPORT:\n" + "\n\n".join(r for r in report if r)
            + f"\n\nSCORES:\n{json.dumps(scores)}")


def transcript_md(transcript: list[dict]) -> str:
    out = []
    for e in transcript:
        if e["who"] == "you":
            line = f"**You:** {e['text']}" + (f" <!-- q={e['q']} -->" if e.get("q") else "")
            if "metrics" in e:
                line += f"\n\n> _{format_metrics(e['metrics'])}_"
            if "latency" in e:
                line += f"  \n> _latency: {e['latency']}_"
            out.append(line)
        else:
            out.append(f"**{e['who']}:** {e['text']}")
    return "\n\n".join(out)


def debrief(cfg, llm, transcript, bank, profile, notes: str) -> tuple[str, str, dict]:
    themes = {q["id"]: q["theme"] for q in bank["questions"]}
    lines = []
    for e in transcript:
        if e["who"] == "you":
            m = format_metrics(e["metrics"]) if "metrics" in e else e.get("metrics_str", "")
            tag = f" [q={e.get('q', '?')}, theme={themes.get(e.get('q'), e.get('q', '?'))}; {m}]"
            lines.append(f"CANDIDATE{tag}: {e['text']}")
        else:
            lines.append(f"INTERVIEWER ({e['who']}): {e['text']}")
    # themes actually asked, for the SCORES line (weak_spots.json only keeps these names)
    asked = list(dict.fromkeys(themes.get(e["q"], "intro") for e in transcript if e["who"] == "you" and e.get("q")))
    lo, hi = cfg["interview"]["target_answer_seconds"]
    p = cfg["process"]
    vals = cfg.get("values") or {}
    if vals:
        values_section = (f'"# Values": a table rating each of the {len(vals)} {p["company"]} values 1-5 across the '
                          "whole interview, with the strongest evidence and any red flag quoted, and \"not shown\" "
                          "when there was no evidence.")
    else:
        values_section = ('"# What they assess": a table rating each thing the interviewer assesses 1-5 across the '
                          "whole interview, with the strongest evidence and any red flag quoted, and \"not shown\" "
                          "when there was no evidence.")
    ideas = (f"e.g. about {p['candidate_question_ideas']}" if p.get("candidate_question_ideas")
             else "informed by what the interviewer said about the team and its work")
    fmt = dict(lo=lo, hi=hi, themes=", ".join(asked) or "intro",
               fit_with="the value it tests" if vals else "what the question tests",
               values_section=values_section, question_ideas=ideas)
    context = (f"Candidate profile: {profile_text(profile)}\n\n"
               + (f"Candidate's prepared notes (stories and answers):\n{notes}\n\n" if notes else "")
               + "Transcript:\n" + "\n".join(lines) + "\n\n")
    user = context + DEBRIEF_FORMAT.format(**fmt)
    if isinstance(vals, dict) and vals:
        vtxt = ("\nThe values are never asked about by name; the interviewer reads them from the stories. "
                f"{p['company']} values, what they mean, and red flags:\n"
                + "\n".join(f"- {n}: {(v or {}).get('brief', '')}. Red flags: {(v or {}).get('red_flags', '')}"
                            for n, v in vals.items()))
    else:
        vtxt = ""
    system = DEBRIEF_SYSTEM.format(round_name=round_name(p), minutes=cfg["interview"]["minutes"], team=p["team"],
                                   panel=panel_desc(cfg), role=p["role"], assess=p["assess"],
                                   good_answers=p["good_answers"], level=level_text(cfg),
                                   failures=p["known_failure_reasons"].rstrip(".") + ".", values=vtxt)
    kw = dict(model=cfg["llm"]["debrief_model"], max_tokens=48000, effort=cfg["llm"].get("debrief_effort", "low"))
    mode = cfg["llm"].get("debrief_mode", "auto")
    if mode == "sections" or (mode == "auto" and llm.is_local(cfg["llm"]["debrief_model"])):
        text = _debrief_sections(llm, system, context, fmt, transcript, lines, bank, **kw)
    else:
        text = llm.complete(system, [{"role": "user", "content": user}], label="debrief", **kw)
    spoken, report, scores = _section(text, "SPOKEN", "REPORT"), _section(text, "REPORT", "SCORES"), _scores(text)
    # a model that skips the REPORT label puts the report under SPOKEN: the verdict ends at the first heading
    m = re.search(r"^\s*#", spoken, re.M)
    if m:
        spoken, report = spoken[:m.start()].strip(), (spoken[m.start():].strip() + "\n\n" + report).strip()
    known = set(themes.values()) | {"intro"}
    scores = {k: v for k, v in scores.items() if k in known}   # never let invented names into weak spots
    report = _tidy_headings(report or text)
    known = set(re.findall(r"\d+(?:\.\d+)?", (profile + " " + notes + " " + " ".join(lines)).replace(",", "")))
    blocks = re.findall(r"Say it like this:?\*{0,2}\s*(.+?)(?:\n\s*[-*]\s*\*\*|\n#|\Z)", report, re.S)
    invented = sorted({n for b in blocks for n in re.findall(r"\d+(?:\.\d+)?", b.replace(",", ""))} - known, key=float)
    if invented:
        report += ("\n\n> **Check:** the model answers above use numbers that never came up in this interview, "
                   f"your profile or notes: {', '.join(invented)}. Treat them as placeholders for your own figures.")
    return spoken, report, scores


def save_session(mode: str, transcript, report: str, llm) -> Path:
    SESS_DIR.mkdir(parents=True, exist_ok=True)
    path = SESS_DIR / f"{dt.datetime.now():%Y-%m-%d-%H%M}-{mode}.md"
    tin, tout = llm.totals()
    usage = "\n".join(f"| {l} | {i} | {o} |" for l, i, o in llm.usage)
    path.write_text(
        f"# Mock {mode} — {dt.datetime.now():%Y-%m-%d %H:%M}\n\n"
        + (f"## Debrief\n\n{report}\n\n" if report else "")
        + f"## Transcript\n\n{transcript_md(transcript)}\n\n"
        + f"## Tokens ({llm.backend}, {tin} in / {tout} out)\n\n| call | input | output |\n|---|---|---|\n{usage}\n"
    )
    return path


def parse_session_md(text: str) -> list[dict]:
    """Rebuild a transcript from a saved session file (for re-scoring old sessions). A turn may span several
    paragraphs; the question id is an invisible <!-- q=... --> marker at the end of an answer."""
    body = text.split("## Transcript", 1)[-1].split("\n## Tokens", 1)[0]
    out: list[dict] = []
    for block in re.split(r"\n\s*\n", body):
        block = block.strip()
        if not block:
            continue
        m = re.match(r"\*\*([^*]+):\*\*\s*(.*)", block, re.S)
        if m:
            out.append({"who": "you" if m.group(1) == "You" else m.group(1), "text": m.group(2).strip()})
        elif block.startswith("> _") and out and out[-1]["who"] == "you":
            out[-1]["metrics_str"] = re.match(r"> _(.*?)_", block).group(1)
        elif out and "metrics_str" not in out[-1]:
            out[-1]["text"] += "\n\n" + block              # a further paragraph of the same turn
    for e in out:
        q = re.search(r"\s*<!-- q=([\w-]+) -->\s*$", e["text"])
        if q:
            e["text"], e["q"] = e["text"][:q.start()].rstrip(), q.group(1)
    return out


def load_bank(cfg: dict) -> dict:
    from .process import bank
    return bank(cfg)
