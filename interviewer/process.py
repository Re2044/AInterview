"""Loads a recruitment process folder and merges it with the tool defaults.

A process folder holds everything about one employer and one interview round:
    process.yaml     company, role, round, team facts, panel, values, rubric (all optional but `company`)
    profile.md       the candidate's short profile
    questions.yaml   optional: this process's question bank (default: the tool's generic bank)
    notes file       optional: prepared stories, sent only with the debrief
    sessions/, weak_spots.json   written here by the tool

The result is one `cfg` dict: the tool's config.yaml with process overrides merged in, plus
cfg["process"] (the employer fields, with generic defaults filled in), cfg["panel"] and cfg["values"].
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent          # the AInterviewer folder
TEMPLATE = ROOT / "templates" / "process"
DEFAULT_BANK = ROOT / "questions.yaml"
SAMPLES = ROOT / "samples"                              # built-in processes that need no personal data
DEFAULT_SAMPLE = "behavioural"
# Built-in samples are read-only; their sessions and weak spots go here instead of the tool folder.
DATA_ROOT = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share") / "ainterviewer"

# Tool sections a process may override (merged key by key).
OVERRIDABLE = ("interview", "llm", "stt", "tts", "audio_device")

# Generic defaults for every employer-specific field. A process only sets what it knows.
DEFAULTS = {
    "company": "the company",
    "role": "the role",
    "round": "behavioural",
    "round_description": "",
    "team": "the team",
    "interviewer_from": "",            # default: the team
    "format": ("A conversation about the candidate's past experience; not a technical test and not a quiz. "
               "You ask a prepared question, listen, then dig into the story with short follow-ups until you "
               "know what happened, what the candidate personally did and what came of it. Then you move on. "
               "The candidate asks their questions at the end."),
    "assess": "ownership, self-organisation, prioritisation, how they work with others, and clear, genuine "
              "motivation for this company and role",
    "good_answers": "structured (STAR), concise and straight to the point, backed by concrete numbers where the "
                    "candidate has them, showing the value for users or the business and the candidate's own "
                    "contribution",
    "candidate_level": "experienced hire at the level the role describes",
    "interviewer_rules": "",
    "known_failure_reasons": "vague answers without numbers; hiding in \"we\" with no personal contribution; a "
                             "pattern where someone else is always the problem; no real failure or low point "
                             "(sounds rehearsed); complaining about a previous employer; lukewarm motivation; not "
                             "knowing what the company does",
    "closing": "say the recruiter will be in touch about next steps",
    "candidate_question_ideas": "",
    "team_facts": "",
    "vocabulary": [],
    "profile": "profile.md",
    "notes": "",
    "questions": "",
    "panel": [{"role": "Engineering Manager",
               "people": [{"name": "Alex", "voice": "am_michael"}, {"name": "Sarah", "voice": "bf_emma"},
                          {"name": "Tom", "voice": "bm_george"}, {"name": "Maya", "voice": "af_heart"}]}],
    "values": {},
}


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def samples() -> list[str]:
    return sorted(d.name for d in SAMPLES.iterdir() if (d / "process.yaml").exists())


def find_dir(arg: str | None) -> Path:
    """--process (a folder, or a built-in sample's name), else $AINTERVIEWER_PROCESS, else the current
    folder if it has a process.yaml, else the built-in generic behavioural sample."""
    for cand in (arg, os.environ.get("AINTERVIEWER_PROCESS"), "."):
        if not cand:
            continue
        if (Path(cand).expanduser() / "process.yaml").exists():
            return Path(cand).expanduser().resolve()
        if (SAMPLES / cand / "process.yaml").exists():
            return SAMPLES / cand
        if cand != ".":
            raise SystemExit(f"No process.yaml in {Path(cand).expanduser().resolve()}, and no built-in sample "
                             f"called '{cand}' (samples: {', '.join(samples())}). "
                             f"Create a process with: ainterviewer init {cand}")
    print(f"\033[2mNo process folder here: using the built-in '{DEFAULT_SAMPLE}' sample (no personal data "
          f"needed). Set up your own with: ainterviewer init <dir>\033[0m")
    return SAMPLES / DEFAULT_SAMPLE


def is_sample(pdir: Path) -> bool:
    return pdir.resolve().parent == SAMPLES.resolve()


def data_dir(pdir: Path) -> Path:
    """Where sessions and weak spots go: the process folder itself, or a per-user folder for samples."""
    return DATA_ROOT / pdir.name if is_sample(pdir) else pdir


def tool_config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text())


class ProcessError(ValueError):
    """A process folder that can't be turned into a working interview; the message names the key."""


VOICES = ["af_heart", "af_bella", "af_nicole", "bf_emma", "am_michael", "am_fenrir", "bm_george", "bm_fable"]
TEXT_KEYS = ("company", "role", "round", "round_description", "team", "interviewer_from", "format", "assess",
             "good_answers", "candidate_level", "interviewer_rules", "known_failure_reasons", "closing",
             "candidate_question_ideas", "team_facts", "profile", "notes", "questions")
# interview settings the code does arithmetic with: (type, lowest, highest)
NUMBERS = {"minutes": (int, 5, 240), "candidate_minutes": (int, 0, 60), "max_questions": (int, 1, 40),
           "max_followups": (int, 0, 10), "max_candidate_questions": (int, 0, 10),
           "max_answer_seconds": (int, 20, 900), "intro_skip_chance": (float, 0, 1),
           "extra_question_chance": (float, 0, 1), "curveball_chance": (float, 0, 1),
           "technical_chance": (float, 0, 1), "cross_followup_chance": (float, 0, 1)}


def _truthy(v) -> bool:
    return v.strip().lower() in ("true", "yes", "y", "1", "on") if isinstance(v, str) else bool(v)


def _text(v) -> str:
    """Any YAML value as one line of text: lists and mappings are joined, None is empty."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return " ".join(_text(x) for x in v if _text(x))
    if isinstance(v, dict):
        return " ".join(f"{k}: {_text(x)}" for k, x in v.items())
    return " ".join(str(v).split())


def _number(key: str, v, notes: list[str]):
    kind, lo, hi = NUMBERS[key]
    if isinstance(v, bool) or v is None:
        raise ProcessError(f"interview.{key}: expected a number, got {v!r}")
    if isinstance(v, (int, float)):
        x = float(v)
    else:
        m = re.search(r"-?\d+(?:\.\d+)?", str(v))
        if not m:
            raise ProcessError(f"interview.{key}: expected a number, got {v!r}")
        x = float(m.group())
        if str(v).strip().endswith("%"):
            x /= 100
        notes.append(f"interview.{key}: {v!r} read as {x:g}")
    if kind is float and 1 < x <= 100:                     # a chance written as a percentage
        notes.append(f"interview.{key}: {x:g} read as {x / 100:g}")
        x /= 100
    if not lo <= x <= hi:
        notes.append(f"interview.{key}: {x:g} is outside {lo}-{hi}; using {min(max(x, lo), hi):g}")
        x = min(max(x, lo), hi)
    return int(round(x)) if kind is int else x


def _panel(raw, notes: list[str]) -> list[dict]:
    """A panel as a list of {role, people: [{name, voice}], opens, technical, observer}, whatever shape it came in."""
    roles = raw if isinstance(raw, list) else [raw] if raw else []
    out, used = [], set()
    for i, r in enumerate(roles, 1):
        if isinstance(r, str):
            r = {"role": r}
        if not isinstance(r, dict):
            notes.append(f"panel entry {i} ignored (not a role): {r!r}")
            continue
        role = _text(r.get("role") or r.get("title") or r.get("name")) or f"Interviewer {i}"
        if not r.get("role"):
            notes.append(f"panel entry {i}: no `role`, using {role!r}")
        people = r.get("people") or []
        people = people if isinstance(people, list) else [people]
        fixed = []
        for person in people:
            person = {"name": person} if isinstance(person, str) else person
            if isinstance(person, dict) and _text(person.get("name")):
                fixed.append({"name": _text(person["name"]).split()[0], "voice": person.get("voice")})
        if not fixed:
            fixed = [{"name": n} for n in ("Alex", "Sam", "Jordan")]
            notes.append(f"panel role {role!r}: no people with a name; using generic names")
        for person in fixed:
            if person.get("voice") not in VOICES or person["voice"] in used:
                person["voice"] = next((v for v in VOICES if v not in used), VOICES[len(used) % len(VOICES)])
            used.add(person["voice"])
        out.append({"role": role, "people": fixed, "opens": _truthy(r.get("opens")),
                    "technical": _truthy(r.get("technical")), "observer": _truthy(r.get("observer")),
                    **({"key": r["key"]} if r.get("key") else {})})
    if not out:
        if roles:
            notes.append("panel: no usable roles; using the default interviewer")
        return _panel([dict(r, people=[dict(x) for x in r["people"]]) for r in DEFAULTS["panel"]], notes)
    if all(r["observer"] for r in out):
        out[0]["observer"] = False
        notes.append(f"panel: everyone was an observer; {out[0]['role']!r} now speaks")
    return out


def _values(raw, notes: list[str]) -> dict:
    if not raw:
        return {}
    if isinstance(raw, (list, tuple)):
        raw = {_text(v): {} for v in raw if _text(v)}
    if not isinstance(raw, dict):
        notes.append(f"values ignored (expected names with descriptions): {raw!r}")
        return {}
    out = {}
    for name, v in raw.items():
        if isinstance(v, str):
            v = {"brief": v}
        elif not isinstance(v, dict):
            v = {}
        out[_text(name)] = {k: _text(v.get(k)) for k in ("brief", "probe", "red_flags") if _text(v.get(k))}
    return out


def _vocabulary(raw) -> list[str]:
    if isinstance(raw, str):
        raw = re.split(r"[,;\n]", raw)
    if not isinstance(raw, (list, tuple)):
        return []
    return [_text(x) for x in raw if _text(x)]


def load(pdir: Path) -> dict:
    """Tool config + process.yaml, normalized: every key the interview uses has the type it expects,
    whatever was written (text numbers, a string for a list, a bare name for a person...). What was changed
    is listed in cfg["load_notes"]; what can't be fixed raises ProcessError naming the key."""
    cfg = tool_config()
    notes: list[str] = []
    try:
        proc = yaml.safe_load((pdir / "process.yaml").read_text()) or {}
    except yaml.YAMLError as e:
        raise ProcessError(f"process.yaml is not valid YAML: {e}")
    if not isinstance(proc, dict):
        raise ProcessError("process.yaml must be a mapping of keys (company: ..., role: ...)")
    for k in OVERRIDABLE:
        if k in proc and proc[k] is not None:
            if isinstance(cfg.get(k), dict) and not isinstance(proc[k], dict):
                notes.append(f"`{k}:` ignored (expected settings like `minutes: 30`, got {proc[k]!r})")
                continue
            cfg[k] = _merge(cfg[k], proc[k]) if isinstance(cfg.get(k), dict) else proc[k]
    ic = cfg["interview"]
    for key in NUMBERS:
        if key in ic:
            ic[key] = _number(key, ic[key], notes)
    if ic["candidate_minutes"] >= ic["minutes"]:
        ic["candidate_minutes"] = max(0, ic["minutes"] // 4)
        notes.append(f"interview.candidate_minutes: must be less than minutes; using {ic['candidate_minutes']}")
    t = ic.get("target_answer_seconds")
    try:
        lo, hi = (float(x) for x in t)
        ic["target_answer_seconds"] = [int(lo), int(hi)] if 0 < lo < hi else [60, 120]
    except (TypeError, ValueError):
        ic["target_answer_seconds"] = [60, 120]
        notes.append(f"interview.target_answer_seconds: expected [low, high], got {t!r}; using [60, 120]")

    p = {}
    for k, v in DEFAULTS.items():
        raw = proc.get(k)
        p[k] = v if raw is None or raw == "" and k == "company" else raw
    if "candidate_level" not in proc and ic.get("candidate_level"):     # older configs kept it under interview:
        p["candidate_level"] = ic["candidate_level"]
    for k in TEXT_KEYS:
        before = p[k]
        p[k] = _text(p[k]) if p[k] is not None else ""
        if before is not None and not isinstance(before, (str, int, float)):
            notes.append(f"{k}: {type(before).__name__} joined into text")
        if not p[k] and DEFAULTS.get(k) and k not in ("profile", "notes", "questions"):
            p[k] = DEFAULTS[k]
    p["interviewer_from"] = p["interviewer_from"] or p["team"]
    p["panel"] = _panel(proc.get("panel"), notes)
    p["values"] = _values(proc.get("values"), notes)
    p["vocabulary"] = _vocabulary(proc.get("vocabulary"))
    p["dir"] = pdir
    cfg["process"] = p
    cfg["panel"] = p["panel"]
    cfg["values"] = p["values"]
    cfg["stt"]["vocabulary"] = p["vocabulary"]
    ic.setdefault("candidate_level", p["candidate_level"])
    cfg["load_notes"] = notes
    return cfg


def _read(pdir: Path, name: str) -> str:
    if not name:
        return ""
    path = (pdir / name).expanduser()
    return path.read_text() if path.exists() else ""


def profile(cfg: dict) -> str:
    """The candidate profile, or "" when the process has none (`profile: ""`): the interviewer then
    learns about the candidate from the call itself."""
    name = cfg["process"]["profile"]
    text = _read(cfg["process"]["dir"], name)
    if name and not text.strip():
        print(f"\033[1;33mNo profile at {cfg['process']['dir'] / name}: continuing without one "
              f"(set `profile: \"\"` in process.yaml to silence this).\033[0m")
    return text.strip()


def notes(cfg: dict) -> str:
    return _read(cfg["process"]["dir"], cfg["process"]["notes"])


_PLACEHOLDER = re.compile(r"\{(company|team|role)\}")


def _fill(text: str, p: dict) -> str:
    """Generic wordings say {company}, {team} or {role}; fill them from the process."""
    return _PLACEHOLDER.sub(lambda m: p[m.group(1)], text)


def _questions(raw, where: str, notes: list[str]) -> list[dict]:
    """Questions as {id, theme, value, core, q: [wordings]}; entries with no usable wording are dropped."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ProcessError(f"{where}: `questions:` must be a list of questions")
    out, ids = [], set()
    for i, q in enumerate(raw, 1):
        if isinstance(q, str):
            q = {"q": [q]}
        if not isinstance(q, dict):
            notes.append(f"{where}: question {i} ignored (not a question): {q!r}")
            continue
        w = q.get("q") if q.get("q") is not None else q.get("question") or q.get("wordings")
        w = w if isinstance(w, list) else [w]
        w = [_text(x) for x in w if _text(x)]
        if not w:
            notes.append(f"{where}: question {q.get('id') or i} has no wording; dropped")
            continue
        qid = re.sub(r"\W+", "_", _text(q.get("id")).lower()).strip("_") or f"q{i}"
        while qid in ids:
            qid += "_2"
        ids.add(qid)
        out.append({**{k: v for k, v in q.items() if k not in ("q", "question", "wordings")},
                    "id": qid, "theme": re.sub(r"\W+", "_", _text(q.get("theme")).lower()).strip("_") or "general",
                    "value": _text(q.get("value")), "core": _truthy(q.get("core")), "q": w})
    return out


def bank(cfg: dict) -> dict:
    """The process's questions.yaml, or the tool's generic bank. A process bank with
    `extends_default: true` adds to the generic one (same id = replaced; ids under `exclude:` dropped).
    Normalized like load(): every question has an id, a theme, a value (maybe "") and a list of wordings."""
    p = cfg["process"]
    notes = cfg.setdefault("load_notes", [])
    default = yaml.safe_load(DEFAULT_BANK.read_text())
    own = None
    if p["questions"]:
        path = (p["dir"] / p["questions"]).expanduser()
        if not path.exists():
            raise ProcessError(f"Question bank not found: {path}")
        text = path.read_text()
        if not text.strip():
            raise ProcessError(f"{p['questions']} is empty: add questions, or remove `questions:` from process.yaml "
                               "to use the default bank")
        try:
            own = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise ProcessError(f"{p['questions']} is not valid YAML: {e}")
        if isinstance(own, list):                    # a bare list of questions
            own = {"questions": own}
        if not isinstance(own, dict):
            raise ProcessError(f"{p['questions']} must be a mapping with `questions:`")
    defaults = _questions(default["questions"], "default bank", notes)
    if own is None:
        b = {"intro": default["intro"], "questions": defaults}
    else:
        mine = _questions(own.get("questions"), p["questions"], notes)
        for q in mine:
            q["own"] = True                          # process-specific: asked before the generic bank
        if _truthy(own.get("extends_default")):
            excl = own.get("exclude") or []
            ids = {q["id"] for q in mine} | {_text(x) for x in (excl if isinstance(excl, list) else [excl])}
            b = {"questions": [q for q in defaults if q["id"] not in ids] + mine}
        else:
            b = {"questions": mine}
        b["intro"] = own.get("intro") or default["intro"]
    if not b["questions"]:
        raise ProcessError(f"{p['questions'] or 'the question bank'} has no usable questions")
    known = set(cfg.get("values") or {})
    for q in b["questions"]:
        q["q"] = [_fill(w, p) for w in q["q"]]
        if q["value"] and known and q["value"] not in known:
            notes.append(f"question {q['id']}: value {q['value']!r} is not a process value; ignored")
            q["value"] = ""
    intro = b["intro"] if isinstance(b["intro"], list) else [b["intro"]]
    b["intro"] = [_fill(_text(w), p) for w in intro if _text(w)] or [_fill(w, p) for w in default["intro"]]
    return b


def init(target: str) -> None:
    dst = Path(target).expanduser().resolve()
    if (dst / "process.yaml").exists():
        raise SystemExit(f"{dst} already has a process.yaml; not overwriting.")
    dst.mkdir(parents=True, exist_ok=True)
    for f in TEMPLATE.iterdir():
        if not (dst / f.name).exists():
            shutil.copy(f, dst / f.name)
    print(f"New process in {dst}. Fill in process.yaml and profile.md, then run from there:\n"
          f"  {ROOT / 'ainterviewer'} interview")
