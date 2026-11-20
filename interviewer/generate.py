"""`ainterviewer init <dir> -d "..." -f job_ad.txt -f cv.pdf`: builds a process folder from what the user
says about the recruitment process (and optional files: job ad, recruiter email, candidate reports, CV).

One LLM call writes process.yaml, questions.yaml (company-specific questions on top of the default bank)
and, if the sources say anything about the candidate, profile.md. The result is validated by loading
it exactly as an interview would; on an error the model gets one retry with the error message.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import yaml

from . import process as P

VOICES = ["af_heart", "af_bella", "af_nicole", "bf_emma", "am_michael", "am_fenrir", "bm_george", "bm_fable"]
MAX_SOURCE_CHARS = 60_000

SYSTEM = """You set up a recruitment-process folder for AInterviewer, a voice mock-interview tool. The candidate describes the interview they are preparing for; you turn it into the tool's config files.

Hard rules:
- Use only what the sources say. Never invent facts about the company, team, product, numbers, values, interview format or the candidate. Leave a key out when the sources don't give it: the tool then uses a sensible generic default. A wrong "fact" makes the mock interviewer mislead the candidate; a missing one does not.
- Interviewer names are the exception: they are fictional voices. Give 2-4 people per panel role, with distinct first names.
- Where something useful is missing, add a short YAML comment starting "# TODO:" saying what to find out (e.g. "# TODO: ask the recruiter how long the round is").
- Copy every number exactly as the sources give it (900 ms stays 900 ms). Never round, shorten or recompute a figure.
- Write the folded text fields (format, assess, good_answers, candidate_level, team_facts, ...) as plain sentences the interviewer model will read; keep each under ~120 words. Write interviewer_rules as instructions to the interviewer about "the candidate", never "you".
- Always include `panel:` with the interviewer roles the sources describe, by the role they play in this round (e.g. "Bar Raiser"). Someone who only observes, shadows or takes notes gets `observer: true` (introduced, never speaks). If the sources describe no interviewers, use one "Hiring Manager" role and a TODO.
- interviewer_rules: only what the interviewer must ask or avoid in this process (e.g. "the candidate is still employed, so don't ask why they left"). Reasons candidates fail go in known_failure_reasons, not here.
- Put timings and probing depth the sources give under `interview:`: minutes (total length), candidate_minutes (time left for the candidate's questions), max_followups (follow-up layers per story).
- If the sources name company values or principles, list every one of them under `values:`. For each, `brief` is what the sources say about it; if they give only the name, write a plain reading of the name and put "# TODO: replace with the official wording" above it. Never drop a named value.

The process.yaml format, as a commented example (keys and meaning; values are only an example):
{template}

Panel voices (use each at most once across the whole panel): {voices}. Mark `technical: true` on a role that would ask how things worked technically (an engineer). `opens: true` on the role that runs the call.

Default question bank (already included; don't repeat these, and list in `exclude` any that make no sense for this process):
{bank}

Output exactly these four sections and nothing else, no code fences:
=== process.yaml ===
(the YAML; `questions: questions.yaml`; `profile: profile.md` if you write a profile, else `profile: ""`; `notes: ""`)
=== questions.yaml ===
(YAML with `extends_default: true`, optional `exclude: [ids from the default bank]`, and `questions:` with 3-10 questions specific to this company, role or round: motivation for this company, every question the sources report being asked (its first wording copied word for word from the sources), role-specific behavioural ones. Each: id (snake_case, new), theme (reuse a default theme where it fits), value (only if the process has values: exactly one of their names), core: true for topics the recruiter said are covered, q: 2-3 alternative wordings of the SAME question, short and spoken, about past experience; the tool picks one at random and generates follow-ups itself, so never put a follow-up question in q)
=== profile.md ===
(only if the sources describe the candidate: "Name: <first name>." then background, 2-4 key stories with their numbers copied exactly, what they're applying to, one known weak spot if given; under 120 words, facts only. Otherwise the single word NONE)
=== summary ===
(3-6 short lines for the candidate: what you used, what is generic or missing and worth filling in)"""


def read_source(path: Path) -> str:
    """Text of a job ad / email / CV: .txt/.md/.yaml/.html/.json as text, .pdf via pdftotext, .docx unzipped."""
    path = path.expanduser()
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        if not shutil.which("pdftotext"):
            raise SystemExit(f"{path.name}: reading PDFs needs `pdftotext` (poppler). Install it, or save the text "
                             f"as a .txt file.")
        return subprocess.run(["pdftotext", "-layout", str(path), "-"], capture_output=True, text=True,
                              check=True).stdout
    if suffix == ".docx":
        with zipfile.ZipFile(path) as z:
            xml = z.read("word/document.xml").decode("utf-8", "replace")
        xml = re.sub(r"</w:p>", "\n", xml)
        return re.sub(r"<[^>]+>", "", xml)
    text = path.read_text(errors="replace")
    if suffix in (".html", ".htm"):
        text = re.sub(r"(?is)<(script|style).*?</\1>", "", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
    return text


def ask_description() -> str:
    print("Describe the recruitment process: company, role, which round, who interviews you, how long, what\n"
          "the recruiter said they look for, questions candidates report, anything about you they should know.\n"
          "Paste freely; finish with Ctrl-D on an empty line.\n")
    try:
        return sys.stdin.read().strip()
    except KeyboardInterrupt:
        return ""


def _bank_summary() -> str:
    bank = yaml.safe_load(P.DEFAULT_BANK.read_text())
    return "\n".join(f"- {q['id']} ({q['theme']}{', core' if q.get('core') else ''}): "
                     f"{q['q'][0] if isinstance(q['q'], list) else q['q']}" for q in bank["questions"])


def _sections(text: str) -> dict[str, str]:
    parts = re.split(r"^\s*=== *([\w.]+) *=== *$", text, flags=re.M)
    out = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        body = re.sub(r"^\s*```[\w-]*\s*\n|\n\s*```\s*$", "", body.strip("\n")).strip()
        out[name.strip().lower()] = body
    return out


_NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
    "seventeen eighteen nineteen twenty".split())}
_QUESTION_START = re.compile(r"^(tell me|tell us|describe|walk me|walk us|give me|give an|what|why|how|when|have you|"
                             r"can you|could you|do you|did you|share)\b", re.I)


def _numbers(text: str) -> set[str]:
    t = re.sub(r"\b(" + "|".join(_NUM_WORDS) + r")\b", lambda m: _NUM_WORDS[m.group(1).lower()], text, flags=re.I)
    return set(re.findall(r"\d+(?:\.\d+)?", t.replace(",", "")))


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def _reported_questions(sources: str) -> list[str]:
    """Questions quoted in the sources ("Tell me about a time ..."): these must be asked word for word."""
    out = []
    # pair quotes strictly in order (a short quote like "we" must not shift the pairing)
    quoted = re.findall(r'"([^"]*)"', sources) + re.findall(r"“([^”]*)”", sources)
    for raw in quoted:
        q = " ".join(raw.split())
        if 12 <= len(q) <= 240 and (q.endswith("?") or _QUESTION_START.match(q)) and q not in out:
            out.append(q)
    return out


def _fix(proc: dict, questions: dict, sources: str) -> list[str]:
    """Mechanical repairs instead of a retry (a retry can make a small model delete content to pass):
    voices, values used by questions but not defined, reported questions not kept word for word."""
    notes, used = [], set()
    roles = [r for r in (proc.get("panel") if isinstance(proc.get("panel"), list) else []) if isinstance(r, dict)]
    for role in roles:
        for person in [x for x in (role.get("people") if isinstance(role.get("people"), list) else [])
                       if isinstance(x, dict)]:
            v = person.get("voice")
            if v not in VOICES or v in used:
                nv = next((x for x in VOICES if x not in used), VOICES[len(used) % len(VOICES)])
                notes.append(f"voice {v!r} -> {nv}")
                person["voice"] = v = nv
            used.add(v)
    for role in roles:
        if not role.get("observer") and re.search(r"shadow|observer|note.?taker", str(role.get("role")), re.I):
            role["observer"] = True
            notes.append(f"{role.get('role')!r} only observes")
    proc.setdefault("questions", "questions.yaml")
    proc.setdefault("notes", "")
    questions.setdefault("extends_default", True)
    qs = questions.get("questions") or []
    qs = qs if isinstance(qs, list) else []
    # a cut-off list can end in an empty "- " item; a bare string is a question with one wording
    qs = [{"q": [q]} if isinstance(q, str) else q for q in qs if isinstance(q, (dict, str)) and q]
    questions["questions"] = qs
    for q in qs:
        w = q.get("q") if isinstance(q.get("q"), list) else [q.get("q")]
        q["q"] = [str(x).strip() for x in w if isinstance(x, (str, int, float)) and str(x).strip()]
        if not isinstance(q.get("value"), (str, type(None))):        # e.g. a whole value definition
            notes.append(f"question {q.get('id')}: `value` must be a value's name; cleared")
            q["value"] = ""
        if not isinstance(q.get("theme"), (str, type(None))):
            q["theme"] = str(q.get("theme"))
    default_ids = {q["id"] for q in yaml.safe_load(P.DEFAULT_BANK.read_text())["questions"]}
    if questions.get("exclude"):
        excl = questions["exclude"] if isinstance(questions["exclude"], list) else [questions["exclude"]]
        questions["exclude"] = [x for x in excl if isinstance(x, str) and x in default_ids]
    # values: a question's value must exist; add it if the sources name it, else drop the tag
    vals = proc.get("values") if isinstance(proc.get("values"), dict) else {}
    for q in qs:
        v = q.get("value")
        if v and v not in vals:
            if " ".join(v.lower().split()) in " ".join(sources.lower().split()):
                vals[v] = {"brief": ""}
                notes.append(f"added value {v!r} named in the sources (describe it)")
            else:
                q["value"] = ""
                notes.append(f"dropped unknown value {v!r} from {q.get('id')}")
    if vals:
        proc["values"] = vals
    # reported questions: first wording word for word, on the most similar generated question or a new one
    for i, rq in enumerate(_reported_questions(sources), 1):
        if any(_norm(rq) == _norm(w) for q in qs for w in q["q"]):
            continue
        words = set(_norm(rq).split())
        best, score = None, 0.0
        for q in qs:
            for w in q["q"]:
                ww = set(_norm(w).split())
                j = len(words & ww) / max(1, len(words | ww))
                if j > score:
                    best, score = q, j
        if best is not None and score >= 0.4:
            best["q"] = [rq] + [w for w in best["q"] if _norm(w) != _norm(rq)][:2]
            notes.append(f"kept reported wording on {best.get('id')}")
        else:
            qs.append({"id": f"reported_{i}", "theme": f"reported_{i}", "core": True, "q": [rq]})
            notes.append(f"added reported question {rq!r}")
    notes += _split_themes(questions)
    return notes


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _split_themes(questions: dict) -> list[str]:
    """The picker asks one core question per theme. A model that gives every value question the same theme
    ("leadership_principle") would get one of them asked, so such a theme is split by value."""
    notes, by_theme = [], {}
    for q in questions.get("questions") or []:
        if q.get("core"):
            by_theme.setdefault(str(q.get("theme")), []).append(q)
    for theme, qs in list(by_theme.items()):
        values = {q.get("value") for q in qs if isinstance(q.get("value"), str) and q.get("value")}
        if len(qs) >= 3 and len(values) >= 2:
            for q in qs:
                if isinstance(q.get("value"), str) and q.get("value"):
                    q["theme"] = _slug(q["value"])
            notes.append(f"split theme {theme!r} by value")
    return notes


def _unsourced_values(proc: dict, sources: str) -> list[str]:
    """Values whose description isn't backed by the sources (written from the model's general knowledge)."""
    src = set(re.findall(r"[a-z]{4,}", sources.lower()))
    out = []
    vals = proc.get("values") if isinstance(proc.get("values"), dict) else {}
    for name, v in vals.items():
        brief = v.get("brief", "") if isinstance(v, dict) else v
        words = set(re.findall(r"[a-z]{4,}", str(brief or "").lower())) - set(str(name).lower().split())
        if words and len(words & src) / len(words) < 0.5:
            out.append(name)
    return out


def _todos(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip().startswith("# TODO")]


def _keep_comments(text: str, proc: dict, extra_todos: list[tuple[str, str]]) -> str:
    """The model's YAML text with the repairs applied in place, so its comments survive: voices, top-level
    keys (replaced, never duplicated), `observer: true` lines, and extra TODO comments placed above their key
    (`extra_todos`: (key, comment)). When the text can't be patched to match, a clean dump with every TODO
    comment kept at the top."""
    voices = iter([person.get("voice") for role in (proc.get("panel") if isinstance(proc.get("panel"), list) else [])
                   if isinstance(role, dict)
                   for person in (role.get("people") if isinstance(role.get("people"), list) else [])
                   if isinstance(person, dict)])
    t = re.sub(r"(\bvoice:\s*[\"']?)([\w-]+)", lambda m: m.group(1) + next(voices, m.group(2)), text)
    for role in [r for r in (proc.get("panel") if isinstance(proc.get("panel"), list) else []) if isinstance(r, dict)]:
        if role.get("observer"):
            m = re.search(rf"^(\s*)- role:\s*[\"']?{re.escape(str(role.get('role')))}[\"']?\s*$", t, re.M)
            if m and not re.search(r"^\s*observer:", t[m.end():].split("\n  - ", 1)[0], re.M):
                t = t[:m.end()] + f"\n{m.group(1)}  observer: true" + t[m.end():]
            elif m:
                t = re.sub(r"(observer:\s*)\w+", r"\1true", t, count=1) if "observer: false" in t else t
    for key in ("profile", "questions", "notes"):
        lines = [i for i, line in enumerate(t.split("\n")) if re.match(rf"{key}:", line)]
        want = f"{key}: {json.dumps(proc.get(key))}"
        rows = t.split("\n")
        if lines:
            rows[lines[0]] = want
            for i in reversed(lines[1:]):                    # never leave a duplicate key behind
                del rows[i]
        else:
            rows.append(want)
        t = "\n".join(rows)
    for key, comment in extra_todos:
        t, n = re.subn(rf"^({key}:)", comment + r"\n\1", t, count=1, flags=re.M)
        if not n:
            t = comment + "\n" + t
    try:
        if yaml.safe_load(t) == proc and not _MergingLoader.merged:
            return t.rstrip("\n") + "\n"
    except yaml.YAMLError:
        pass
    todos = _todos(text) + [c for _, c in extra_todos]
    return ("\n".join(todos) + "\n\n" if todos else "") + \
        yaml.safe_dump(proc, sort_keys=False, allow_unicode=True, width=110)


def _invented_numbers(proc: dict, profile: str, sources: str) -> list[str]:
    """Numbers in the facts the interviewer and debrief will trust (profile, team facts, format...) that the
    sources never mention: a 9B model once turned "900 ms to 250 ms" into "900ms to 25ms"."""
    known = _numbers(sources)
    fields = {k: str(proc.get(k) or "") for k in ("team_facts", "format", "candidate_level", "round_description",
                                                  "closing", "known_failure_reasons")}
    fields["profile.md"] = profile
    minutes = (proc.get("interview") or {}).get("minutes")
    if minutes is not None:
        fields["interview.minutes"] = str(minutes)
    out = []
    for name, text in fields.items():
        bad = sorted(_numbers(text) - known, key=float)
        if bad:
            out.append(f"{name}: {', '.join(bad)}")
    return out


def _stems(text: str) -> set[str]:
    return {w[:4] for w in re.findall(r"[a-z]{4,}", text.lower())}


def _unsupported_sentences(proc: dict, profile: str, sources: str) -> list[str]:
    """Sentences in the profile and team facts whose content words mostly don't appear in the sources:
    likely invented (a model once listed a "team conflict resolution" story nobody mentioned)."""
    src = _stems(sources)
    skip = {"name", "background", "applying", "known", "weak", "spot", "stories", "story", "available", "story",
            "candidate", "profile", "shown", "interviewer", "every", "turn", "under", "words", "keep"}
    out = []
    for where, text in (("profile.md", profile), ("team_facts", str(proc.get("team_facts") or ""))):
        for sent in re.split(r"(?<=[.!?])\s+|\n+|;\s+|,\s+(?=[A-Z])", text):
            words = _stems(sent) - {w[:4] for w in skip}
            if len(words) >= 3 and len(words & src) / len(words) < 0.4:
                out.append(f"{where}: {sent.strip()[:120]}")
    return out


def _validate(folder: Path, sources: str) -> str:
    """Load the folder exactly as an interview would, then dry-run a whole interview, drill and debrief on it
    (selftest): a process that is written is a process that runs."""
    from .selftest import dry_run
    cfg = P.load(folder)
    bank = P.bank(cfg)
    profile = P.profile(cfg)
    dry_run(cfg, bank, profile, P.notes(cfg))
    vals = set(cfg["values"])
    bad = [q["id"] for q in bank["questions"] if q.get("value") and vals and q["value"] not in vals]
    if bad:
        raise ValueError(f"questions {bad} use a `value` that is not one of the process values {sorted(vals)}")
    missing = [q.get("id", "?") for q in bank["questions"] if not (q.get("id") and q.get("theme") and q.get("q"))]
    if missing:
        raise ValueError(f"questions without id/theme/q: {missing}")
    info = f"{len(bank['questions'])} questions, {len(cfg['values'])} values, panel: " + \
        ", ".join(r["role"] + (" (observer)" if r.get("observer") else "") for r in cfg["panel"])
    if cfg.get("load_notes"):
        info += "\nRead as: " + "; ".join(dict.fromkeys(cfg["load_notes"]))
    return info


class _MergingLoader(yaml.SafeLoader):
    """SafeLoader that keeps repeated keys instead of silently dropping all but the last (a small model wrote
    `brief:` twice per value): repeated text values are joined, and each merge is recorded in `merged`."""
    merged: list[str] = []

    def construct_mapping(self, node, deep=False):
        out = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            if key in out and isinstance(out[key], str) and isinstance(value, str) and value != out[key]:
                out[key] = f"{out[key].rstrip('. ')}. {value}"
                _MergingLoader.merged.append(str(key))
            else:
                out[key] = value
        return out


_MergingLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _MergingLoader.construct_mapping)


def _load(text: str):
    return yaml.load(text, Loader=_MergingLoader)


_SCALAR_LINE = re.compile(r"^(\s*(?:-\s+)?[\w.-]+:\s+|\s*-\s+)(.+?)\s*$")


_KEY_LINE = re.compile(r"^(\s*)(?:-\s+)?[A-Za-z_][\w .'&;/-]{0,60}:(?:\s|$)")


def _indent_continuations(text: str) -> str:
    """Indent lines that continue the previous key's value but start at (or left of) the key's column."""
    rows, out, key_indent = text.split("\n"), [], None
    for row in rows:
        stripped = row.strip()
        m = _KEY_LINE.match(row)
        if m:
            key_indent = len(m.group(1))
        elif (stripped and key_indent is not None and not stripped.startswith(("#", "- ", "-\t"))
              and len(row) - len(row.lstrip()) <= key_indent):
            row = " " * (key_indent + 2) + stripped
        out.append(row)
    return "\n".join(out)


def _load_yaml(text: str):
    """yaml.safe_load, with one mechanical repair for small models' quoting mistakes
    (probe: '"Why didn't you stop?"' and the like): every scalar value that doesn't parse on its own
    is re-quoted as a plain double-quoted string."""
    try:
        return _load(text), text
    except yaml.YAMLError:
        pass
    rows = text.split("\n")
    for i, row in enumerate(rows):
        m = _SCALAR_LINE.match(row)
        if not m or row.lstrip().startswith("#") or m.group(2)[:1] in "[{|>&*!#":
            continue
        try:
            ok = isinstance(yaml.safe_load("k: " + m.group(2)), dict)
        except yaml.YAMLError:
            ok = False
        if not ok:
            v = m.group(2)
            if len(v) > 1 and v[0] == v[-1] and v[0] in "'\"":
                v = v[1:-1]
            rows[i] = m.group(1) + json.dumps(v, ensure_ascii=False)
    fixed = "\n".join(rows)
    try:
        return _load(fixed), fixed
    except yaml.YAMLError:
        pass
    # continuation lines of a multi-line value written without indentation ("key: >-" then text at column 0)
    fixed = _indent_continuations(fixed)
    try:
        return _load(fixed), fixed
    except yaml.YAMLError:
        pass
    # tabs used for indentation (not allowed in YAML)
    fixed = "\n".join(re.sub(r"^\t+", lambda m: "  " * len(m.group()), r) for r in fixed.split("\n"))
    try:
        return _load(fixed), fixed
    except yaml.YAMLError as e:
        err = e
    # output cut off by a length cap: drop the unfinished last lines, but only when the error is at the end
    mark = getattr(err, "problem_mark", None)
    rows = fixed.rstrip().split("\n")
    if mark is not None and mark.line >= len(rows) - 3:
        for cut in range(1, min(12, len(rows))):
            try:
                d = _load("\n".join(rows[:-cut]))
                _MergingLoader.merged.append(f"cut-off tail ({cut} lines dropped)")
                return d, "\n".join(rows[:-cut]) + "\n"
            except yaml.YAMLError:
                continue
    raise err                                            # still broken: the model gets a retry


def _build(raw: str, sources: str) -> dict:
    """Parse, repair and check one model output. Raises ValueError on a problem that needs a retry; returns
    the files plus `soft` problems (invented numbers) that a retry may fix but that don't block the result."""
    sec = _sections(raw.replace("\r\n", "\n").replace("\r", "\n"))
    _MergingLoader.merged = []
    if "process.yaml" not in sec:
        raise ValueError("no '=== process.yaml ===' section in the output")
    proc, sec["process.yaml"] = _load_yaml(sec["process.yaml"])
    proc = proc or {}
    questions, sec["questions.yaml"] = _load_yaml(sec.get("questions.yaml") or "")
    questions = questions or {"extends_default": True}
    if not isinstance(proc, dict) or not proc.get("company"):
        raise ValueError("process.yaml must be a mapping with at least `company`")
    if not isinstance(questions, dict):
        raise ValueError("questions.yaml must be a mapping")
    if not proc.get("panel"):
        raise ValueError("process.yaml has no `panel:`; add the interviewer roles the sources describe")
    profile = sec.get("profile.md", "").strip()
    has_profile = bool(profile) and profile.upper() != "NONE"
    proc["profile"] = "profile.md" if has_profile else ""
    try:
        files, fixes = _repair(sec, proc, questions, sources)
    except Exception as e:                           # noqa: BLE001 - repairs are best-effort; the check below decides
        files = {"process.yaml": ("\n".join(_todos(sec["process.yaml"])) + "\n\n" if _todos(sec["process.yaml"]) else "")
                 + yaml.safe_dump(proc, sort_keys=False, allow_unicode=True, width=110),
                 "questions.yaml": yaml.safe_dump(questions, sort_keys=False, allow_unicode=True, width=110)}
        fixes = [f"automatic repairs skipped ({type(e).__name__}: {e}); the files were checked as written"]
    if has_profile:
        files["profile.md"] = profile + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        for name, content in files.items():
            (Path(tmp) / name).write_text(content)
        info = _validate(Path(tmp), sources)
    def safe(fn):                                    # the content checks are advisory: never block a sound draft
        try:
            return fn(proc, profile if has_profile else "", sources)
        except Exception:                            # noqa: BLE001
            return []
    return {"files": files, "info": info, "fixes": fixes, "summary": sec.get("summary", ""),
            "soft": safe(_invented_numbers), "unsupported": safe(_unsupported_sentences)}


def _repair(sec: dict, proc: dict, questions: dict, sources: str) -> tuple[dict, list[str]]:
    """The mechanical repairs, applied to the model's text where possible so its comments survive."""
    before_q = json.dumps(questions, sort_keys=True, default=str)
    known = _numbers(sources)
    guessed = []
    if isinstance(proc.get("interview"), dict):        # a length or limit the sources never gave is a guess
        for k, v in list(proc["interview"].items()):
            nums = _numbers(str(v)) if isinstance(v, (int, float, str)) else set()
            if nums and not nums <= known:
                guessed.append(f"interview.{k}: {v}")
                del proc["interview"][k]
        if not proc["interview"]:
            del proc["interview"]
    added_values = list(proc["values"]) if isinstance(proc.get("values"), dict) else []
    fixes = _fix(proc, questions, sources)
    fixes += [f"removed a guessed setting ({g}); the default applies" for g in guessed]
    cut = [m for m in _MergingLoader.merged if m.startswith("cut-off")]
    merged = [m for m in _MergingLoader.merged if not m.startswith("cut-off")]
    if merged:
        fixes.append(f"merged {len(merged)} repeated keys ({', '.join(sorted(set(merged)))})")
    fixes += [f"the model's output was cut off: {c}" for c in cut]
    new_values = [v for v in (proc.get("values") or {}) if v not in added_values]
    extra = [("values", f"# TODO: describe the value {v!r} (brief, probe, red_flags); the model left it out")
             for v in new_values]
    unsourced = [v for v in _unsourced_values(proc, sources) if v not in new_values]
    if unsourced and not re.search(r"#\s*TODO.*(official|wording)", sec["process.yaml"], re.I):
        extra.append(("values", "# TODO: these value descriptions are not in your sources (the model wrote them "
                                "from general knowledge): check them against the official wording"))
        fixes.append(f"flagged {len(unsourced)} value descriptions not found in the sources")
    files = {"process.yaml": _keep_comments(sec["process.yaml"], proc, extra),
             "questions.yaml": (sec.get("questions.yaml", "").rstrip() + "\n"
                                if json.dumps(questions, sort_keys=True, default=str) == before_q
                                and sec.get("questions.yaml")
                                else yaml.safe_dump(questions, sort_keys=False, allow_unicode=True, width=110))}
    return files, fixes


def _count(soft: list[str]) -> int:
    """Invented numbers in `soft` entries like "profile.md: 4, 25"."""
    return sum(len(x.split(":", 1)[-1].split(",")) for x in soft)


RETRY = ("Checking those files found a problem: {error}\n"
         "Fix only that. Keep everything else exactly as it was: never delete values, questions, panel roles or "
         "facts to make the check pass. Output all four sections again.")


def generate(dst: Path, description: str, files: list[str], llm, model: str) -> None:
    sources = []
    if description:
        sources.append(f"## The candidate's description\n{description}")
    for f in files:
        sources.append(f"## File: {Path(f).name}\n{read_source(Path(f)).strip()}")
    text = "\n\n".join(sources)
    if len(text) > MAX_SOURCE_CHARS:
        raise SystemExit(f"The description and files are {len(text):,} characters; keep them under "
                         f"{MAX_SOURCE_CHARS:,} (trim long files to the relevant parts).")

    prov, model_id = llm.resolve(model)
    what = "your description and files" if files and description else ("your files" if files else "your description")
    plural = bool(files)
    where = ((f"stay{'' if plural else 's'} on this machine") if llm.providers[prov].get("local")
             else f"{'are' if plural else 'is'} sent to {prov}")
    print(f"Writing the process with {prov}: {model_id} ({what} {where})...", flush=True)
    system = SYSTEM.format(template=(P.TEMPLATE / "process.yaml").read_text(), voices=", ".join(VOICES),
                           bank=_bank_summary())
    msgs = [{"role": "user", "content": f"Sources:\n\n{text}"}]
    best, raw, error = None, "", ""
    for attempt in range(2):
        if error:
            msgs = msgs[:1] + [{"role": "assistant", "content": raw},
                               {"role": "user", "content": RETRY.format(error=error)}]
        # continue only a reply that was cut off: it has the files' start but not their end. A reply without
        # any section is not cut off, it is wrong, and gets a retry instead.
        done = lambda t: (not re.search(r"^\s*=== *process\.yaml *===", t, re.M | re.I)
                          or re.search(r"^\s*=== *(summary|profile\.md) *===", t, re.M | re.I))
        raw = llm.complete_until(system, msgs, done, rounds=1,
                                 label="init", model=model, max_tokens=16000, effort="medium")
        try:
            result = _build(raw, text)
        except Exception as e:                       # noqa: BLE001 - any failure gets one retry
            error = f"{type(e).__name__}: {e}"
            print(f"\033[2m[draft {attempt + 1} didn't load ({error})"
                  f"{'; retrying' if attempt == 0 else ''}]\033[0m", flush=True)
            continue
        if best is None or _count(result["soft"]) < _count(best["soft"]):
            best = result
        if not result["soft"]:
            break
        error = ("these numbers are not in the sources, so they are wrong or invented: "
                 + "; ".join(result["soft"]) + ". Use the exact figures from the sources, or leave them out.")
        print(f"\033[2m[draft {attempt + 1}: {error}{' Retrying.' if attempt == 0 else ''}]\033[0m", flush=True)
    if best is None:
        dump = dst.with_name(dst.name + "-init-output.txt")
        dump.write_text(raw)
        raise SystemExit(f"Could not build a valid process ({error}). The model's output is in {dump}; "
                         f"or run `ainterviewer init {dst} --template` and fill it in by hand.")

    dst.mkdir(parents=True, exist_ok=True)
    for name, content in best["files"].items():
        (dst / name).write_text(content)
    # keep what the user said, so the process can be regenerated or checked later (file contents are not copied)
    (dst / "sources.md").write_text("# What this process was generated from\n\n"
                                    + (f"{description}\n\n" if description else "")
                                    + "".join(f"- file: {Path(f).expanduser().resolve()}\n" for f in files))
    print(f"\nNew process in {dst}: {best['info']}")
    if best["fixes"]:
        print("Repaired: " + "; ".join(best["fixes"]))
    if best["soft"]:
        print(f"\033[1;33mCheck these numbers, they are not in your sources: {'; '.join(best['soft'])}\033[0m")
    if best["unsupported"]:
        print("\033[1;33mCheck these lines, your sources don't clearly say this:\033[0m\n"
              + "\n".join(f"  - {u}" for u in best["unsupported"]))
    if best["summary"]:
        print("\n" + best["summary"])
    print(f"\nReview process.yaml (look for TODO), then from that folder:\n"
          f"  {P.ROOT / 'ainterviewer'} show        # what the interviewer will be told\n"
          f"  {P.ROOT / 'ainterviewer'} interview")
