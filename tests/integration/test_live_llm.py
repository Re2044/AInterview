"""Live model checks: a real interview, debrief, drill and init against a real provider, with detectors for the
ways models drift from their instructions. Each test is one rule, so a run lists every misalignment found.

Opt in (costs tokens on paid providers; ~10 min on a local 9B model):
    AINTERVIEWER_LIVE=http://127.0.0.1:8000/v1 .venv/bin/python -m pytest -m live
    AINTERVIEWER_LIVE=claude-cli .venv/bin/python -m pytest -m live
    AINTERVIEWER_LIVE_MODEL=<model> to pick a model (default: the provider's default / first local model)
"""
import os
import random
import re
import types

import pytest
import yaml

import interviewer.generate as G
import interviewer.process as P
import interviewer.session as S
from interviewer.__main__ import llm_config
from interviewer.llm import LLM
from tests.helpers import ScriptedIO, make_process

LIVE = os.environ.get("AINTERVIEWER_LIVE")
pytestmark = [pytest.mark.live, pytest.mark.skipif(not LIVE, reason="set AINTERVIEWER_LIVE=<provider or URL>")]

PROFILE = ("Name: Sam.\nBackend engineer, 4 years at a ticketing company. Owned the checkout service (Java, ~2k req/s "
           "peak); cut p99 latency from 900 ms to 250 ms. Led the review after a 40-minute outage.\n")
GOOD = ["I'm Sam, a backend engineer for four years at a ticketing company. I owned checkout and cut p99 latency "
        "from 900 milliseconds to 250 by moving inventory checks off the hot path.",
        "I profiled slow checkouts myself, found the synchronous inventory call, and replaced it with one batched "
        "reservation. Conversion went up about 3 percent.",
        "I wrote the migration that caused a 40-minute outage. I found the lock, rolled it back, and added a CI check "
        "for locking migrations; no repeat in 18 months.",
        "I want Northwind because payments at your scale is exactly the reliability problem I enjoy.",
        "I chose the queue because the email step didn't need to block the customer.",
        "How does the team decide between reliability work and new features?",
        "What does on-call look like for the team?"]
WEAK = ["Yeah, I've done backend stuff, worked on a few things with the team.",
        "We did a project once, it went fine, we all worked hard on it.",
        "I'm not sure, we just kind of did it together as a team.",
        "It was okay, nothing special happened really.",
        "I guess I like the company.",
        "No questions, thanks."]


def live_llm():
    tool = P.tool_config()["llm"]
    return LLM(llm_config(tool, types.SimpleNamespace(provider=LIVE, model=os.environ.get("AINTERVIEWER_LIVE_MODEL"))))


def live_process(tmp_path_factory):
    d = make_process(tmp_path_factory.mktemp("live"), profile_text=PROFILE, panel=[
        {"role": "Engineering Manager", "opens": True, "people": [{"name": "Alex", "voice": "am_michael"}]},
        {"role": "Senior Engineer", "technical": True, "people": [{"name": "Kate", "voice": "af_heart"}]},
        {"role": "Shadow", "observer": True, "people": [{"name": "Priya", "voice": "af_nicole"}]}],
        interview={"minutes": 14, "candidate_minutes": 3, "max_questions": 3, "intro_skip_chance": 0,
                   "max_candidate_questions": 2})
    cfg = P.load(d)
    cfg["llm"] = llm_config(cfg["llm"], types.SimpleNamespace(provider=LIVE, model=os.environ.get("AINTERVIEWER_LIVE_MODEL")))
    return cfg, P.bank(cfg)


def interview(cfg, bank, answers, llm):
    random.seed(7)
    io = ScriptedIO(answers=list(answers), default=answers[-1], seconds=75)
    iv = S.Interview(cfg, io, llm, bank, PROFILE, S.pick_panel(cfg))
    iv.clock = lambda: io.clock[0]
    iv.run()
    return iv


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    cfg, bank = live_process(tmp_path_factory)
    llm = live_llm()
    iv = interview(cfg, bank, GOOD, llm)
    spoken, report, scores = S.debrief(cfg, llm, iv.transcript, bank, PROFILE, "")
    return types.SimpleNamespace(cfg=cfg, bank=bank, iv=iv, spoken=spoken, report=report, scores=scores)


@pytest.fixture(scope="module")
def weak_session(tmp_path_factory):
    cfg, bank = live_process(tmp_path_factory)
    cfg["interview"]["max_questions"] = 2
    llm = live_llm()
    iv = interview(cfg, bank, WEAK, llm)
    return S.debrief(cfg, llm, iv.transcript, bank, PROFILE, "")


# --- helpers ---------------------------------------------------------------------------------------------

def interviewer_lines(s):
    return [(e["who"], e["text"]) for e in s.iv.transcript if e["who"] != "you"]


def follow_ups(s):
    """Interviewer lines that are not the opening, a main question as written, a hand-over or the closing."""
    wordings = [S._norm(w) for q in s.bank["questions"] for w in q["q"]] + [S._norm(w) for w in s.bank["intro"]]
    lines = interviewer_lines(s)[1:-1]
    return [t for _, t in lines if not any(w in S._norm(t) for w in wordings)
            and "questions do you have" not in t.lower()]


def report(found):
    return "\n  - " + "\n  - ".join(found)


# --- interviewer behaviour ------------------------------------------------------------------------------

def test_follow_ups_are_one_short_question(session):
    bad = [t for t in follow_ups(session) if len(t.split()) > 30 or t.count("?") > 1]
    assert not bad, "long or multi-part follow-ups:" + report(bad)


def test_no_evaluation_of_answers(session):
    praise = re.compile(r"\b(great|excellent|impressive|good answer|well done|nice|fantastic|amazing|perfect|"
                        r"sounds (?:good|efficient|great))\b", re.I)
    bad = [t for t in follow_ups(session) if praise.search(t)]
    assert not bad, "evaluations:" + report(bad)


def test_no_markdown_or_stage_directions(session):
    bad = [t for _, t in interviewer_lines(session) if re.search(r"[*#\[\]]|^\s*-\s|\((?:pause|smiles|laughs)", t, re.M)]
    assert not bad, "markdown or stage directions:" + report(bad)


def test_main_questions_asked_as_written(session):
    ids = [i for i in dict.fromkeys(e.get("q") for e in session.iv.transcript if e["who"] == "you") if i and i != "intro"]
    said = " ".join(S._norm(t) for _, t in interviewer_lines(session))
    missing = [i for i in ids if not any(S._norm(w) in said for q in session.bank["questions"] if q["id"] == i for w in q["q"])]
    assert not missing, f"main questions not heard word for word: {missing}"


def test_observer_never_speaks(session):
    assert "Priya" not in {w for w, _ in interviewer_lines(session)}


def test_no_invented_premise(session):
    """'You mentioned X' must refer to something the candidate actually said."""
    said = set(re.findall(r"[a-z]{4,}", " ".join(e["text"] for e in session.iv.transcript if e["who"] == "you").lower()))
    bad = []
    for t in follow_ups(session):
        for m in re.finditer(r"you (?:said|mentioned|described|talked about|proposed)\s+([^.?!,]{6,80})", t, re.I):
            w = set(re.findall(r"[a-z]{4,}", m.group(1).lower()))
            if w and len(w & said) / len(w) < 0.5:
                bad.append(t)
    assert not bad, "follow-ups built on things never said:" + report(bad)


def test_candidate_name_only_in_greeting_and_goodbye(session):
    uses = [t for _, t in interviewer_lines(session) if "Sam" in t]
    assert len(uses) <= 2, report(uses)


def test_closes_with_next_steps(session):
    last = interviewer_lines(session)[-1][1].lower()
    assert "recruiter" in last or "next step" in last, last


# --- debrief -----------------------------------------------------------------------------------------------

SECTIONS = ["Interviewer and system issues", "Overview", "Answer by answer", "Patterns across the interview",
            "Values", "Self-contradictions", "Story choice", "Your questions at the end", "Delivery", "Top 3 to practise"]


def test_debrief_has_every_section(session):
    heads = {S._norm(h) for h in re.findall(r"^#\s+(.+)$", session.report, re.M)}
    missing = [s for s in SECTIONS if S._norm(s) not in heads]
    assert not missing, f"missing sections: {missing}"


def test_debrief_verdict_and_scores(session):
    assert re.search(r"\b(strong hire|hire|lean no|no)\b", session.spoken, re.I), session.spoken
    known = {q["theme"] for q in session.bank["questions"]} | {"intro"}
    assert session.scores and set(session.scores) <= known
    assert all(isinstance(v, (int, float)) and 1 <= v <= 5 for v in session.scores.values()), session.scores


def test_model_answers_use_only_numbers_that_came_up(session):
    known = G._numbers(PROFILE + " ".join(e["text"] for e in session.iv.transcript))
    blocks = re.findall(r"Say it like this:?\*{0,2}\s*(.+?)(?:\n\s*[-*]\s*\*\*|\n#|\Z)", session.report, re.S)
    invented = sorted({n for b in blocks for n in G._numbers(b) - known})
    assert blocks, "no model answers found"
    assert not invented, f"numbers in model answers that never came up: {invented}"


def test_weak_candidate_is_not_rated_a_hire(weak_session):
    spoken = weak_session[0].lower()
    assert not re.search(r"\b(strong hire|(?<!no )hire)\b", spoken) or "lean no" in spoken or re.search(r"\bno\b", spoken), \
        f"lenient verdict for a weak candidate: {weak_session[0]}"


# --- drill ---------------------------------------------------------------------------------------------------

def test_drill_feedback(tmp_path_factory, isolated):
    cfg, bank = live_process(tmp_path_factory)
    io, log = ScriptedIO(answers=[GOOD[1], "stop"]), []
    S.run_drill(cfg, io, live_llm(), bank, PROFILE, log)
    coach = log[2]["text"]
    assert "score" in log[1], "no SCORE line"
    assert len(re.findall(r"[.!?](?:\s|$)", coach)) <= 5 and "SCORE" not in coach and "*" not in coach, coach


# --- init on varied descriptions ------------------------------------------------------------------------------

DESCRIPTIONS = {
    "complex": ('Final loop for SDE II at Amazon, Prime Video Playback (London). This interview is the Bar Raiser '
                'round: 60 minutes, one Bar Raiser plus a shadow who only listens. 2-3 Leadership Principles, two '
                'follow-up layers per story, STAR with data. Principles: Customer Obsession, Ownership, Dive Deep, '
                'Earn Trust, Deliver Results. People fail by saying "we", reusing stories, having no failure story. '
                'Reported on Blind: "Tell me about a time you disagreed with your manager", "Tell me about a time you '
                'missed a deadline". Last 5 minutes for my questions. About me: Ana, 4 years backend, owned checkout '
                '(~2k req/s), cut p99 from 900 ms to 250 ms.'),
    "sparse": "Interview next week at Spotify for a data engineer role, with the hiring manager.",
}


@pytest.mark.parametrize("kind", ["complex", "sparse", "cv-only"])
def test_init(kind, tmp_path, capsys):
    files, desc = [], DESCRIPTIONS.get(kind, "")
    if kind == "cv-only":
        cv = tmp_path / "cv.txt"
        cv.write_text("Jordan Lee\nData analyst, 5 years in e-commerce. Built a pricing experiment framework (12 tests "
                      "per quarter). Cut dashboard load time from 40 s to 5 s.\nApplying: Senior Data Analyst, Brightwave.")
        files = [str(cv)]
    dst = tmp_path / "proc"
    llm = live_llm()
    G.generate(dst, desc, files, llm, model=llm.cfg.get("debrief_model") or llm.cfg["model"])
    out = capsys.readouterr().out
    cfg = P.load(dst)
    bank = P.bank(cfg)
    problems = []
    if "Check these numbers" in out:
        problems.append("invented numbers: " + re.sub(r"\x1b\[[0-9;]*m", "", out.split("not in your sources:", 1)[1].splitlines()[0]).strip())
    text = desc + "".join(open(f).read() for f in files)
    wordings = {S._norm(w) for q in bank["questions"] for w in q["q"]}
    problems += [f"reported question reworded: {q}" for q in G._reported_questions(text) if S._norm(q) not in wordings]
    if kind == "complex":
        named = ["Customer Obsession", "Ownership", "Dive Deep", "Earn Trust", "Deliver Results"]
        problems += [f"value dropped: {v}" for v in named if v not in cfg["values"]]
        if not any(r["observer"] for r in cfg["panel"]):
            problems.append("the shadow is not an observer")
        if cfg["interview"]["minutes"] != 60:
            problems.append(f"length {cfg['interview']['minutes']} min, not 60")
    assert not problems, report(problems)
