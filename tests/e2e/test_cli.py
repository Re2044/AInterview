"""End to end: the real `ainterviewer` command, run as a subprocess against a stand-in OpenAI-compatible
server (--provider <url>), with typed answers on stdin. Data folders are redirected to a temp dir."""
import json
import os
import subprocess

import pytest
import yaml

import interviewer.process as P
from tests.helpers import CLI, FakeServer, make_process, scripted_server_reply

pytestmark = pytest.mark.e2e


@pytest.fixture
def srv():
    s = FakeServer(reply=scripted_server_reply)
    yield s
    s.close()


@pytest.fixture
def env(tmp_path):
    e = {k: v for k, v in os.environ.items()
         if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AINTERVIEWER_PROCESS", "INTERVIEWER_DATA_DIR")}
    e["XDG_DATA_HOME"] = str(tmp_path / "xdg")
    e["PYTHONUNBUFFERED"] = "1"
    return e


def run(env, *args, stdin="", cwd=None, timeout=120):
    p = subprocess.run([str(CLI), *map(str, args)], input=stdin, capture_output=True, text=True, env=env,
                       cwd=cwd, timeout=timeout)
    out = p.stdout + p.stderr
    assert "Traceback" not in out, out
    return p.returncode, out


ANSWERS = "\n".join([
    "I'm Sam, a backend engineer; I cut checkout latency from 900 ms to 250 ms.",
    "I profiled it myself and moved inventory checks off the hot path.",
    "Because Northwind's payments scale is the problem I want.",
    "Conversion went up 3 percent.",
    "I owned the migration end to end and wrote the runbook.",
    "It took six weeks and nobody got paged.",
    "How do you measure success?",
    "What does on-call look like?",
] + ["More detail on that."] * 30) + "\n"


# --- no model needed -----------------------------------------------------------------------------------------

def test_help(env):
    code, out = run(env, "--help")
    assert code == 0
    for word in ("interview", "drill", "rescore", "init", "show", "check", "--provider", "--process", "vllm"):
        assert word in out


def test_show(env, tmp_path):
    d = make_process(tmp_path / "p")
    code, out = run(env, "show", "-p", d)
    assert code == 0 and "Process: Northwind" in out and "You play the interviewer" in out and "Name: Sam." in out


@pytest.mark.parametrize("target", ["behavioural", str(P.TEMPLATE)])
def test_check_shipped(env, target):
    code, out = run(env, "check", "-p", target)
    assert code == 0 and "dry run ok" in out


def test_check_reports_what_it_reads_differently(env, tmp_path):
    d = make_process(tmp_path / "p", interview={"minutes": "45 min"}, vocabulary="A, B")
    code, out = run(env, "check", "-p", d)
    assert code == 0 and "Read as:" in out and "'45 min' read as 45" in out


def test_check_rejects_a_broken_process_cleanly(env, tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "process.yaml").write_text("company: [unclosed")
    code, out = run(env, "check", "-p", d)
    assert code != 0 and "Process problem" in out and "not valid YAML" in out


def test_current_folder_is_the_process(env, tmp_path):
    d = make_process(tmp_path / "p")
    code, out = run(env, "show", cwd=d)
    assert code == 0 and "Northwind" in out


def test_falls_back_to_the_sample(env, tmp_path):
    code, out = run(env, "show", cwd=tmp_path)
    assert code == 0 and "built-in 'behavioural' sample" in out and "Northwind" in out


def test_role_and_level_overrides(env):
    code, out = run(env, "show", "-p", "behavioural", "--role", "Data Analyst", "--level", "graduate")
    assert "applying for: Data Analyst" in out and "Role level: graduate" in out


def test_unknown_process_name(env):
    code, out = run(env, "show", "-p", "no-such-thing")
    assert code != 0 and "no built-in sample" in out


# --- init ------------------------------------------------------------------------------------------------------

def test_init_template(env, tmp_path):
    code, out = run(env, "init", tmp_path / "new", "--template")
    assert code == 0 and (tmp_path / "new" / "process.yaml").exists()
    code, out = run(env, "init", tmp_path / "new", "--template")
    assert code != 0 and "not overwriting" in out


def test_init_from_description(env, tmp_path, srv):
    desc = "Hiring Manager round at Northwind, Payments team. I'm Sam, 4 years, cut latency from 900 ms to 250 ms."
    code, out = run(env, "init", tmp_path / "new", "-d", desc, "--provider", srv.url)
    assert code == 0 and "New process" in out and "stays on this machine" in out, out
    files = {f.name for f in (tmp_path / "new").iterdir()}
    assert files == {"process.yaml", "questions.yaml", "profile.md", "sources.md"}
    assert desc in (tmp_path / "new" / "sources.md").read_text()
    code, out = run(env, "check", "-p", tmp_path / "new")
    assert code == 0 and "dry run ok" in out


def test_init_reads_files_and_stdin(env, tmp_path, srv):
    ad = tmp_path / "ad.html"
    ad.write_text("<p>Backend Engineer at Northwind, Payments team, 4 years, 900 ms to 250 ms</p>")
    code, out = run(env, "init", tmp_path / "new", "-f", ad, "--provider", srv.url, stdin="")
    assert code == 0 and "your files stay on this machine" in out
    sent = srv.requests[0]["messages"][-1]["content"]          # the first request carries the sources
    assert "## File: ad.html" in sent and "<p>" not in sent
    code, out = run(env, "init", tmp_path / "new2", "--provider", srv.url,
                    stdin="Northwind HM round, Payments. Sam, 4 years, 900 ms to 250 ms.")
    assert code == 0 and (tmp_path / "new2" / "process.yaml").exists()


def test_init_with_nothing_described_copies_template(env, tmp_path):
    code, out = run(env, "init", tmp_path / "new", stdin="")
    assert code == 0 and "copying the template" in out and (tmp_path / "new" / "process.yaml").exists()


def test_init_failure_writes_nothing(env, tmp_path):
    s = FakeServer(reply=lambda body: "I can't help with that.")
    try:
        code, out = run(env, "init", tmp_path / "new", "-d", "anything", "--provider", s.url)
    finally:
        s.close()
    assert code != 0 and "Could not build" in out and not (tmp_path / "new").exists()


# --- interview, drill, rescore ------------------------------------------------------------------------------------

def session_files(folder):
    return sorted((folder / "sessions").glob("*.md")) if (folder / "sessions").exists() else []


def test_typed_interview_end_to_end(env, tmp_path, srv):
    d = make_process(tmp_path / "p")
    code, out = run(env, "interview", "-p", d, "--text", "--mute", "--questions", "2", "--provider", srv.url,
                    stdin=ANSWERS)
    assert code == 0 and "Saved:" in out, out
    files = session_files(d)
    assert len(files) == 1
    text = files[0].read_text()
    assert "## Debrief" in text and "**Verdict:**" in text and "## Transcript" in text and "## Tokens" in text
    assert "# Answer by answer" in text and "<!-- q=" in text
    assert "[F]" not in text.split("## Transcript")[1] and "<think>" not in text
    weak = json.loads((d / "weak_spots.json").read_text())
    assert weak["themes"] and weak["asked"]
    assert "the debrief was written by a local model" in out


def test_interview_on_the_sample_keeps_data_out_of_the_tool(env, tmp_path, srv):
    code, out = run(env, "interview", "--text", "--mute", "--questions", "1", "--provider", srv.url,
                    stdin=ANSWERS, cwd=tmp_path)
    assert code == 0
    assert session_files(tmp_path / "xdg" / "ainterviewer" / "behavioural")
    assert not (P.SAMPLES / "behavioural" / "sessions").exists()


def test_interview_ended_before_any_answer(env, tmp_path, srv):
    d = make_process(tmp_path / "p")
    code, out = run(env, "interview", "-p", d, "--text", "--mute", "--provider", srv.url, stdin="")
    assert code == 0 and "Preparing feedback" not in out


def test_no_debrief_flag(env, tmp_path, srv):
    d = make_process(tmp_path / "p")
    code, out = run(env, "interview", "-p", d, "--text", "--mute", "--questions", "1", "--no-debrief",
                    "--provider", srv.url, stdin=ANSWERS)
    assert code == 0 and "## Debrief" not in session_files(d)[0].read_text()


def test_drill(env, tmp_path, srv):
    d = make_process(tmp_path / "p")
    code, out = run(env, "drill", "-p", d, "--text", "--mute", "--provider", srv.url,
                    stdin="My answer with a number: 40 percent.\nnext\nSecond answer.\nstop\n")
    assert code == 0 and "Coach:" in out and "SCORE" not in out and "<think>" not in out
    weak = json.loads((d / "weak_spots.json").read_text())
    assert sum(weak["asked"].values()) == 2 and weak["themes"]


def test_rescore(env, tmp_path, srv):
    d = make_process(tmp_path / "p")
    run(env, "interview", "-p", d, "--text", "--mute", "--questions", "1", "--no-debrief", "--provider", srv.url,
        stdin=ANSWERS)
    session = session_files(d)[0]
    before, weak_before = session.read_text(), (d / "weak_spots.json").read_text()
    code, out = run(env, "rescore", session, "--provider", srv.url)
    assert code == 0, out
    rescored = session.with_name(session.stem + "-rescored.md")
    assert rescored.exists() and "## Debrief" in rescored.read_text()
    assert session.read_text() == before and (d / "weak_spots.json").read_text() == weak_before


def test_rescore_finds_its_process_from_the_session_path(env, tmp_path, srv):
    d = make_process(tmp_path / "p")
    run(env, "interview", "-p", d, "--text", "--mute", "--questions", "1", "--no-debrief", "--provider", srv.url,
        stdin=ANSWERS)
    code, out = run(env, "rescore", session_files(d)[0], "--provider", srv.url, cwd=tmp_path)
    assert code == 0 and "Process: Northwind" in out


# --- provider errors are messages, not tracebacks ------------------------------------------------------------------

@pytest.mark.parametrize("provider, message", [("nope", "Unknown LLM provider"), ("lmstudio", "Can't reach lmstudio"),
                                               ("groq", "GROQ_API_KEY")])
def test_provider_errors(env, tmp_path, provider, message):
    d = make_process(tmp_path / "p")
    env = {k: v for k, v in env.items() if k != "GROQ_API_KEY"}
    code, out = run(env, "interview", "-p", d, "--text", "--mute", "--provider", provider, "--model", "m",
                    stdin=ANSWERS)
    assert code != 0 and message in out


def test_server_errors_mid_interview_are_spoken_not_fatal(env, tmp_path):
    calls = {"n": 0}

    def flaky(body):
        calls["n"] += 1
        return (503, {"error": {"message": "busy"}}) if calls["n"] == 3 else scripted_server_reply(body)
    s = FakeServer(reply=flaky)
    try:
        d = make_process(tmp_path / "p")
        code, out = run(env, "interview", "-p", d, "--text", "--mute", "--questions", "1", "--no-debrief",
                        "--provider", s.url, stdin=ANSWERS)
    finally:
        s.close()
    assert code == 0 and "lost my notes" in out
