"""interviewer.generate: building a process folder from the model's output. The guarantee under test: whatever
gets written loads and passes the dry run; everything else is a retry or a clean failure."""
import shutil
import subprocess
import zipfile

import pytest
import yaml

import interviewer.generate as G
import interviewer.process as P
from interviewer.llm import LLM
from tests.helpers import BASE_PROCESS, BASE_QUESTIONS, INIT_OUTPUT, PROFILE, FakeServer

SOURCES = ('## The candidate\'s description\nHiring Manager round at Northwind for a Backend Engineer, '
           '30 minutes, with the Payments team. Values: Ownership, Customer First. Reported: "Tell me about a '
           'time you missed a deadline" and "Why Northwind?". I\'m Sam, 4 years backend, cut checkout latency '
           'from 900 ms to 250 ms.')


def raw(process=None, questions=None, profile=PROFILE, summary="- done"):
    p = process if isinstance(process, str) else yaml.safe_dump(process or BASE_PROCESS, sort_keys=False)
    q = questions if isinstance(questions, str) else yaml.safe_dump(questions or BASE_QUESTIONS, sort_keys=False)
    return f"=== process.yaml ===\n{p}\n=== questions.yaml ===\n{q}\n=== profile.md ===\n{profile}\n=== summary ===\n{summary}"


# --- reading sources --------------------------------------------------------------------------------

class TestReadSource:
    def test_text(self, tmp_path):
        f = tmp_path / "ad.md"
        f.write_text("# Job\nBackend role")
        assert G.read_source(f) == "# Job\nBackend role"

    def test_html_drops_tags_and_scripts(self, tmp_path):
        f = tmp_path / "ad.html"
        f.write_text("<html><script>evil()</script><style>x{}</style><p>Backend <b>role</b></p></html>")
        out = G.read_source(f)
        assert "Backend" in out and "role" in out and "evil" not in out and "<" not in out

    def test_docx(self, tmp_path):
        f = tmp_path / "cv.docx"
        with zipfile.ZipFile(f, "w") as z:
            z.writestr("word/document.xml", "<w:document><w:p><w:r><w:t>Sam</w:t></w:r></w:p>"
                                            "<w:p><w:r><w:t>Backend engineer</w:t></w:r></w:p></w:document>")
        assert G.read_source(f).split() == ["Sam", "Backend", "engineer"]

    @pytest.mark.skipif(not shutil.which("pdftotext"), reason="pdftotext not installed")
    def test_pdf(self, tmp_path):
        text = "Sam backend engineer"
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
        objs = [b"<< /Type /Catalog /Pages 2 0 R >>", b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
                b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
        out, offs = b"%PDF-1.4\n", []
        for i, o in enumerate(objs, 1):
            offs.append(len(out))
            out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
        x = len(out)
        out += b"xref\n0 6\n0000000000 65535 f \n" + b"".join(b"%010d 00000 n \n" % o for o in offs)
        out += b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % x
        f = tmp_path / "cv.pdf"
        f.write_bytes(out)
        assert G.read_source(f).strip() == text

    def test_pdf_without_pdftotext(self, tmp_path, monkeypatch):
        (tmp_path / "cv.pdf").write_bytes(b"%PDF")
        monkeypatch.setattr(G.shutil, "which", lambda name: None)
        with pytest.raises(SystemExit, match="pdftotext"):
            G.read_source(tmp_path / "cv.pdf")

    def test_missing_file(self, tmp_path):
        with pytest.raises(SystemExit, match="not found"):
            G.read_source(tmp_path / "nope.txt")


# --- parsing and repairing YAML ----------------------------------------------------------------------

class TestSections:
    def test_split(self):
        sec = G._sections(INIT_OUTPUT)
        assert set(sec) == {"process.yaml", "questions.yaml", "profile.md", "summary"}

    def test_code_fences_and_prose(self):
        sec = G._sections("Sure!\n=== process.yaml ===\n```yaml\ncompany: A\n```\n=== summary ===\nok")
        assert sec["process.yaml"] == "company: A" and sec["summary"] == "ok"

    def test_case_insensitive_names(self):
        assert "process.yaml" in G._sections("=== Process.YAML ===\ncompany: A")


class TestLoadYaml:
    def test_valid(self):
        assert G._load_yaml("a: 1")[0] == {"a": 1}

    def test_quoting_mistake_is_requoted(self):
        d, fixed = G._load_yaml("probe: '\"Why didn't you stop?\" \"What then?\"'\nb: 2")
        assert d["probe"] == '"Why didn\'t you stop?" "What then?"' and d["b"] == 2

    def test_repeated_keys_are_merged(self):
        G._MergingLoader.merged = []
        d, _ = G._load_yaml("v:\n  brief: one\n  brief: two\n")
        assert d["v"]["brief"] == "one. two" and G._MergingLoader.merged == ["brief"]

    def test_tabs(self):
        assert G._load_yaml("panel:\n\t- role: A\n")[0] == {"panel": [{"role": "A"}]}

    def test_cut_off_tail_is_dropped(self):
        G._MergingLoader.merged = []
        d, _ = G._load_yaml("a: 1\nb: [1, 2\n")
        assert d == {"a": 1} and G._MergingLoader.merged[0].startswith("cut-off")

    def test_error_in_the_middle_is_not_silently_cut(self):
        with pytest.raises(yaml.YAMLError):
            G._load_yaml("a: [1\n" + "".join(f"k{i}: {i}\n" for i in range(20)))


class TestFix:
    def fix(self, proc=None, questions=None, sources=SOURCES):
        proc = yaml.safe_load(yaml.safe_dump(proc or BASE_PROCESS))
        questions = yaml.safe_load(yaml.safe_dump(questions or BASE_QUESTIONS))
        notes = G._fix(proc, questions, sources)
        return proc, questions, notes

    def test_voices_valid_and_unique(self):
        proc = dict(BASE_PROCESS, panel=[{"role": "A", "people": [{"name": "X", "voice": "nope"},
                                                                  {"name": "Y", "voice": "nope"}]}])
        p, _, notes = self.fix(proc)
        voices = [x["voice"] for x in p["panel"][0]["people"]]
        assert len(set(voices)) == 2 and all(v in G.VOICES for v in voices) and notes

    def test_shadow_becomes_observer(self):
        proc = dict(BASE_PROCESS, panel=BASE_PROCESS["panel"] + [{"role": "Shadow Observer", "people": [{"name": "P"}]}])
        p, _, _ = self.fix(proc)
        assert p["panel"][1]["observer"] is True

    def test_value_named_in_sources_is_added_not_dropped(self):
        q = {"questions": [{"id": "c", "theme": "customer", "value": "Customer First", "core": True, "q": ["X?"]}]}
        p, q2, notes = self.fix(questions=q)
        assert "Customer First" in p["values"] and q2["questions"][0]["value"] == "Customer First"

    def test_value_name_wrapped_across_lines_in_sources(self):
        q = {"questions": [{"id": "c", "theme": "x", "value": "Have Backbone; Disagree and Commit", "q": ["X?"]}]}
        p, _, _ = self.fix(questions=q, sources="Values: Have Backbone; Disagree and\nCommit, Deliver Results")
        assert "Have Backbone; Disagree and Commit" in p["values"]

    def test_unknown_value_is_dropped_from_question(self):
        q = {"questions": [{"id": "c", "theme": "x", "value": "Made Up", "q": ["X?"]}]}
        _, q2, notes = self.fix(questions=q)
        assert q2["questions"][0]["value"] == "" and any("Made Up" in n for n in notes)

    def test_value_given_as_definition(self):
        q = {"questions": [{"id": "c", "theme": "x", "value": {"brief": "b", "probe": "p"}, "q": ["X?"]}]}
        _, q2, _ = self.fix(questions=q)
        assert q2["questions"][0]["value"] == ""

    def test_reported_question_wording_is_restored(self):
        q = {"questions": [{"id": "dl", "theme": "pace", "core": True,
                            "q": ["Tell me about a time you missed a critical deadline and recovered."]}]}
        _, q2, notes = self.fix(questions=q)
        assert q2["questions"][0]["q"][0] == "Tell me about a time you missed a deadline"

    def test_reported_question_missing_is_added(self):
        q = {"questions": [{"id": "x", "theme": "x", "q": ["Something unrelated entirely?"]}]}
        _, q2, _ = self.fix(questions=q)
        assert ["Why Northwind?"] in [x["q"] for x in q2["questions"]]

    def test_shared_theme_is_split_by_value(self):
        q = {"questions": [{"id": f"q{i}", "theme": "leadership_principle", "core": True, "value": v, "q": [f"{v}?"]}
                           for i, v in enumerate(["Ownership", "Customer First", "Ownership"])]}
        _, q2, _ = self.fix(questions=q, sources="Values: Ownership, Customer First.")
        assert {x["theme"] for x in q2["questions"]} == {"ownership", "customer_first"}

    def test_exclude_keeps_only_default_ids(self):
        q = dict(BASE_QUESTIONS, exclude=["why_leave", "not_a_default", {"id": "x"}])
        _, q2, _ = self.fix(questions=q)
        assert q2["exclude"] == ["why_leave"]

    @pytest.mark.parametrize("questions", [{"questions": "not a list"}, {"questions": [None, "Plain?", 3]}, {}])
    def test_odd_question_lists(self, questions):
        _, q2, _ = self.fix(questions=questions)
        assert isinstance(q2["questions"], list)


class TestContentChecks:
    def test_reported_questions_found(self):
        assert G._reported_questions(SOURCES) == ["Tell me about a time you missed a deadline", "Why Northwind?"]

    def test_short_quotes_do_not_shift_pairing(self):
        src = 'They hate "we". Asked "Tell me about a failure" and "not enthusiastic enough".'
        assert G._reported_questions(src) == ["Tell me about a failure"]

    def test_invented_numbers(self):
        bad = G._invented_numbers({"team_facts": "We have 40 engineers.", "interview": {"minutes": 45}},
                                  "Cut latency from 900 ms to 25 ms.", SOURCES)
        assert bad == ["team_facts: 40", "profile.md: 25", "interview.minutes: 45"]

    def test_number_words_count(self):
        assert G._numbers("four years, two layers, 60 minutes") == {"4", "2", "60"}

    def test_unsupported_sentence_is_flagged(self):
        out = G._unsupported_sentences({}, "Name: Sam. Key stories: Team conflict resolution.", SOURCES)
        assert out and "conflict" in out[0]

    def test_faithful_profile_is_not_flagged(self):
        assert G._unsupported_sentences({}, PROFILE, SOURCES) == []

    def test_unsourced_value_descriptions(self):
        proc = {"values": {"Ownership": {"brief": "Leaders act on behalf of the entire company, beyond their own team."}}}
        assert G._unsourced_values(proc, SOURCES) == ["Ownership"]


class TestKeepComments:
    def test_comments_survive_repairs(self):
        text = "# TODO: ask the recruiter\ncompany: A\npanel:\n  - role: Shadow\n    people: [{name: P, voice: bad}]\nprofile: \"\"\n"
        proc = yaml.safe_load(text)
        proc["panel"][0]["people"][0]["voice"] = "af_heart"
        proc["panel"][0]["observer"] = True
        proc["profile"] = "profile.md"
        out = G._keep_comments(text, proc, [])
        assert "# TODO: ask the recruiter" in out and yaml.safe_load(out) == proc
        assert out.count("profile:") == 1

    def test_extra_todo_goes_above_its_key(self):
        text = "company: A\nvalues:\n  X: {brief: b}\n"
        out = G._keep_comments(text, yaml.safe_load(text), [("values", "# TODO: check X")])
        assert out.index("# TODO: check X") < out.index("values:")


# --- _build: whatever is returned has passed the dry run -------------------------------------------------

class TestBuild:
    def test_good_draft(self):
        r = G._build(raw(), SOURCES)
        assert set(r["files"]) == {"process.yaml", "questions.yaml", "profile.md"}
        assert "questions" in r["info"] and r["soft"] == []

    def test_profile_none(self):
        r = G._build(raw(profile="NONE"), SOURCES)
        assert "profile.md" not in r["files"] and yaml.safe_load(r["files"]["process.yaml"])["profile"] == ""

    def test_windows_line_endings(self):
        assert G._build(raw().replace("\n", "\r\n"), SOURCES)["files"]

    @pytest.mark.parametrize("bad, match", [("", "no '=== process.yaml"), ("company: A\n", "no '=== process.yaml"),
                                            (raw(process="- a\n- b"), "mapping"),
                                            (raw(process=dict(BASE_PROCESS, panel=None)), "panel"),
                                            (raw(process=dict(BASE_PROCESS, company="")), "company"),
                                            (raw(questions="1. Why?\n2. How?"), "mapping")])
    def test_rejected_drafts(self, bad, match):
        with pytest.raises(ValueError, match=match):
            G._build(bad, SOURCES)

    def test_invented_numbers_are_soft(self):
        r = G._build(raw(profile="Name: Sam.\nCut latency from 900 ms to 25 ms."), SOURCES)
        assert r["soft"] == ["profile.md: 25"]

    @pytest.mark.parametrize("cut", range(200, 1600, 200))
    def test_draft_cut_off_anywhere_in_questions(self, cut):
        full = raw(questions={"extends_default": True, "questions": [
            {"id": f"q{i}", "theme": f"t{i}", "core": True, "q": [f"Question number {i} about your work?"]}
            for i in range(30)]})
        qi = full.index("=== questions.yaml ===")
        draft = full[:qi + cut]
        try:
            r = G._build(draft, SOURCES)
        except ValueError:
            return                                        # a retry: fine
        P.bank(P.load(_write(r)))                          # written: must load

    def test_repairs_that_crash_fall_back_to_the_draft(self, monkeypatch):
        monkeypatch.setattr(G, "_fix", lambda *a: 1 / 0)
        r = G._build(raw(), SOURCES)
        assert any("repairs skipped" in f for f in r["fixes"])


def _write(result):
    import tempfile
    from pathlib import Path
    d = Path(tempfile.mkdtemp())
    for name, content in result["files"].items():
        (d / name).write_text(content)
    return d


# --- generate(): the whole flow against a stand-in server ------------------------------------------------

class TestGenerate:
    def llm(self, server):
        return LLM({"model": "auto", "debrief_model": "auto", "feedback_model": "auto", "max_tokens": 160,
                    "provider": server.url})

    def test_writes_a_working_folder(self, tmp_path, capsys):
        srv = FakeServer(reply=lambda b: INIT_OUTPUT)
        try:
            G.generate(tmp_path / "new", "Hiring Manager round at Northwind, Payments team. I'm Sam, 4 years, "
                       "cut checkout latency from 900 ms to 250 ms.", [], self.llm(srv), model="auto")
        finally:
            srv.close()
        d = tmp_path / "new"
        assert {f.name for f in d.iterdir()} == {"process.yaml", "questions.yaml", "profile.md", "sources.md"}
        assert "# TODO" in (d / "process.yaml").read_text()
        out = capsys.readouterr().out
        assert "stays on this machine" in out and "New process" in out
        P.bank(P.load(d))

    def test_retry_after_a_bad_draft(self, tmp_path):
        drafts = iter(["nonsense without sections", INIT_OUTPUT])
        srv = FakeServer(reply=lambda b: next(drafts))
        try:
            G.generate(tmp_path / "new", "Northwind, Payments. Sam, 4 years, 900 ms to 250 ms.", [], self.llm(srv), model="auto")
            assert "Checking those files found a problem" in srv.requests[-1]["messages"][-1]["content"]
            assert "never delete" in srv.requests[-1]["messages"][-1]["content"]
        finally:
            srv.close()
        assert (tmp_path / "new" / "process.yaml").exists()

    def test_two_bad_drafts_write_nothing(self, tmp_path):
        srv = FakeServer(reply=lambda b: "still nonsense")
        try:
            with pytest.raises(SystemExit, match="Could not build"):
                G.generate(tmp_path / "new", "x", [], self.llm(srv), model="auto")
        finally:
            srv.close()
        assert not (tmp_path / "new").exists()
        assert (tmp_path / "new-init-output.txt").read_text() == "still nonsense"

    def test_invented_number_gets_one_retry_and_keeps_the_better_draft(self, tmp_path):
        bad = INIT_OUTPUT.replace("250 ms", "25 ms")
        drafts = iter([bad, INIT_OUTPUT])
        srv = FakeServer(reply=lambda b: next(drafts))
        try:
            G.generate(tmp_path / "new", "Northwind, Payments. Sam, 4 years, 900 ms to 250 ms.", [], self.llm(srv), model="auto")
        finally:
            srv.close()
        assert "250 ms" in (tmp_path / "new" / "profile.md").read_text()

    def test_sources_too_long(self, tmp_path, server):
        with pytest.raises(SystemExit, match="characters"):
            G.generate(tmp_path / "new", "x" * (G.MAX_SOURCE_CHARS + 1), [], self.llm(server), model="auto")

    def test_cut_off_reply_is_continued(self, tmp_path):
        """A server cap cuts the first reply before the profile; the continuation completes it."""
        cut = INIT_OUTPUT.index("=== profile.md ===")
        replies = iter([INIT_OUTPUT[:cut], INIT_OUTPUT[cut:]])
        srv = FakeServer(reply=lambda b: next(replies))
        try:
            G.generate(tmp_path / "new", "Northwind, Payments. Sam, 4 years, 900 ms to 250 ms.", [], self.llm(srv), model="auto")
        finally:
            srv.close()
        assert (tmp_path / "new" / "profile.md").exists()


class TestLiveFindings:
    def test_unindented_continuation_lines(self):
        text = ("company: A\ncandidate_level: >-\nSenior analyst with 5 years.\nExpected to own projects.\n"
                "interviewer_rules: >-\nDo not ask about gaps.\npanel:\n  - role: HM\n    people: [Kim]\n")
        d, _ = G._load_yaml(text)
        assert d["candidate_level"] == "Senior analyst with 5 years. Expected to own projects."
        assert d["interviewer_rules"] == "Do not ask about gaps." and d["panel"][0]["role"] == "HM"

    def test_guessed_interview_settings_are_removed(self):
        proc = dict(BASE_PROCESS, interview={"minutes": 45, "max_followups": 2})
        r = G._build(raw(process=proc), SOURCES.replace("30 minutes", "two follow-up layers"))
        assert "minutes" not in (yaml.safe_load(r["files"]["process.yaml"]).get("interview") or {})
        assert yaml.safe_load(r["files"]["process.yaml"])["interview"]["max_followups"] == 2
        assert any("guessed setting (interview.minutes: 45)" in f for f in r["fixes"]) and r["soft"] == []
