"""Drill mode and the end-of-interview debrief (both layouts), with a scripted model."""
import json

import pytest

import interviewer.session as S
from tests.helpers import FakeLLM, ScriptedIO, default_reply, run_interview


# --- drill -------------------------------------------------------------------------------------------

def drill(cfg, bank, answers, script=default_reply):
    io, llm, log = ScriptedIO(answers=answers), FakeLLM(script), []
    S.run_drill(cfg, io, llm, bank, "Name: Sam.", log)
    return io, llm, log


class TestDrill:
    def test_one_round_then_stop(self, cfg, bank):
        io, llm, log = drill(cfg, bank, ["My answer with 3 numbers.", "stop"])
        assert [e["who"] for e in log] == ["interviewer", "you", "coach"]
        assert log[1]["score"] == 4 and log[1]["q"]

    def test_score_line_is_never_spoken(self, cfg, bank):
        io, llm, log = drill(cfg, bank, ["Answer.", "stop"])
        coach = [t for w, t in io.said if w == "Coach" and "Again" not in t]
        assert coach and all("SCORE" not in t for t in coach)
        assert "SCORE" not in log[2]["text"]

    def test_score_split_across_chunks_is_not_spoken(self, cfg, bank):
        def script(label, system, messages):
            return ["Good. Add a number.\nSC", "ORE: 2"] if label.startswith("drill") else default_reply(label, system, messages)
        io, llm, log = drill(cfg, bank, ["Answer.", "stop"], script)
        assert all("SC" not in t for w, t in io.said if w == "Coach")
        assert log[1]["score"] == 2

    def test_missing_score_is_tolerated(self, cfg, bank):
        def script(label, system, messages):
            return "Good answer overall." if label.startswith("drill") else default_reply(label, system, messages)
        io, llm, log = drill(cfg, bank, ["Answer.", "stop"], script)
        assert "score" not in log[1] and S.load_weak()["themes"] == {}

    def test_weak_spots_are_saved(self, cfg, bank):
        drill(cfg, bank, ["Answer.", "stop"])
        w = S.load_weak()
        assert list(w["themes"].values()) == [4] and sum(w["asked"].values()) == 1

    def test_again_repeats_the_same_question(self, cfg, bank):
        io, llm, log = drill(cfg, bank, ["First try.", "again", "Second try.", "stop"])
        asked = [e["text"] for e in log if e["who"] == "interviewer"]
        assert len(asked) == 2 and asked[0] == asked[1]

    def test_next_picks_a_new_question(self, cfg, bank):
        io, llm, log = drill(cfg, bank, ["A.", "next", "B.", "stop"])
        ids = [e["q"] for e in log if e["who"] == "you"]
        assert len(ids) == 2

    def test_silence_is_waited_out(self, cfg, bank):
        io, llm, log = drill(cfg, bank, ["", "Answer.", "stop"])
        assert [e["who"] for e in log].count("you") == 1

    def test_command_listening_uses_a_short_pause(self, cfg, bank):
        io, llm, log = drill(cfg, bank, ["Answer.", "stop"])
        assert io.listen_calls[1]["silence_end"] == 1.2

    def test_feedback_uses_feedback_model_and_red_flags(self, cfg, bank):
        own = [q for q in bank["questions"] if q.get("value") == "Ownership"]
        bank = dict(bank, questions=own)
        io, llm, log = drill(cfg, bank, ["Answer.", "stop"])
        call = next(c for c in llm.calls if c["label"].startswith("drill"))
        assert call["model"] == cfg["llm"]["feedback_model"] and "blames others" in call["messages"][0]["content"]


# --- debrief -------------------------------------------------------------------------------------------

@pytest.fixture
def interview(cfg, bank):
    iv, io, llm = run_interview(cfg, bank, "Name: Sam.")
    return iv.transcript


class TestDebriefSingle:
    def test_parses_spoken_report_scores(self, cfg, bank, interview):
        llm = FakeLLM()
        spoken, report, scores = S.debrief(cfg, llm, interview, bank, "Name: Sam.", "my notes")
        assert spoken == "Hire. Lead with results." and report.startswith("# Interviewer and system issues")
        assert scores == {"intro": 4, "ownership": 3} or "invented_theme" not in scores
        assert "invented_theme" not in scores
        assert llm.labels() == ["debrief"]

    def test_prompt_contains_everything_needed(self, cfg, bank, interview):
        llm = FakeLLM()
        S.debrief(cfg, llm, interview, bank, "Name: Sam.", "my notes")
        c = llm.calls[0]
        user = c["messages"][0]["content"]
        assert "Name: Sam." in user and "my notes" in user and "CANDIDATE [q=intro" in user
        assert "INTERVIEWER (Alex)" in user and "SCORES:" in user and "# Values" in user
        assert "Northwind values" in c["system"] and "blames others" in c["system"]
        assert c["model"] == cfg["llm"]["debrief_model"]

    def test_without_values_rates_what_they_assess(self, cfg, bank, interview):
        cfg["values"] = {}
        llm = FakeLLM()
        S.debrief(cfg, llm, interview, bank, "", "")
        assert "# What they assess" in llm.calls[0]["messages"][0]["content"]

    @pytest.mark.parametrize("text", [
        "## SPOKEN:\nHire.\n\n## REPORT:\n# Overview\nx\n\n## SCORES:\n{\"intro\": 4}",
        "**SPOKEN:** Hire.\n\n**REPORT:**\n# Overview\nx\n\n**SCORES:** {\"intro\": 4}",
        "### Spoken\nHire.\n\n### Report\n# Overview\nx\n\nScores: {\"intro\": 4}",
        "SPOKEN:\nHire.\n# Overview\nx\nSCORES:\n{\"intro\": 4}",                   # REPORT label missing
    ])
    def test_label_variants(self, cfg, bank, interview, text):
        llm = FakeLLM(lambda *a: text)
        spoken, report, scores = S.debrief(cfg, llm, interview, bank, "", "")
        assert spoken == "Hire." and report.startswith("# Overview") and scores == {"intro": 4}

    def test_unparseable_output_is_kept_as_report(self, cfg, bank, interview):
        llm = FakeLLM(lambda *a: "Just some prose with no sections.")
        spoken, report, scores = S.debrief(cfg, llm, interview, bank, "", "")
        assert spoken == "" and "Just some prose" in report and scores == {}

    def test_bad_scores_json(self, cfg, bank, interview):
        llm = FakeLLM(lambda *a: "SPOKEN:\nHire.\nREPORT:\n# x\nSCORES:\n{intro: four}")
        assert S.debrief(cfg, llm, interview, bank, "", "")[2] == {}

    def test_headings_are_tidied(self, cfg, bank, interview):
        llm = FakeLLM(lambda *a: 'SPOKEN:\nHire.\nREPORT:\n## 1. # Interviewer and system issues\nNone\n# # Delivery\nok\n'
                                 '## 2. "# Overview"\nrow\n## 3. Why Northwind?\nfine\nSCORES:\n{}')
        report = S.debrief(cfg, llm, interview, bank, "", "")[1]
        assert "# Interviewer and system issues" in report.splitlines() and "# Delivery" in report.splitlines()
        assert "# Overview" in report.splitlines() and "## 3. Why Northwind?" in report.splitlines()


class TestDebriefSections:
    def test_local_debrief_model_uses_sections(self, cfg, bank, interview):
        llm = FakeLLM(local=True)
        spoken, report, scores = S.debrief(cfg, llm, interview, bank, "", "")
        mains = S._main_questions(interview, bank)
        labels = llm.labels()
        assert labels[0] == "debrief:1" and labels[-2:] == ["debrief:rest1", "debrief:rest2"]
        assert sum(l.startswith("debrief:q") for l in labels) == len(mains)
        assert spoken == "Hire." and scores == {"intro": 3}
        for heading in ("# Interviewer and system issues", "# Answer by answer", "# Patterns across the interview",
                        "# Top 3 to practise"):
            assert heading in report

    @pytest.mark.parametrize("mode, local, sections", [("single", True, False), ("sections", False, True),
                                                       ("auto", False, False), ("auto", True, True)])
    def test_mode_setting(self, cfg, bank, interview, mode, local, sections):
        cfg["llm"]["debrief_mode"] = mode
        llm = FakeLLM(local=local)
        S.debrief(cfg, llm, interview, bank, "", "")
        assert (llm.labels()[0] == "debrief:1") == sections

    def test_each_question_gets_only_its_part_of_the_transcript(self, cfg, bank, interview):
        llm = FakeLLM(local=True)
        S.debrief(cfg, llm, interview, bank, "", "")
        parts = [c["messages"][0]["content"] for c in llm.calls if c["label"].startswith("debrief:q")]
        mains = [q for q, _ in S._main_questions(interview, bank)]
        for i, part in enumerate(parts):
            section = part.split("This question's part of the transcript", 1)[1]
            later = [m for m in mains[i + 1:] if m not in mains[:i + 1]]
            assert all(m not in section for m in later)

    def test_missing_scores_get_their_own_call(self, cfg, bank, interview):
        def script(label, system, messages):
            if label == "debrief:rest2":
                return "# Top 3 to practise\n1. x"
            if label == "debrief:scores":
                return '{"intro": 2}'
            return default_reply(label, system, messages)
        llm = FakeLLM(script, local=True)
        assert S.debrief(cfg, llm, interview, bank, "", "")[2] == {"intro": 2}
        assert llm.labels()[-1] == "debrief:scores"

    def test_report_label_missing_in_first_part(self, cfg, bank, interview):
        def script(label, system, messages):
            if label == "debrief:1":
                return "SPOKEN:\nHire.\n\n# Interviewer and system issues\nNone"
            return default_reply(label, system, messages)
        spoken, report, _ = S.debrief(cfg, FakeLLM(script, local=True), interview, bank, "", "")
        assert spoken == "Hire." and report.startswith("# Interviewer and system issues")


class TestMainQuestions:
    def test_from_question_ids(self, interview, bank):
        mains = S._main_questions(interview, bank)
        ids = list(dict.fromkeys(e["q"] for e in interview if e.get("q")))
        assert len(mains) == len(ids)

    def test_from_bank_wordings_when_ids_are_missing(self, interview, bank):
        stripped = [{k: v for k, v in e.items() if k != "q"} for e in interview]
        with_ids = S._main_questions(interview, bank)
        without = S._main_questions(stripped, bank)
        assert [q for q, _ in without] == [q for q, _ in with_ids]

    @pytest.mark.parametrize("line, want", [
        ("Hi, I'm Alex. Walk me through your background, and your biggest result.", "Walk me through your background, and your biggest result."),
        ("Thanks. Why Northwind? I'm curious.", "Why Northwind?"),
        ("Tell me about yourself", "Tell me about yourself"),
    ])
    def test_question_of_a_line(self, line, want):
        assert S._question_of(line) == want


def test_debrief_and_scores_update_weak_spots_only_for_known_themes(cfg, bank, interview):
    spoken, report, scores = S.debrief(cfg, FakeLLM(), interview, bank, "", "")
    known = {q["theme"] for q in bank["questions"]} | {"intro"}
    assert set(scores) <= known


def test_numbers_invented_in_model_answers_are_flagged(cfg, bank, interview):
    reply = ('SPOKEN:\nHire.\nREPORT:\n# Answer by answer\n## 1. Q\n- **Say it like this:** "I cut latency from 900 ms '
             'to 250 ms across 17 services."\n- **Follow-ups:** none\nSCORES:\n{}')
    report = S.debrief(cfg, FakeLLM(lambda *a: reply), interview, bank, "Name: Sam. 900 ms to 250 ms.", "")[1]
    assert "**Check:**" in report and "17" in report.split("**Check:**")[1] and "900" not in report.split("**Check:**")[1]


def test_no_check_note_when_numbers_came_up(cfg, bank, interview):
    reply = 'SPOKEN:\nHire.\nREPORT:\n## 1. Q\n- **Say it like this:** "From 900 ms to 250 ms."\nSCORES:\n{}'
    assert "**Check:**" not in S.debrief(cfg, FakeLLM(lambda *a: reply), interview, bank, "900 ms to 250 ms", "")[1]


def test_prose_starting_with_a_label_word_does_not_cut_the_report(cfg, bank, interview):
    reply = ("SPOKEN:\nHire.\n\nREPORT:\n# Values\nScores across the values are uneven.\nReport cards aside, fine.\n"
             "Spoken delivery was clear.\n\n# Delivery\nok\n\n# Top 3 to practise\n1. x\n\nSCORES:\n{\"intro\": 3}")
    spoken, report, scores = S.debrief(cfg, FakeLLM(lambda *a: reply), interview, bank, "", "")
    assert spoken == "Hire." and "# Top 3 to practise" in report and scores == {"intro": 3}
    assert "Scores across the values" in report and "Spoken delivery" in report
