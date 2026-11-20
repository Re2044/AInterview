"""interviewer.process: finding a process folder, loading and normalizing it, the question bank, init."""
from pathlib import Path

import pytest
import yaml

import interviewer.process as P
from tests.helpers import BASE_PROCESS, BASE_QUESTIONS, make_process


# --- finding the process folder -------------------------------------------------------------------

class TestFindDir:
    def test_explicit_folder(self, proc_dir):
        assert P.find_dir(str(proc_dir)) == proc_dir.resolve()

    def test_env_var_when_no_argument(self, proc_dir, monkeypatch):
        monkeypatch.setenv("AINTERVIEWER_PROCESS", str(proc_dir))
        assert P.find_dir(None) == proc_dir.resolve()

    def test_argument_beats_env(self, proc_dir, tmp_path, monkeypatch):
        other = make_process(tmp_path / "other")
        monkeypatch.setenv("AINTERVIEWER_PROCESS", str(other))
        assert P.find_dir(str(proc_dir)) == proc_dir.resolve()

    def test_current_folder(self, proc_dir, monkeypatch):
        monkeypatch.chdir(proc_dir)
        assert P.find_dir(None) == proc_dir.resolve()

    def test_falls_back_to_builtin_sample(self, tmp_path, capsys):
        assert P.find_dir(None) == P.SAMPLES / P.DEFAULT_SAMPLE
        assert "built-in" in capsys.readouterr().out

    def test_sample_by_name(self):
        assert P.find_dir("behavioural") == P.SAMPLES / "behavioural"

    def test_unknown_name_is_an_error_naming_samples(self):
        with pytest.raises(SystemExit, match="behavioural"):
            P.find_dir("no-such-process")

    def test_folder_without_process_yaml(self, tmp_path):
        (tmp_path / "empty").mkdir()
        with pytest.raises(SystemExit, match="init"):
            P.find_dir(str(tmp_path / "empty"))

    def test_samples_list(self):
        assert "behavioural" in P.samples()


class TestDataDir:
    def test_own_process_keeps_data_in_its_folder(self, proc_dir):
        assert P.data_dir(proc_dir) == proc_dir

    def test_samples_write_to_user_data_folder_not_the_tool(self):
        d = P.data_dir(P.SAMPLES / "behavioural")
        assert d == P.DATA_ROOT / "behavioural"
        assert P.ROOT not in d.parents


# --- loading and normalizing ----------------------------------------------------------------------

def load_with(tmp_path, **overrides):
    return P.load(make_process(tmp_path / "p", **overrides))


class TestLoadBasics:
    def test_process_fields_and_tool_defaults(self, cfg):
        assert cfg["process"]["company"] == "Northwind"
        assert cfg["llm"]["model"]                       # tool config merged in
        assert cfg["interview"]["intro_skip_chance"] == 0   # process override
        assert cfg["interview"]["max_followups"] == 3       # tool default kept

    def test_minimal_process_gets_generic_defaults(self, tmp_path):
        d = tmp_path / "min"
        d.mkdir()
        (d / "process.yaml").write_text("company: Acme\n")
        cfg = P.load(d)
        p = cfg["process"]
        assert p["company"] == "Acme" and p["round"] == "behavioural" and p["team"] == "the team"
        assert p["interviewer_from"] == "the team"
        assert cfg["panel"][0]["people"] and cfg["values"] == {}

    def test_empty_process_yaml_still_loads(self, tmp_path):
        d = tmp_path / "e"
        d.mkdir()
        (d / "process.yaml").write_text("")
        assert P.load(d)["process"]["company"] == "the company"

    def test_folded_text_becomes_one_line(self, tmp_path):
        cfg = load_with(tmp_path, team_facts="line one\n  line two\n")
        assert cfg["process"]["team_facts"] == "line one line two"

    def test_vocabulary_reaches_speech_recognition(self, cfg):
        assert cfg["stt"]["vocabulary"] == ["Northwind", "Kafka"]

    def test_legacy_candidate_level_under_interview(self, tmp_path):
        proc = dict(BASE_PROCESS)
        proc.pop("candidate_level")
        proc["interview"] = {"candidate_level": "graduate"}
        cfg = P.load(make_process(tmp_path / "p", process=proc))
        assert cfg["process"]["candidate_level"] == "graduate"

    def test_process_can_override_llm_settings(self, tmp_path):
        cfg = load_with(tmp_path, llm={"model": "ollama/qwen3:8b"})
        assert cfg["llm"]["model"] == "ollama/qwen3:8b"
        assert cfg["llm"]["debrief_model"]               # the rest of the tool's llm section stays

    def test_invalid_yaml_is_a_clear_error(self, tmp_path):
        d = tmp_path / "bad"
        d.mkdir()
        (d / "process.yaml").write_text("company: [unclosed")
        with pytest.raises(P.ProcessError, match="not valid YAML"):
            P.load(d)

    def test_non_mapping_is_a_clear_error(self, tmp_path):
        d = tmp_path / "list"
        d.mkdir()
        (d / "process.yaml").write_text("- a\n- b\n")
        with pytest.raises(P.ProcessError, match="mapping"):
            P.load(d)


class TestNumbers:
    @pytest.mark.parametrize("raw, want", [(45, 45), ("45", 45), ("45 min", 45), ("about 45 minutes", 45),
                                           (45.6, 46), (0, 5), (-3, 5), (999, 240)])
    def test_minutes(self, tmp_path, raw, want):
        assert load_with(tmp_path, interview={"minutes": raw})["interview"]["minutes"] == want

    @pytest.mark.parametrize("raw", [True, None, "an hour", {}])
    def test_minutes_that_cannot_be_read(self, tmp_path, raw):
        with pytest.raises(P.ProcessError, match="interview.minutes"):
            load_with(tmp_path, interview={"minutes": raw})

    @pytest.mark.parametrize("raw, want", [(0.25, 0.25), (25, 0.25), ("25%", 0.25), ("0.4", 0.4), (1, 1.0), (-1, 0)])
    def test_chances(self, tmp_path, raw, want):
        assert load_with(tmp_path, interview={"curveball_chance": raw})["interview"]["curveball_chance"] == pytest.approx(want)

    def test_integer_settings_are_ints(self, tmp_path):
        ic = load_with(tmp_path, interview={"max_followups": "2", "max_questions": 3.0})["interview"]
        assert ic["max_followups"] == 2 and isinstance(ic["max_followups"], int)
        assert ic["max_questions"] == 3 and isinstance(ic["max_questions"], int)

    def test_candidate_minutes_must_leave_time_for_questions(self, tmp_path):
        ic = load_with(tmp_path, interview={"minutes": 20, "candidate_minutes": 30})["interview"]
        assert ic["candidate_minutes"] < ic["minutes"]

    @pytest.mark.parametrize("raw", ["60-120", [120, 60], [60], "x"])
    def test_bad_target_answer_seconds_fall_back(self, tmp_path, raw):
        assert load_with(tmp_path, interview={"target_answer_seconds": raw})["interview"]["target_answer_seconds"] == [60, 120]

    def test_changes_are_reported(self, tmp_path):
        cfg = load_with(tmp_path, interview={"minutes": "60 min"})
        assert any("interview.minutes" in n for n in cfg["load_notes"])

    @pytest.mark.parametrize("section", ["interview", "llm", "stt", "tts"])
    def test_section_that_is_not_a_mapping_is_ignored_with_a_note(self, tmp_path, section):
        cfg = load_with(tmp_path, **{section: "fast please"})
        assert any(f"`{section}:` ignored" in n for n in cfg["load_notes"])


class TestText:
    @pytest.mark.parametrize("raw, want", [(["a", "b"], "a b"), ({"length": "60"}, "length: 60"), (123, "123"),
                                           ("  spaced\n text ", "spaced text")])
    def test_text_fields(self, tmp_path, raw, want):
        assert load_with(tmp_path, team_facts=raw)["process"]["team_facts"] == want

    def test_empty_company_falls_back(self, tmp_path):
        assert load_with(tmp_path, company="")["process"]["company"] == "the company"

    def test_braces_are_kept_literally(self, tmp_path):
        cfg = load_with(tmp_path, team_facts="uses {weird} braces {0}")
        assert cfg["process"]["team_facts"] == "uses {weird} braces {0}"


class TestPanel:
    def roles(self, tmp_path, panel):
        return load_with(tmp_path, panel=panel)["panel"]

    def test_normal_panel(self, cfg):
        r = cfg["panel"][0]
        assert r["role"] == "Engineering Manager" and r["opens"] and not r["observer"]
        assert r["people"][0] == {"name": "Alex", "voice": "am_michael"}

    def test_single_role_mapping(self, tmp_path):
        assert self.roles(tmp_path, {"role": "CTO", "people": [{"name": "Kim"}]})[0]["role"] == "CTO"

    def test_role_as_string_gets_generic_people(self, tmp_path):
        r = self.roles(tmp_path, ["Bar Raiser"])[0]
        assert r["role"] == "Bar Raiser" and len(r["people"]) >= 1

    def test_people_as_names(self, tmp_path):
        r = self.roles(tmp_path, [{"role": "EM", "people": ["Elena Petrova", "Marcus"]}])[0]
        assert [p["name"] for p in r["people"]] == ["Elena", "Marcus"]     # first names only

    @pytest.mark.parametrize("panel", [[], None, 7, "", [None, 3]])
    def test_unusable_panel_uses_default_interviewer(self, tmp_path, panel):
        r = self.roles(tmp_path, panel)
        assert r and r[0]["people"] and "observer" in r[0]

    def test_voices_are_valid_and_unique(self, tmp_path):
        r = self.roles(tmp_path, [{"role": "A", "people": [{"name": "X", "voice": "nope"}, {"name": "Y", "voice": "nope"}]},
                                  {"role": "B", "people": [{"name": "Z", "voice": "af_heart"}, {"name": "W"}]}])
        voices = [p["voice"] for role in r for p in role["people"]]
        assert all(v in P.VOICES for v in voices) and len(set(voices)) == len(voices)

    def test_role_without_name_gets_one(self, tmp_path):
        assert self.roles(tmp_path, [{"people": [{"name": "Kim"}]}])[0]["role"] == "Interviewer 1"

    @pytest.mark.parametrize("raw, want", [("false", False), ("no", False), ("yes", True), ("true", True), (1, True)])
    def test_flags_written_as_text(self, tmp_path, raw, want):
        r = self.roles(tmp_path, [{"role": "A", "people": ["Kim"], "technical": raw}, {"role": "B", "people": ["Lee"]}])
        assert r[0]["technical"] is want

    def test_all_observers_leaves_one_speaker(self, tmp_path):
        r = self.roles(tmp_path, [{"role": "A", "people": ["Kim"], "observer": True}])
        assert not r[0]["observer"]


class TestValues:
    def values(self, tmp_path, values):
        return load_with(tmp_path, values=values)["values"]

    def test_mapping(self, cfg):
        assert cfg["values"]["Ownership"]["probe"] == "What did you do yourself?"

    def test_list_of_names(self, tmp_path):
        assert self.values(tmp_path, ["Ownership", "Dive Deep"]) == {"Ownership": {}, "Dive Deep": {}}

    def test_value_as_text_is_its_brief(self, tmp_path):
        assert self.values(tmp_path, {"Ownership": "own it"}) == {"Ownership": {"brief": "own it"}}

    def test_nested_lists_become_text(self, tmp_path):
        assert self.values(tmp_path, {"O": {"brief": ["a", "b"], "probe": None}}) == {"O": {"brief": "a b"}}

    def test_string_is_ignored_with_note(self, tmp_path):
        cfg = load_with(tmp_path, values="be good")
        assert cfg["values"] == {} and any("values ignored" in n for n in cfg["load_notes"])


@pytest.mark.parametrize("raw, want", [("Amazon, AWS; ECS\nLambda", ["Amazon", "AWS", "ECS", "Lambda"]),
                                       (["A", None, 3], ["A", "3"]), (42, []), (None, [])])
def test_vocabulary_shapes(tmp_path, raw, want):
    assert load_with(tmp_path, vocabulary=raw)["stt"]["vocabulary"] == want


# --- profile and notes -------------------------------------------------------------------------

class TestProfile:
    def test_profile_is_read(self, cfg):
        assert P.profile(cfg).startswith("Name: Sam.")

    def test_no_profile_configured_is_fine(self, tmp_path):
        cfg = load_with(tmp_path, profile="")
        assert P.profile(cfg) == ""

    def test_missing_profile_file_warns_and_continues(self, tmp_path, capsys):
        cfg = load_with(tmp_path, profile="nope.md")
        assert P.profile(cfg) == "" and "No profile" in capsys.readouterr().out

    def test_notes(self, tmp_path):
        d = make_process(tmp_path / "p", notes="notes.md")
        (d / "notes.md").write_text("My stories")
        assert P.notes(P.load(d)) == "My stories"


# --- question bank -----------------------------------------------------------------------------

def bank_with(tmp_path, questions_doc, **overrides):
    cfg = P.load(make_process(tmp_path / "p", questions_doc=questions_doc, **overrides))
    return P.bank(cfg), cfg


class TestBank:
    def test_extends_default_adds_own_questions(self, bank):
        ids = {q["id"] for q in bank["questions"]}
        assert {"why_northwind", "own_it", "achievement"} <= ids

    def test_own_questions_are_marked(self, bank):
        own = {q["id"] for q in bank["questions"] if q.get("own")}
        assert own == {"why_northwind", "own_it"}

    def test_replace_mode_uses_only_own_questions(self, tmp_path):
        b, _ = bank_with(tmp_path, {"questions": BASE_QUESTIONS["questions"]})
        assert {q["id"] for q in b["questions"]} == {"why_northwind", "own_it"}

    def test_same_id_replaces_default(self, tmp_path):
        b, _ = bank_with(tmp_path, {"extends_default": True,
                                    "questions": [{"id": "achievement", "theme": "achievement", "q": ["Mine?"]}]})
        ach = [q for q in b["questions"] if q["id"] == "achievement"]
        assert len(ach) == 1 and ach[0]["q"] == ["Mine?"]

    @pytest.mark.parametrize("exclude", [["why_leave"], "why_leave"])
    def test_exclude(self, tmp_path, exclude):
        b, _ = bank_with(tmp_path, dict(BASE_QUESTIONS, exclude=exclude))
        assert "why_leave" not in {q["id"] for q in b["questions"]}

    def test_no_questions_key_uses_default_bank(self, tmp_path):
        b, _ = bank_with(tmp_path, BASE_QUESTIONS, questions="")
        assert {q["id"] for q in b["questions"]} == {q["id"] for q in yaml.safe_load(P.DEFAULT_BANK.read_text())["questions"]}

    def test_placeholders_are_filled(self, tmp_path):
        b, _ = bank_with(tmp_path, {"questions": [{"id": "w", "theme": "m", "q": ["Why {company}, in {team}, as {role}?"]}]})
        assert b["questions"][0]["q"] == ["Why Northwind, in the Payments team, as Backend Engineer?"]

    def test_default_bank_has_no_unfilled_placeholders(self, bank):
        assert not [w for q in bank["questions"] for w in q["q"] if "{company}" in w or "{team}" in w or "{role}" in w]

    def test_every_question_is_normalized(self, bank):
        for q in bank["questions"]:
            assert q["id"] and q["theme"] and isinstance(q["value"], str) and isinstance(q["core"], bool)
            assert q["q"] and all(isinstance(w, str) and w for w in q["q"])

    def test_wording_shapes(self, tmp_path):
        b, _ = bank_with(tmp_path, {"questions": [
            {"id": "a", "theme": "t", "q": "Single string?"},
            {"id": "b", "theme": "t", "q": [None, 42, "", "Real?"]},
            "Bare string question?",
            {"id": "c", "theme": "t"},                       # no wording: dropped
            None]})
        got = {q["id"]: q["q"] for q in b["questions"]}
        assert got["a"] == ["Single string?"] and got["b"] == ["42", "Real?"]
        assert ["Bare string question?"] in got.values() and "c" not in got

    def test_ids_and_themes_are_made_safe_and_unique(self, tmp_path):
        b, _ = bank_with(tmp_path, {"questions": [{"id": "Same Id", "q": ["A?"]}, {"id": "same id", "q": ["B?"]},
                                                  {"q": ["C?"], "theme": "Leadership Principle"}]})
        ids = [q["id"] for q in b["questions"]]
        assert len(set(ids)) == 3 and ids[0] == "same_id"
        assert b["questions"][2]["theme"] == "leadership_principle" and b["questions"][0]["theme"] == "general"

    def test_core_written_as_text(self, tmp_path):
        b, _ = bank_with(tmp_path, {"questions": [{"id": "a", "q": ["A?"], "core": "yes"},
                                                  {"id": "b", "q": ["B?"], "core": "no"}]})
        assert [q["core"] for q in b["questions"]] == [True, False]

    def test_unknown_value_is_cleared_with_note(self, tmp_path):
        b, cfg = bank_with(tmp_path, {"questions": [{"id": "a", "q": ["A?"], "value": "Nope"}]})
        assert b["questions"][0]["value"] == "" and any("Nope" in n for n in cfg["load_notes"])

    def test_bare_list_file(self, tmp_path):
        b, _ = bank_with(tmp_path, "- Why us?\n- Tell me about a failure.\n")
        assert [q["q"] for q in b["questions"]] == [["Why us?"], ["Tell me about a failure."]]

    @pytest.mark.parametrize("intro, n", [("Tell me about yourself.", 1), ([], 4), (None, 4)])
    def test_intro_shapes(self, tmp_path, intro, n):
        b, _ = bank_with(tmp_path, dict(BASE_QUESTIONS, intro=intro))
        assert len(b["intro"]) == n and all(isinstance(w, str) and w for w in b["intro"])

    @pytest.mark.parametrize("text, match", [("questions: [\n", "not valid YAML"), ("", "empty"),
                                             ("questions: {a: 1}\n", "must be a list"), ("42\n", "mapping"),
                                             ("extends_default: false\nquestions: [{id: x}]\n", "no usable")])
    def test_bank_errors(self, tmp_path, text, match):
        with pytest.raises(P.ProcessError, match=match):
            bank_with(tmp_path, text)

    def test_missing_bank_file(self, tmp_path):
        cfg = load_with(tmp_path, questions="nope.yaml")
        with pytest.raises(P.ProcessError, match="not found"):
            P.bank(cfg)


# --- the shipped files load ---------------------------------------------------------------------

@pytest.mark.parametrize("folder", [P.SAMPLES / "behavioural", P.TEMPLATE])
def test_shipped_processes_load(folder):
    cfg = P.load(folder)
    b = P.bank(cfg)
    assert b["questions"] and cfg["panel"] and not cfg["load_notes"], cfg["load_notes"]


def test_shipped_default_bank_is_clean():
    raw = yaml.safe_load(P.DEFAULT_BANK.read_text())
    ids = [q["id"] for q in raw["questions"]]
    assert len(ids) == len(set(ids))
    assert any(q.get("core") and q["theme"] == "motivation" for q in raw["questions"])
    assert all(q.get("q") for q in raw["questions"])


# --- init from the template -----------------------------------------------------------------------

class TestInitTemplate:
    def test_copies_template(self, tmp_path):
        P.init(str(tmp_path / "new"))
        assert {f.name for f in (tmp_path / "new").iterdir()} >= {"process.yaml", "profile.md", "questions.yaml"}
        P.bank(P.load(tmp_path / "new"))

    def test_refuses_to_overwrite(self, proc_dir):
        before = (proc_dir / "process.yaml").read_text()
        with pytest.raises(SystemExit, match="not overwriting"):
            P.init(str(proc_dir))
        assert (proc_dir / "process.yaml").read_text() == before

    def test_keeps_existing_files(self, tmp_path):
        d = tmp_path / "keep"
        d.mkdir()
        (d / "profile.md").write_text("mine")
        P.init(str(d))
        assert (d / "profile.md").read_text() == "mine"
