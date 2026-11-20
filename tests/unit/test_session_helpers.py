"""interviewer.session: the pure helpers (tags, wording, question picking, weak spots, prompt pieces)."""
import json
import random
from collections import Counter

import pytest

import interviewer.session as S
from interviewer.metrics import answer_metrics, format_metrics


def tagged(chunks, **kw):
    t = S.Tagged(iter(chunks), **kw)
    text = "".join(t)                 # the tag is known only once the stream has been read
    return t.tag, text


# --- the [F]/[N]/[E] tag reader ---------------------------------------------------------------

class TestTagged:
    @pytest.mark.parametrize("tag", ["F", "N", "E"])
    def test_reads_and_strips_tag(self, tag):
        assert tagged([f"[{tag}] What was the result?"]) == (tag, "What was the result?")

    def test_tag_split_across_chunks(self):
        assert tagged(["[", "N", "] Nex", "t question?"]) == ("N", "Next question?")

    def test_default_tag_is_followup(self):
        assert tagged(["How did you measure it?"]) == ("F", "How did you measure it?")

    def test_note_before_tag_is_never_spoken(self):
        assert tagged(["The answer was vague. [F] What did you do yourself?"]) == ("F", "What did you do yourself?")

    @pytest.mark.parametrize("label", ["Kate: ", "Kate here. ", "Kate, Lead Engineer: "])
    def test_speaker_labels_are_removed(self, label):
        assert tagged([f"[F] {label}What happened next?"])[1] == "What happened next?"

    def test_leading_candidate_name_is_removed(self):
        assert tagged(["[F] Sam, what happened next?"], name="Sam")[1] == "What happened next?"

    def test_name_later_in_sentence_stays(self):
        assert tagged(["[E] Thanks, Sam. Goodbye."], name="Sam")[1] == "Thanks, Sam. Goodbye."

    @pytest.mark.parametrize("opener", ["That sounds efficient. ", "Great. ", "Great, thanks. ", "Interesting! ",
                                        "That's impressive. ", "Thanks for sharing that. ", "It must have been hard. "])
    def test_opening_evaluation_is_removed(self, opener):
        assert tagged([f"[F] {opener}What was the result?"])[1] == "What was the result?"

    @pytest.mark.parametrize("text", ["That's all from my side. What questions do you have?",
                                      "Thank you for your time today. Goodbye.",
                                      "Thanks for your time. The recruiter will be in touch."])
    def test_closings_are_not_evaluations(self, text):
        assert tagged([f"[E] {text}"])[1] == text

    def test_evaluation_alone_is_kept(self):
        # nothing would be left to say otherwise
        assert tagged(["[F] Great."])[1] == "Great."

    def test_replace_speaks_fixed_text_and_stops_the_model(self):
        closed = []

        def gen():
            try:
                yield "[N] So, moving along, could you maybe tell me about a time..."
                yield " more model text"
            finally:
                closed.append(True)
        t = S.Tagged(gen(), replace={"N": "Tell me about a mistake."})
        assert "".join(t) == "Tell me about a mistake." and t.tag == "N" and closed

    def test_replace_does_not_touch_other_tags(self):
        assert tagged(["[F] Why?"], replace={"N": "X"}) == ("F", "Why?")

    def test_long_untagged_text_is_released(self):
        text = "word " * 100
        assert tagged([text])[1].strip() == text.strip()


class TestAsksSame:
    @pytest.mark.parametrize("q, text", [
        ("Tell me about a project that didn't go well.", "Do you have a different example of a project that didn't go well?"),
        ("What's your biggest weakness?", "So, what would you say your biggest weakness is these days?"),
    ])
    def test_paraphrase_of_next_question(self, q, text):
        assert S.asks_same(q, text)

    @pytest.mark.parametrize("q, text", [
        ("How do you prioritise when everything seems urgent?", "How did you prioritise that?"),
        ("Tell me about a time you failed.", "What was the result?"),
        ("Why Northwind?", "Why?"),
    ])
    def test_ordinary_follow_ups_are_not_the_next_question(self, q, text):
        assert not S.asks_same(q, text)


class TestUnfinished:
    @pytest.mark.parametrize("text", ["So I built the, and", "We had, um", "It took about...", "I think that the"])
    def test_thinking_pauses(self, text):
        assert S.unfinished(text, 5)

    @pytest.mark.parametrize("text, secs", [("It took two weeks.", 5), ("And then", 40), ("", 3)])
    def test_finished_or_long(self, text, secs):
        assert not S.unfinished(text, secs)


def test_condense_keeps_start_and_end():
    text = " ".join(f"w{i}" for i in range(200))
    out = S._condense(text, 20)
    assert out.startswith("w0 ") and out.endswith("w199") and " ... " in out and len(out.split()) == 21
    assert S._condense("short answer", 20) == "short answer"


def test_wording_picks_one_of_the_wordings():
    random.seed(0)
    q = {"q": ["A?", "B?", "C?"]}
    assert {S.wording(q) for _ in range(50)} == {"A?", "B?", "C?"}
    assert S.wording({"q": "Only?"}) == "Only?"


@pytest.mark.parametrize("secs, want", [(0, "0:00"), (59, "0:59"), (61, "1:01"), (1800, "30:00"), (-5, "0:00")])
def test_fmt_clock(secs, want):
    assert S._fmt_clock(secs) == want


# --- weak spots and question picking -------------------------------------------------------------

class TestWeakSpots:
    def test_moving_average(self):
        w = {"themes": {}, "asked": {}}
        S.update_weak(w, "ownership", 2)
        assert w["themes"]["ownership"] == 2
        S.update_weak(w, "ownership", 4)
        assert w["themes"]["ownership"] == pytest.approx(2.8)

    def test_weak_and_rarely_asked_weigh_more(self):
        weak = {"themes": {"a": 1.0, "b": 5.0}, "asked": {"q2": 3}}
        assert S.weight({"id": "q1", "theme": "a"}, weak) > S.weight({"id": "q1", "theme": "b"}, weak)
        assert S.weight({"id": "q1", "theme": "a"}, weak) > S.weight({"id": "q2", "theme": "a"}, weak)

    def test_save_and_load(self, isolated):
        S.save_weak({"themes": {"x": 3}, "asked": {}})
        assert S.load_weak()["themes"] == {"x": 3}

    def test_corrupt_file_starts_fresh(self, isolated):
        S.WEAK_FILE.parent.mkdir(parents=True, exist_ok=True)
        S.WEAK_FILE.write_text("{not json")
        assert S.load_weak() == {"themes": {}, "asked": {}}

    def test_use_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "WEAK_FILE", S.WEAK_FILE)
        monkeypatch.setattr(S, "SESS_DIR", S.SESS_DIR)
        S.use_data_dir(tmp_path / "x")
        assert S.WEAK_FILE == tmp_path / "x" / "weak_spots.json" and S.SESS_DIR == tmp_path / "x" / "sessions"

    def test_env_overrides_data_dir(self, tmp_path, monkeypatch):
        monkeypatch.setattr(S, "WEAK_FILE", S.WEAK_FILE)
        monkeypatch.setattr(S, "SESS_DIR", S.SESS_DIR)
        monkeypatch.setenv("INTERVIEWER_DATA_DIR", str(tmp_path / "env"))
        S.use_data_dir(tmp_path / "x")
        assert S.SESS_DIR == tmp_path / "env" / "sessions"


def q(id, theme, core=True, own=False, value=""):
    return {"id": id, "theme": theme, "core": core, "own": own, "value": value, "q": [f"{id}?"]}


class TestPickQuestions:
    weak = {"themes": {}, "asked": {}}

    def test_one_core_question_per_theme(self):
        bank = [q("a1", "a"), q("a2", "a"), q("b1", "b"), q("c1", "c")]
        for seed in range(20):
            random.seed(seed)
            plan = S.pick_questions(bank, 10, self.weak, extra_chance=0)
            assert Counter(x["theme"] for x in plan) == {"a": 1, "b": 1, "c": 1}

    def test_one_motivation_question_early(self):
        bank = [q("m1", "motivation"), q("m2", "motivation"), q("a", "a"), q("b", "b"), q("c", "c"), q("d", "d")]
        for seed in range(20):
            random.seed(seed)
            plan = S.pick_questions(bank, 10, self.weak, extra_chance=0)
            pos = [i for i, x in enumerate(plan) if x["theme"] == "motivation"]
            assert len(pos) == 1 and pos[0] <= 3

    def test_process_questions_come_first(self):
        bank = [q("g1", "g1"), q("g2", "g2"), q("g3", "g3"), q("o1", "o1", own=True), q("o2", "o2", own=True)]
        for seed in range(20):
            random.seed(seed)
            plan = [x["id"] for x in S.pick_questions(bank, 10, self.weak, extra_chance=0)]
            assert set(plan[:2]) == {"o1", "o2"}

    def test_own_question_wins_within_a_theme(self):
        bank = [q("generic", "ownership"), q("mine", "ownership", own=True)]
        for seed in range(20):
            random.seed(seed)
            assert [x["id"] for x in S.pick_questions(bank, 10, self.weak, extra_chance=0)] == ["mine"]

    def test_own_motivation_question_wins(self):
        bank = [q("why_company", "motivation"), q("why_us", "motivation", own=True), q("a", "a")]
        for seed in range(20):
            random.seed(seed)
            assert "why_company" not in [x["id"] for x in S.pick_questions(bank, 10, self.weak, extra_chance=0)]

    def test_extra_question_sometimes(self):
        bank = [q("a", "a"), q("b", "b"), q("x", "x", core=False)]
        random.seed(1)
        assert "x" in [x["id"] for x in S.pick_questions(bank, 10, self.weak, extra_chance=1)]
        assert "x" not in [x["id"] for x in S.pick_questions(bank, 10, self.weak, extra_chance=0)]

    def test_bank_without_core_questions_still_has_a_plan(self):
        bank = [q("a", "a", core=False), q("b", "b", core=False)]
        assert S.pick_questions(bank, 10, self.weak, extra_chance=0)

    def test_cap(self):
        bank = [q(f"t{i}", f"t{i}") for i in range(12)]
        assert len(S.pick_questions(bank, 4, self.weak, extra_chance=0)) == 4

    def test_weak_theme_tends_to_come_first(self):
        bank = [q("weak", "weak"), q("ok", "ok")]
        weak = {"themes": {"weak": 1.0, "ok": 5.0}, "asked": {}}
        firsts = Counter()
        for seed in range(200):
            random.seed(seed)
            firsts[S.pick_questions(bank, 10, weak, extra_chance=0)[0]["id"]] += 1
        assert firsts["weak"] > firsts["ok"] * 2


# --- prompt pieces ---------------------------------------------------------------------------------

class TestPromptPieces:
    def test_values_text(self, cfg):
        assert S.values_text(cfg) == "- Ownership: own outcomes"
        assert S.values_text(cfg, "probe") == "- Ownership: What did you do yourself?"
        assert S.values_text({"values": {}}) == "" and S.values_text({"values": ["A", "B"]}) == "A, B"

    def test_value_field(self, cfg):
        assert S.value_field(cfg, "Ownership", "red_flags") == "blames others"
        assert S.value_field(cfg, "Missing", "probe") == "" and S.value_field(cfg, "", "probe") == ""

    def test_round_name(self):
        assert S.round_name({"company": "A", "round": "Team Fit", "round_description": "culture fit"}) == \
            "A Team Fit interview (culture fit)"
        assert S.round_name({"company": "A", "round": "HM", "round_description": ""}) == "A HM interview"

    def test_close_note(self, cfg):
        assert "say goodbye by name" in S.close_note(cfg, "Sam") and "recruiter will be in touch" in S.close_note(cfg, "Sam")
        assert "by name" not in S.close_note(cfg, "")

    def test_panel_desc(self, cfg):
        assert S.panel_desc(cfg) == "one interviewer, the Engineering Manager"
        cfg["panel"] = [{"role": "A"}, {"role": "B", "observer": True}]
        assert S.panel_desc(cfg) == "a panel of 2: A; B (observer)"

    def test_pick_panel(self, cfg):
        m = S.pick_panel(cfg)[0]
        assert m["name"] == "Alex" and m["voice"] == "am_michael" and m["opens"] and not m["observer"]

    def test_legacy_keys_still_work(self, cfg):
        cfg["panel"] = [{"role": "Head", "key": "head", "people": [{"name": "A", "voice": "af_heart"}]},
                        {"role": "Py", "key": "python", "people": [{"name": "B", "voice": "bf_emma"}]}]
        a, b = S.pick_panel(cfg)
        assert a["opens"] and b["technical"]

    def test_profile_text(self):
        assert S.profile_text("  Name: Sam.  ") == "Name: Sam."
        assert "know nothing" in S.profile_text("")

    def test_interviewer_system(self, cfg):
        sys = S.interviewer_system(cfg, S.pick_panel(cfg), "Name: Sam.")
        for must in ("Northwind Hiring Manager interview (behavioural)", "Backend Engineer", "Payments builds",
                     "Name: Sam.", "Mid-level engineer", "Northwind values:", "Ownership", "[F]", "[N]", "[E]"):
            assert must in sys, must
        assert "{" not in sys.replace("{0}", "")          # no unfilled placeholders

    def test_interviewer_system_without_values_or_facts(self, cfg):
        cfg["values"], cfg["process"]["team_facts"] = {}, ""
        sys = S.interviewer_system(cfg, S.pick_panel(cfg), "")
        assert "values:" not in sys and "answer plausibly" in sys and "know nothing" in sys

    def test_observer_is_marked_in_prompt(self, cfg):
        panel = [{"name": "A", "role": "Lead", "voice": "x"}, {"name": "B", "role": "Shadow", "voice": "y", "observer": True}]
        sys = S.interviewer_system(cfg, panel, "")
        assert "B, Shadow (only observes" in sys and "One person from" in sys and "with an observer" in sys

    def test_drill_system(self, cfg):
        sys = S.drill_system(cfg, "Name: Sam.")
        assert "Northwind Hiring Manager interview" in sys and "Ownership" in sys and "Name: Sam." in sys

    def test_level_text_default(self):
        assert S.level_text({"interview": {}}) == "experienced hire"


# --- metrics ------------------------------------------------------------------------------------

class TestMetrics:
    def test_counts(self):
        m = answer_metrics("Um, so I built it, and we, like, shipped it. Basically I did it.", 10)
        assert m["words"] == 14 and m["wpm"] == 84 and m["i"] == 2 and m["we"] == 1 and m["fillers"] >= 3

    def test_short_answers_have_no_wpm(self):
        assert answer_metrics("Yes.", 2)["wpm"] is None

    def test_format(self):
        assert format_metrics({"seconds": 30, "words": 60, "wpm": 120, "fillers": 2, "i": 3, "we": 1}) == \
            "30s, 60 words, 120 wpm, 2 fillers, I/we 3/1"
        assert "wpm" not in format_metrics({"seconds": 2, "words": 1, "wpm": None, "fillers": 0, "i": 0, "we": 0})


# --- saved sessions ----------------------------------------------------------------------------------

class TestSessionFiles:
    transcript = [
        {"who": "Alex", "text": "Tell me about yourself."},
        {"who": "you", "text": "I build payment systems.", "q": "intro",
         "metrics": answer_metrics("I build payment systems.", 6), "latency": {"llm_first_token": 0.5}},
        {"who": "Alex", "text": "Why Northwind?"},
        {"who": "you", "text": "Because of the scale.\n\nAnd the team.", "q": "why_northwind"},
    ]

    def test_round_trip(self, fake_llm):
        path = S.save_session("interview", self.transcript, "# Overview\nok", fake_llm)
        back = S.parse_session_md(path.read_text())
        assert [e["who"] for e in back] == ["Alex", "you", "Alex", "you"]
        assert back[1]["text"] == "I build payment systems." and back[1]["q"] == "intro"
        assert "metrics_str" in back[1] and back[3]["q"] == "why_northwind"

    def test_question_ids_are_invisible_in_markdown(self):
        md = S.transcript_md(self.transcript)
        assert "<!-- q=intro -->" in md

    def test_file_layout(self, fake_llm):
        text = S.save_session("drill", self.transcript, "", fake_llm).read_text()
        assert text.startswith("# Mock drill") and "## Transcript" in text and "## Tokens" in text
        assert "## Debrief" not in text

    def test_multi_paragraph_turns_survive(self, fake_llm):
        path = S.save_session("interview", self.transcript, "", fake_llm)
        back = S.parse_session_md(path.read_text())
        assert back[3]["text"] == "Because of the scale.\n\nAnd the team." and back[3]["q"] == "why_northwind"

    def test_old_files_without_ids(self):
        md = "## Transcript\n\n**Alex:** Hi?\n\n**You:** Hello.\n\n> _3s, 1 words, 0 fillers, I/we 0/0_\n\n## Tokens\n"
        back = S.parse_session_md(md)
        assert back == [{"who": "Alex", "text": "Hi?"},
                        {"who": "you", "text": "Hello.", "metrics_str": "3s, 1 words, 0 fillers, I/we 0/0"}]


# --- found by the live tests ------------------------------------------------------------------------

class TestLiveFindings:
    def test_later_tag_is_not_spoken_and_decides_the_action(self):
        tag, text = tagged(["[F] On our team we rotate on-call weekly.\n\n[E] Thanks, Sam. ", "Goodbye."])
        assert tag == "E" and "[E]" not in text and text.endswith("Thanks, Sam. Goodbye.")

    def test_later_tag_split_across_chunks(self):
        long = "[F] " + "We ship every week and review every change. " * 3
        tag, text = tagged([long, "\n[", "E] Bye."])
        assert tag == "E" and "[" not in text and text.endswith("Bye.")

    def test_short_reply_with_two_tags(self):
        tag, text = tagged(["[F] We ship weekly. [E] Goodbye."])
        assert tag == "E" and text == "We ship weekly. Goodbye."

    def test_brackets_that_are_not_tags_survive(self):
        assert tagged(["[F] " + "x " * 40 + "the [beta] flag?"])[1].endswith("the [beta] flag?")

    @pytest.mark.parametrize("line, want", [("## 7. Story Choice", "# Story choice"), ("**Delivery**", "# Delivery"),
                                            ("### Top 3 to Practise:", "# Top 3 to practise"),
                                            ("## 1. # Interviewer and system issues", "# Interviewer and system issues")])
    def test_section_headings_are_standardised(self, line, want):
        assert S._tidy_headings(f"{line}\ntext") == f"{want}\ntext"

    def test_prompt_notation_copied_literally(self):
        text = '7. "# Story choice": **Mixed**. Used one story twice.\n10. "# Top 3 to practise":\n1. Drill'
        assert S._tidy_headings(text) == "# Story choice\n\n**Mixed**. Used one story twice.\n# Top 3 to practise\n1. Drill"

    def test_numbered_plain_section_title(self):
        assert S._tidy_headings("9. Delivery: fast and clear") == "# Delivery\n\nfast and clear"

    def test_numbered_list_items_are_left_alone(self):
        assert S._tidy_headings("1. The Full Loop Drill: practise it") == "1. The Full Loop Drill: practise it"

    def test_question_headings_are_left_alone(self):
        assert S._tidy_headings("## 2. Tell me about delivery?") == "## 2. Tell me about delivery?"
