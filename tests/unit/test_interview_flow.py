"""Interview.run end to end on a fake clock with a scripted model and candidate."""
import random

import pytest

import interviewer.process as P
import interviewer.session as S
from tests.helpers import FakeLLM, ScriptedIO, default_reply, make_process, run_interview


def norm(t):
    return S._norm(t)


def bank_wordings(bank):
    return {norm(w): q["id"] for q in bank["questions"] for w in q["q"]}


def asked_main_questions(io, bank):
    """Bank questions the candidate heard, in order (a main question is always asked exactly as written)."""
    words = bank_wordings(bank)
    out = []
    for _, text in io.said:
        for w, qid in words.items():
            if w and w in norm(text):
                out.append(qid)
                break
    return out


# --- opening ----------------------------------------------------------------------------------------

class TestOpening:
    def test_opening_then_first_answer_is_the_intro(self, cfg, bank):
        iv, io, llm = run_interview(cfg, bank, "Name: Sam.")
        assert llm.labels()[0] == "intro" and io.said[0][0] == "Alex"
        assert iv.transcript[1] == {**iv.transcript[1], "who": "you", "q": "intro"}

    def test_opening_note_asks_for_the_first_question_word_for_word(self, cfg, bank):
        iv, io, llm = run_interview(cfg, bank, "Name: Sam.")
        note = llm.calls[0]["messages"][0]["content"]
        assert "ask exactly" in note and any(w in note for w in bank["intro"])
        assert "greet the candidate by name" in note and "about 30 minutes" in note

    def test_no_name_known(self, cfg, bank):
        iv, io, llm = run_interview(cfg, bank, "")
        assert "do not know their name" in llm.calls[0]["messages"][0]["content"]

    def test_skipped_intro_starts_with_a_question(self, cfg, bank):
        cfg["interview"]["intro_skip_chance"] = 1
        iv, io, llm = run_interview(cfg, bank)
        first = next(e for e in iv.transcript if e["who"] == "you")
        assert first["q"] != "intro" and "straight into questions" in llm.calls[0]["messages"][0]["content"]

    def test_opening_without_the_first_question_still_asks_it(self, cfg, bank):
        """A weak model may greet and stop. The candidate must still hear a question before answering."""
        def script(label, system, messages):
            return "Hi, I'm Alex from Payments." if label == "intro" else default_reply(label, system, messages)
        iv, io, llm = run_interview(cfg, bank, llm=FakeLLM(script))
        opening = " ".join(t for _, t in io.said[:2])
        assert any(norm(w) in norm(opening) for w in bank["intro"]), io.said[:2]

    def test_empty_opening_still_asks_the_first_question(self, cfg, bank):
        def script(label, system, messages):
            return "" if label == "intro" else default_reply(label, system, messages)
        iv, io, llm = run_interview(cfg, bank, llm=FakeLLM(script))
        assert io.said and any(norm(w) in norm(io.said[0][1]) for w in bank["intro"]), io.said[:1]

    def test_panel_opening_is_split_per_person(self, tmp_path):
        d = make_process(tmp_path / "p", panel=[
            {"role": "Lead", "opens": True, "people": [{"name": "Ann", "voice": "af_heart"}]},
            {"role": "Engineer", "people": [{"name": "Ben", "voice": "bm_george"}]}])
        cfg = P.load(d)

        def script(label, system, messages):
            if label == "intro":
                return "Hi, I'm Ann, I lead the team.\n##\nI'm Ben. Tell me about yourself."
            return default_reply(label, system, messages)
        iv, io, llm = run_interview(cfg, P.bank(cfg), llm=FakeLLM(script))
        assert [w for w, _ in io.said[:2]] == ["Ann", "Ben"]


# --- moving through the questions -------------------------------------------------------------------

class TestQuestionFlow:
    def test_follow_ups_are_capped_then_it_moves_on(self, cfg, bank):
        cfg["interview"]["max_followups"] = 2
        iv, io, llm = run_interview(cfg, bank)
        per_q = {}
        for c in llm.calls:
            if c["label"].startswith("q:") and c["label"] != "q:intro":
                per_q[c["label"]] = per_q.get(c["label"], 0) + 1
        assert per_q and max(per_q.values()) <= 3          # 2 follow-ups + the move-on turn

    def test_intro_gets_one_follow_up(self, cfg, bank):
        iv, io, llm = run_interview(cfg, bank)
        assert llm.labels().count("q:intro") == 2

    def test_next_question_is_spoken_exactly_as_written(self, cfg, bank):
        def script(label, system, messages):
            if label.startswith("q:"):
                return "[N] Great, so, moving on, maybe tell me something about your organising?"
            return default_reply(label, system, messages)
        iv, io, llm = run_interview(cfg, bank, llm=FakeLLM(script))
        spoken = [t for _, t in io.said[1:]]
        assert not any("moving on, maybe" in t for t in spoken)
        words = bank_wordings(bank)
        mains = [t for t in spoken if any(w in norm(t) for w in words)]
        assert len(mains) >= 3

    def test_model_ignoring_move_on_cannot_stall_the_interview(self, cfg, bank):
        def script(label, system, messages):
            return "[F] And then what?" if label.startswith("q:") else default_reply(label, system, messages)
        iv, io, llm = run_interview(cfg, bank, llm=FakeLLM(script))
        assert len(asked_main_questions(io, bank)) >= 3
        assert any("What questions do you have" in t for _, t in io.said)

    def test_paraphrased_next_question_counts_as_moved(self, cfg, bank):
        """A model that asks the next question in its own words with a follow-up tag must not cause the
        same question to be asked twice."""
        random.seed(3)
        plan_holder = {}

        def script(label, system, messages):
            last = messages[-1]["content"]
            if label.startswith("q:") and 'says exactly: "' in last:
                q = last.split('says exactly: "', 1)[1].split('"', 1)[0]
                plan_holder.setdefault("q", q)
                if q == plan_holder["q"]:
                    return "[F] So, in your own words: " + q
            return default_reply(label, system, messages)
        iv, io, llm = run_interview(cfg, bank, llm=FakeLLM(script))
        heard = [norm(t) for _, t in io.said]
        target = norm(plan_holder["q"]).split(" ", 0)[0]
        hits = [h for h in heard if S.asks_same(plan_holder["q"], h) or norm(plan_holder["q"]) in h]
        assert len(hits) == 1, hits

    def test_model_closing_early_does_not_say_goodbye_mid_interview(self, cfg, bank):
        def script(label, system, messages):
            if label.startswith("q:") and "Move on now" not in messages[-1]["content"]:
                return "[E] Thank you for your time, goodbye."
            return default_reply(label, system, messages)
        iv, io, llm = run_interview(cfg, bank, llm=FakeLLM(script))
        goodbyes = [i for i, (_, t) in enumerate(io.said) if "goodbye" in t.lower()]
        assert goodbyes == [len(io.said) - 1], io.said

    def test_main_question_ids_are_logged_and_counted(self, cfg, bank):
        iv, io, llm = run_interview(cfg, bank)
        ids = [e.get("q") for e in iv.transcript if e["who"] == "you"]
        mains = [i for i in dict.fromkeys(ids) if i and i != "intro"]
        assert mains and all(iv.weak["asked"].get(i) == 1 for i in mains)

    def test_memory_of_earlier_answers_is_passed_on(self, cfg, bank):
        io = ScriptedIO(answers=["I am Sam and I love Kafka streams."])
        iv, io, llm = run_interview(cfg, bank, io=io)
        later = [c for c in llm.calls if c["label"].startswith("q:") and c["label"] != "q:intro"]
        assert any("love Kafka" in c["messages"][0]["content"] for c in later)

    def test_value_probe_is_used_for_tagged_questions(self, tmp_path):
        d = make_process(tmp_path / "p", questions_doc={"questions": [
            {"id": "own_it", "theme": "ownership", "value": "Ownership", "core": True, "q": ["Owned anything?"]}]})
        cfg = P.load(d)
        cfg["interview"].update(curveball_chance=0, technical_chance=0)
        iv, io, llm = run_interview(cfg, P.bank(cfg))
        notes = [c["messages"][-1]["content"] for c in llm.calls if c["label"] == "q:own_it"]
        assert any("What did you do yourself?" in n for n in notes)


class TestCutIn:
    def test_long_answer_is_cut_in_on_and_keeps_model_wording(self, cfg, bank):
        long_turn = S.Turn(text="On and on " * 50, speech_seconds=170, cut_off=True)
        io = ScriptedIO(answers=["intro answer", long_turn])

        def script(label, system, messages):
            if "cuts in" in messages[-1]["content"]:
                return "[N] Sorry to stop you there, let's move on."
            return default_reply(label, system, messages)
        iv, io, llm = run_interview(cfg, bank, io=io, llm=FakeLLM(script))
        assert any("Sorry to stop you there" in t for _, t in io.said)


# --- panel ------------------------------------------------------------------------------------------

def panel_cfg(tmp_path, panel, **interview):
    cfg = P.load(make_process(tmp_path / "p", panel=panel))
    cfg["interview"].update(interview)
    return cfg


class TestPanel:
    def test_observer_never_speaks_but_is_introduced(self, tmp_path):
        cfg = panel_cfg(tmp_path, [
            {"role": "Bar Raiser", "opens": True, "people": [{"name": "Elena", "voice": "af_heart"}]},
            {"role": "Shadow", "observer": True, "people": [{"name": "Priya", "voice": "af_nicole"}]}],
            cross_followup_chance=1)
        iv, io, llm = run_interview(cfg, P.bank(cfg))
        assert {w for w, _ in io.said} == {"Elena"}
        assert "Priya (Shadow)" in llm.calls[0]["messages"][0]["content"]

    def test_opener_role_speaks_first(self, tmp_path):
        cfg = panel_cfg(tmp_path, [
            {"role": "Engineer", "people": [{"name": "Ben", "voice": "bm_george"}]},
            {"role": "Head", "opens": True, "people": [{"name": "Ann", "voice": "af_heart"}]}])
        iv, io, llm = run_interview(cfg, P.bank(cfg))
        assert io.said[0][0] == "Ann"

    def test_technical_follow_ups_only_with_a_technical_member(self, tmp_path):
        tech = panel_cfg(tmp_path / "a", [{"role": "Eng", "technical": True, "people": [{"name": "Ben"}]}],
                         technical_chance=1)
        iv, io, llm = run_interview(tech, P.bank(tech))
        assert any("technical curiosity" in c["messages"][-1]["content"] for c in llm.calls)
        plain = panel_cfg(tmp_path / "b", [{"role": "HM", "people": [{"name": "Ann"}]}], technical_chance=1)
        iv, io, llm = run_interview(plain, P.bank(plain))
        assert not any("technical curiosity" in c["messages"][-1]["content"] for c in llm.calls)

    def test_askers_alternate_on_a_panel(self, tmp_path):
        cfg = panel_cfg(tmp_path, [{"role": "A", "opens": True, "people": [{"name": "Ann", "voice": "af_heart"}]},
                                   {"role": "B", "people": [{"name": "Ben", "voice": "bm_george"}]}],
                        cross_followup_chance=0)
        iv, io, llm = run_interview(cfg, P.bank(cfg))
        askers = [w for w, t in io.said[2:] if any(norm(x) in norm(t) for x in bank_wordings(P.bank(cfg)))]
        assert {"Ann", "Ben"} <= set(askers)


# --- time and the end of the interview ----------------------------------------------------------------

class TestTiming:
    @pytest.mark.parametrize("minutes", [5, 15, 30, 60])
    def test_interview_ends_close_to_its_length(self, cfg, bank, minutes):
        cfg["interview"]["minutes"] = minutes
        cfg["interview"]["candidate_minutes"] = min(5, minutes // 4)
        iv, io, llm = run_interview(cfg, bank)
        assert io.clock[0] <= minutes * 60 + 6 * 90          # a few turns of slack for the closing

    def test_hand_over_to_candidate_questions_then_close(self, cfg, bank):
        iv, io, llm = run_interview(cfg, bank)
        labels = llm.labels()
        assert "cand_q" in labels and labels[-1] == "cand_q"
        assert "goodbye" in io.said[-1][1].lower()

    def test_candidate_question_limit(self, cfg, bank):
        cfg["interview"]["max_candidate_questions"] = 1

        def script(label, system, messages):
            return "[F] We ship weekly. Another question?" if label == "cand_q" else default_reply(label, system, messages)
        iv, io, llm = run_interview(cfg, bank, llm=FakeLLM(script))
        assert llm.labels().count("cand_q") == 1

    def test_question_cap(self, cfg, bank):
        cfg["interview"]["max_questions"] = 2
        iv, io, llm = run_interview(cfg, bank)
        mains = {e["q"] for e in iv.transcript if e.get("q") and e["q"] != "intro"}
        assert len(mains) == 2

    def test_silence_is_waited_out(self, cfg, bank):
        io = ScriptedIO(answers=["", "", "Hello, I'm Sam."])
        iv, io, llm = run_interview(cfg, bank, io=io)
        assert next(e for e in iv.transcript if e["who"] == "you")["text"] == "Hello, I'm Sam."


class TestUnfinishedAnswers:
    def test_thinking_pause_is_joined_in_voice_mode(self, cfg, bank):
        io = ScriptedIO(answers=[S.Turn(text="So I built the, and", speech_seconds=4), "the cache myself."])
        io.can_resume = True
        iv, io, llm = run_interview(cfg, bank, io=io)
        first = next(e for e in iv.transcript if e["who"] == "you")
        assert first["text"] == "So I built the, and the cache myself."
        assert io.listen_calls[1]["wait_seconds"] == 6.0

    def test_space_key_ends_the_answer(self, cfg, bank):
        io = ScriptedIO(answers=[S.Turn(text="So I built the, and", speech_seconds=4, by_key=True)])
        io.can_resume = True
        iv, io, llm = run_interview(cfg, bank, io=io)
        assert next(e for e in iv.transcript if e["who"] == "you")["text"] == "So I built the, and"


# --- every prompt the model gets is well formed ---------------------------------------------------------

def test_every_request_is_well_formed(cfg, bank):
    iv, io, llm = run_interview(cfg, bank, "Name: Sam.")
    for c in llm.calls:
        assert c["system"] and c["messages"] and c["messages"][-1]["role"] == "user"
        roles = [m["role"] for m in c["messages"]]
        assert all(a != b for a, b in zip(roles, roles[1:])), roles          # alternating turns
        assert all(isinstance(m["content"], str) and m["content"].strip() for m in c["messages"])
        assert "{" not in c["messages"][-1]["content"].replace("{0}", "")


def test_spoken_text_never_contains_tags_or_notes(cfg, bank):
    def script(label, system, messages):
        r = default_reply(label, system, messages)
        return "Let me think about it. " + r if label.startswith("q:") else r
    iv, io, llm = run_interview(cfg, bank, llm=FakeLLM(script))
    for _, t in io.said:
        assert "[F]" not in t and "[N]" not in t and "[E]" not in t and "<note>" not in t
