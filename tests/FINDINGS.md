# Findings from building the test suite (2026-11-19)

Bugs and misalignments the suite found, what was done, and the test that now guards each. "Fixed" items
have a test that failed before the fix and passes after it.

## Bugs fixed in the code

| # | Found by | Problem | Fix | Guarded by |
|---|---|---|---|---|
| 1 | unit | An opening without the first question (a model that greets and stops), or an empty opening: the candidate was never asked anything but was expected to answer | The code asks the first question itself (with a minimal greeting if nothing was said) | `test_interview_flow.py::TestOpening::test_opening_without_*`, `test_empty_opening_*` |
| 2 | unit | A model closing early (`[E]`) mid-interview: "goodbye" said again and again while the interview went on | In the question phase `[E]` becomes the move-on line | `test_model_closing_early_does_not_say_goodbye_mid_interview` |
| 3 | unit | The hand-over to the candidate's questions used the model's wording (a weak model could ask yet another question there) | Fixed hand-over line | `test_next_question_is_spoken_exactly_as_written` |
| 4 | unit | Short paraphrased questions ("What's your biggest weakness?") were not recognised as the next question, so it could be asked twice | `asks_same` measures against the question's own content words | `TestAsksSame` |
| 5 | unit | Saving a session crashed when the data folder's parent didn't exist (first drill on the built-in sample, quit before answering) | `mkdir(parents=True)` | `TestSessionFiles` |
| 6 | unit | Re-reading a saved session dropped every paragraph after the first in a turn, and with it the question id; `rescore` lost text | Paragraphs join their turn; ids read from the end of the turn | `test_multi_paragraph_turns_survive` |
| 7 | unit | A missing `claude` CLI (or any unusable provider) only failed at the first model call, mid-interview after the audio models loaded | Every provider the run uses is set up at startup | `TestClaudeCLI::test_missing_cli`, `test_provider_errors` (e2e) |
| 8 | unit | Continuing a cut-off reply guessed spaces by word length ("one"+"two" -> "onetwo" or "wor"+"ld" -> "wor ld") | Joined exactly as the model continues | `TestCompleteUntil::test_continues_until_done` |
| 9 | unit | `init` asked the model to "continue" a reply that had no sections at all (garbage, not a cut-off), and saved the doubled text | Continue only replies that start the files but don't finish them | `TestGenerate::test_two_bad_drafts_write_nothing` |
| 10 | unit | Between two drafts with invented numbers, the one with *more* wrong numbers could be kept (counted fields, not numbers) | Count the numbers | `test_invented_number_gets_one_retry_and_keeps_the_better_draft` |
| 11 | unit | Cut-off YAML in a 2-3 line file was never salvaged (off-by-one) | Range fixed | `TestLoadYaml::test_cut_off_tail_is_dropped` |
| 12 | unit | Pressing Space discarded audio still queued: the last words of an answer could be lost | Drain the queue before ending the turn | `test_space_keeps_audio_still_queued` |
| 13 | audio | Whisper turned faint room noise into words ("Yeah."), which would be logged as an answer | Energy gate + Whisper's no-speech confidence | `test_near_silence_is_not_a_hallucinated_answer`, `test_quiet_but_real_speech_is_still_heard` |
| 14 | live | A second tag inside a reply ("...answer. [E] Thanks, goodbye.") was read aloud and its action ignored | Later tags are removed; the last one decides | `TestLiveFindings::test_later_tag_*` |
| 15 | live | Multi-line YAML values with unindented continuation lines (a CV-only `init`) failed both drafts | Continuation lines are indented before loading | `test_generate.py::TestLiveFindings::test_unindented_continuation_lines` |
| 16 | live | A guessed interview length (45 min, never mentioned) was only warned about | Guessed `interview:` settings are removed; the default applies | `test_guessed_interview_settings_are_removed` |
| 17 | live | Four debrief sections "disappeared": the model wrote `7. "# Story choice": text` (the prompt's notation, copied) | Headings standardised, including that form | `test_prompt_notation_copied_literally` |
| 18 | live | A sentence starting with "Scores ..." / "Report ..." was taken for a section label, cutting the report there | Labels must stand alone or end with a colon | `test_prose_starting_with_a_label_word_does_not_cut_the_report` |

## Model misalignments (behaviour of a model, mitigated in the tool)

Found with the live tests on Qwen3.5-9B (int4, local). Re-run `-m live` on another model to compare.

| Rule | Status on the 9B model | Mitigation in the tool |
|---|---|---|
| Debrief model answers use only numbers that came up | **Fails**: invents figures (e.g. 5, 6, 100) | The report ends with a **Check** note listing them; the CLI suggests `rescore` with a stronger model |
| `init` invents numbers or settings | Seen once (a 45-minute length) | Guessed settings removed; other invented numbers warned about and retried |
| One short follow-up question | Passes now; earlier sessions showed long two-part follow-ups | None needed beyond the prompt; watch in the manual checklist |
| No praise | Passes; seen in an earlier session ("That sounds efficient.") | Opening evaluations are stripped |
| Main questions asked as written | Passes | Enforced in code |
| Observer silent, close with next steps, no invented premise | Pass | Observer enforced in code |
| Weak candidate not rated a hire | Passes on this run; earlier debriefs were lenient ("Strong hire") | `rescore` with a stronger model |

## Known limits (not bugs, recorded so changes are noticed)

- A reply that starts *without* a tag and has one later ("The answer was vague. [F] ...") is treated as a note
  to itself and its first part is not spoken. This is right for the notes small models write, but would drop
  real content written before a late tag.
- The unsupported-claim check is a word-overlap heuristic: it can miss a rewording that changes the meaning,
  and flag a faithful one.
- Interviewer quality (relevance, tone, realism) is judged only by rules in the live tests and the manual
  checklist; there is no scoring of it.
