# Testing AInterviewer

Four layers, from fast and free to slow and manual. Run them in this order after a change; stop at the
first layer that fails.

```bash
cd ~/Projects/AInterviewer
uv pip install --python .venv/bin/python -r requirements-dev.txt     # once: pytest

.venv/bin/python -m pytest                                            # 1+2. unit + end-to-end, ~1 min, no tokens
.venv/bin/python -m pytest -m audio                                   # 3a. real speech models, ~20 s, no mic
AINTERVIEWER_LIVE=http://127.0.0.1:8000/v1 .venv/bin/python -m pytest -m live   # 3b. a real LLM, ~10 min local
AINTERVIEWER_LIVE=claude-cli .venv/bin/python -m pytest -m live                 #     (or any --provider value)
# 4. tests/manual/CHECKLIST.md: microphone, speakers, keyboard, a full voice interview
```

`ainterviewer check -p <folder>` is the same dry run as the suite's, for a single process folder.

## 1. Unit tests: `tests/unit/` (default run)

Every module on its own, with test doubles (`tests/helpers.py`): a scriptable fake LLM, a scripted
candidate on a fake clock, an in-process OpenAI-compatible server, and a fake `claude` command.

| File | Covers |
|---|---|
| `test_process.py` | Finding the process folder (flag, env, cwd, built-in sample); loading and normalizing every key (numbers written as text, ranges, chances as percentages, text/list/mapping shapes, panels, people, voices, observers, values, vocabulary); profile and notes; the question bank (extend/replace/exclude, placeholders, wording shapes, ids, themes, errors); the shipped sample, template and default bank; `init --template` |
| `test_session_helpers.py` | The [F]/[N]/[E] tag reader (chunking, notes before the tag, speaker labels, names, praise stripping, fixed replacements); paraphrase detection; unfinished answers; question picking (themes, motivation, process questions first, extras, weak themes); weak spots; prompt builders; metrics; saving and re-reading sessions |
| `test_interview_flow.py` | A whole interview: the opening (with and without the first question, empty, panels), follow-up limits, exact next questions, a model that won't move on, paraphrased move-ons, early closing, cut-ins, observers, technical follow-ups, timing for 5-60 minutes, the candidate's questions, silence, joining thinking pauses, well-formed requests |
| `test_drill_debrief.py` | Drill (scores never spoken, commands, weak spots, feedback model); the debrief in one call and in sections (label variants, missing labels, bad JSON, headings, per-question transcript slices, the extra scores call), main-question detection |
| `test_llm.py` | Provider routing (URL providers, prefixes, auto, errors at startup); the OpenAI-compatible backend (streaming, usage, think-stripping, rejected `stream_options`, 4xx/5xx, a server that dies, keys and options); `complete_until`; the claude CLI backend; the Anthropic backend (fallbacks, effort, refusal, rate limits, bad key) |
| `test_generate.py` | `init` from a description: reading sources (txt, html, docx, pdf), sections, YAML repairs (quoting, repeated keys, tabs, cut-off tails), value/voice/observer/reported-question/theme repairs, keeping comments, invented-number and unsupported-claim checks, `_build` on good, bad and cut-off drafts, and `generate()` end to end with retries and failure |
| `test_selftest_cli.py` | The dry run (shipped processes, no side effects, catches never-ending interviews and broken prompts); per-run `--provider`/`--model` handling |
| `test_audio_logic.py` | Text cleanup for speech, the STT prompt, the speaker's sentence streaming (fake Kokoro), the listener's turn-taking (Space, pauses, time limit, queued audio on Space, chunked transcription) with fake VAD/STT |

## 2. End-to-end: `tests/e2e/` (default run)

The real `ainterviewer` command in a subprocess, against a stand-in model server (`--provider <url>`),
with typed answers on stdin and data folders redirected to a temp dir: `--help`, `show`, `check`
(shipped, messy and broken processes), `init` (template, description, files, stdin, nothing, failure),
a typed interview with debrief, the sample's data location, an interview ended early, `--no-debrief`,
drill, `rescore`, and provider errors (unknown, unreachable, missing key, a 503 mid-interview). Every
run is checked for tracebacks.

## 3. Integration: `tests/integration/` (opt-in)

- `test_audio_models.py` (`-m audio`): Kokoro for every panel voice; Whisper accuracy and vocabulary on
  synthesized speech; no invented words in near-silence; quiet speech still heard; the whole listening
  pipeline (Silero VAD + chunked Whisper) on an answer with a thinking pause.
- `test_live_llm.py` (`-m live`, `AINTERVIEWER_LIVE=<provider or URL>`, optional `AINTERVIEWER_LIVE_MODEL`):
  a real interview, debrief, drill and `init` on three kinds of description. Each test is one rule the
  model should follow, so a run lists every **misalignment**: long or multi-part follow-ups, praise,
  markdown, main questions not asked as written, an observer speaking, follow-ups about things never
  said, the name overused, no next steps at the close; debrief sections missing, verdict/score format,
  numbers in model answers that never came up, a lenient verdict for a weak candidate; drill feedback
  length; `init` inventing numbers, rewording reported questions, dropping named values, missing the
  observer or the length. Failures here are findings about a model, not necessarily bugs in the code.

## 4. Manual: `tests/manual/`

- `CHECKLIST.md`: devices, a full voice interview, each provider, setup by description, and a log for
  misalignments you notice.
- `sim_listen.py`, `latency.py`, `sim_interview.py`: scripts for listening, reply latency and a full
  interview with a real model on a fake clock (see each file's docstring).

## Findings

`FINDINGS.md` lists the bugs and model misalignments this suite has found, what was done about each,
and the test that guards it.

## Writing a test for a bug

Reproduce it at the lowest layer that shows it (a unit test with `FakeLLM`/`ScriptedIO`, or `FakeServer`
for HTTP behaviour), watch it fail, fix, and keep the test. A model misbehaving in a new way becomes a rule
in `test_live_llm.py`, plus a unit test with a scripted reply that reproduces it, so the code's response to
that behaviour is tested without a model.
