# Manual test checklist

What the automated suite cannot reach: a real microphone, speakers, the keyboard in a terminal, and how a
session *feels*. Run this after changes to `audio.py`, the interview flow, prompts, or before relying on a
new provider. Tick each item; write what you saw next to anything that fails.

Setup: a headset, `cd` into a process folder (or use the built-in sample), and a provider that works
(`ainterviewer check` passes first).

## 1. Devices and audio (5 min)

| # | Do | Expect |
|---|---|---|
| 1.1 | `ainterviewer drill` with the headset mic muted | Red warning "the microphone gives pure silence"; the session continues |
| 1.2 | Unmute; `ainterviewer drill` | The question is spoken in the coach's voice; `[listening...]` appears after it ends, not during |
| 1.3 | Answer for ~20 s with one 3-4 s thinking pause, then press **Space** | `[recording]` at your first word; the pause does **not** end the answer; Space ends it at once; the transcript shows both halves, including your last words |
| 1.4 | Say "again" after the feedback | The same question is asked again |
| 1.5 | Say "next", then answer, then "stop" | A different question, then the session ends and `Saved:` prints a path |
| 1.6 | Speakers instead of a headset (optional) | Note whether the interviewer's voice leaks into your transcript |
| 1.7 | Set `audio_device` to a wrong name in config.yaml | A clear error, not a hang; revert afterwards |
| 1.8 | `.venv/bin/python -m tests.manual.sim_listen` | Prints the synthesized answer back with plausible `speech_seconds` and `stt_tail` < 1 s |

## 2. A full voice interview (35 min)

Run `ainterviewer interview` and answer as yourself. Keep notes of anything odd.

| # | Check | Expect |
|---|---|---|
| 2.1 | Opening | Greets you (by name only if the profile has one), introduces the role and team, says the length, ends with a question you can answer |
| 2.2 | Voices | Each panel member keeps one voice; an observer is introduced but never speaks |
| 2.3 | Latency | First audio within ~2-3 s of pressing Space (Claude CLI) or ~1-2 s (local model); note outliers |
| 2.4 | Follow-ups | One short question each, about what you just said; never "great answer"; no reading of tags, markdown or notes |
| 2.5 | "We" answer | Give a vague "we did it" answer: the next follow-up asks what *you* did |
| 2.6 | Reused story | Tell the same story twice: you are asked for a different example (not required every time; note it) |
| 2.7 | Long answer | Talk past 2.5 minutes: the interviewer cuts in politely |
| 2.8 | Clock | The `[mm:ss / 30:00]` prompt advances; questions stop with ~5 minutes left; you are invited to ask questions |
| 2.9 | Your questions | Two questions get short, plausible answers; then a closing with next steps and goodbye |
| 2.10 | Debrief | A 3-4 sentence spoken verdict; `Saved:` path; the file has every section (see the list in tests/integration/test_live_llm.py) |
| 2.11 | Ctrl-C mid-answer in a second run | "Session stopped."; the session is still saved; the terminal is not left in raw mode (typing echoes normally) |

## 3. Providers (10 min each, as needed)

| # | Do | Expect |
|---|---|---|
| 3.1 | `ANTHROPIC_API_KEY=... ainterviewer interview --text --mute --questions 1` | `LLM: anthropic: claude-opus-5-5`; normal turns |
| 3.2 | Same with an invalid key | A one-line authentication error at the first turn, no traceback |
| 3.3 | `ainterviewer interview --provider claude-cli --text --mute --questions 1` | Works with the subscription; token counts in the footer |
| 3.4 | `ollama serve` + `ainterviewer interview --provider ollama --text --mute` | Uses the first pulled model; no `<think>` text appears |
| 3.5 | Stop the local server mid-interview | A clear "Can't reach ..." message; the session up to that point is saved |
| 3.6 | `--provider http://127.0.0.1:8000/v1` (any OpenAI-compatible server) | Same as 3.4 |
| 3.7 | A mixed setup: local `model`, `debrief_model: claude-cli/claude-opus-5-5` | `LLM:` line shows both; the debrief comes from Claude |

## 4. Process setup by description (10 min)

| # | Do | Expect |
|---|---|---|
| 4.1 | `ainterviewer init /tmp/x` and paste a real recruiter email, Ctrl-D | A folder with process.yaml, questions.yaml, sources.md (and profile.md if you described yourself) |
| 4.2 | Read process.yaml | Nothing that isn't in your text, except TODO comments and interviewer first names |
| 4.3 | `ainterviewer check -p /tmp/x` | `dry run ok` |
| 4.4 | `ainterviewer show -p /tmp/x` | The interviewer prompt reads sensibly for that company and round |
| 4.5 | `init` with a CV PDF (`-f cv.pdf`) | profile.md reflects the CV; numbers match exactly |

## 5. Misalignment log

While running sections 2-4, note every time the model (or the tool) does something a real interviewer
wouldn't. These feed prompt or code changes, and new rules in `tests/integration/test_live_llm.py`.

| When | What it did | What it should have done | Model / provider |
|---|---|---|---|
| | | | |
