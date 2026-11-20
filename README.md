# AInterviewer: voice mock interviewer for any employer

You talk to it out loud, it talks back. Speech recognition (faster-whisper) and the voice (Kokoro) run
locally for free. The only tokens spent are short text turns with an LLM (Claude by default; OpenAI,
Gemini, OpenRouter and others, or a local model via Ollama / LM Studio / llama.cpp / vLLM): about 2-3k
input and fewer than 100 output tokens per interviewer turn (about 50k per interview), plus one debrief
call at the end. With a local model, nothing leaves your machine.

The tool knows nothing about any employer. Everything about a specific recruitment process (company,
role, round, team, panel, values, what they assess, why people fail) lives in a **process folder**
that you write. Point the tool at it and it plays that interview.

## No setup: the generic behavioural sample

Don't want to write a profile or share anything about yourself or an employer? Just run it:

```bash
~/Projects/AInterviewer/ainterviewer interview                              # generic behavioural interview
~/Projects/AInterviewer/ainterviewer interview --role "Data Analyst"        # pitched at a role
~/Projects/AInterviewer/ainterviewer drill --role "Product Manager" --level "graduate"
```

With no process folder around, the built-in `samples/behavioural` process is used (or pick it
explicitly with `-p behavioural`). It plays a hiring manager at Northwind, a made-up company,
who knows nothing about you in advance: no name, no CV, no employer. They ask about you instead of
assuming. Questions are the tool's generic behavioural bank, with role- and company-neutral
motivation questions and no engineering-only ones. `--role`, `--level` and `--company` change
the pitch without writing any files. Nothing goes to the model except what you say in the call.
Sessions and weak spots are kept locally in `~/.local/share/ainterviewer/behavioural/`.

## Your own process

Describe the recruitment process in your own words and let the LLM write the folder:

```bash
ainterviewer init ~/Projects/AcmePrep            # asks you to describe it (paste freely, Ctrl-D to finish)
ainterviewer init ~/Projects/AcmePrep -d "Hiring manager round at Acme for Senior Data Analyst, 45 min, \
  with the Head of Growth and a data scientist; recruiter said they care about influencing decisions" \
  -f job_ad.txt -f recruiter_email.md -f cv.pdf  # optional sources: .txt .md .html .pdf .docx
ainterviewer init ~/Projects/AcmePrep --provider ollama   # keep your description and CV on this machine
ainterviewer init ~/Projects/AcmePrep --template          # or just copy the commented template
```

It writes `process.yaml`, `questions.yaml` (questions specific to that company and round, such as
reported questions in their original wording, on top of the default bank) and, if you said anything
about yourself or passed a CV, `profile.md`. It uses only what you gave it: anything unknown is left at
the generic default and marked `# TODO:` in `process.yaml`, so you know what to ask the recruiter.

**What gets written always runs**, whatever the model's quality. Before anything is written, the folder
is loaded the way an interview loads it, then a whole interview, a drill round and the debrief are
dry-run on it: a fake model, a fake clock, no tokens. The dry run also checks that the interview
reaches its end. A draft that fails gets one retry; if that fails too, nothing is written and the
model's raw output is saved next to the folder. `ainterviewer check` runs the same dry run on any folder,
including ones you edited by hand.

The loader accepts the shapes small models (and people) write, and fixes them as it loads:
- numbers written as text (`minutes: "60 min"`), and out-of-range values;
- a string where a list is expected, people given as plain names, a panel given as one role;
- a value given as a string, a question's `value` given as a whole definition;
- repeated keys, broken quoting, tabs, Windows line endings, output cut off by a length cap
  (the unfinished tail is dropped).

`check` lists everything it read differently from how it was written. Only a missing or unparseable
file, or a bank with no usable questions, is rejected, with a message that names the problem.

On top of that, the content is checked against your sources:
- **Repairs:** broken YAML quoting and repeated keys are repaired (best-effort; the dry run has the
  final say).
- **Numbers:** every number in the profile and team facts must appear in your sources.
- **Unsupported claims:** profile and team-fact sentences your sources don't back up are listed for you
  to check.
- **Reported questions:** questions you quoted ("Tell me about a time...") are kept word for word.
- **Values:** a value a question uses is never dropped. Descriptions the model wrote from general
  knowledge get a TODO.
- **Themes:** company-specific questions get their own themes, so they are actually asked.
- **Observers:** a shadow or note-taker becomes an `observer`.
- **Retries:** real problems get one retry, with an instruction never to delete content to pass.

Your description (not file contents) is kept in `sources.md` for reference. Review the files, then:

```bash
cd ~/Projects/AcmePrep
~/Projects/AInterviewer/ainterviewer check                        # dry-run a whole interview on it (no tokens)
~/Projects/AInterviewer/ainterviewer show                         # see what the interviewer will be told
~/Projects/AInterviewer/ainterviewer interview                    # realistic interview + spoken debrief
~/Projects/AInterviewer/ainterviewer drill                        # one question at a time, instant feedback
~/Projects/AInterviewer/ainterviewer rescore sessions/<file>.md   # re-run the debrief on an old session
```
Tip: `ln -s ~/Projects/AInterviewer/ainterviewer ~/.local/bin/ainterviewer` to just type `ainterviewer`.

The process folder is taken from `-p/--process` (a folder or a built-in sample name), else
`$AINTERVIEWER_PROCESS`, else the current folder, else the built-in sample. Sessions (`sessions/*.md`)
and `weak_spots.json` are saved inside it, so each employer keeps its own history.

Options: `--text` (type answers, still hear the voice), `--text --mute` (all text), `--minutes 15`,
`--questions 3` (cap on main questions), `--no-debrief`, `--role`, `--level`, `--company`.

**Controls:** speak naturally and press **Space** when you've finished. Pauses never end your answer
(set `stt.end_turn: silence` to also end on a 2.5 s pause). The interviewer cuts in politely after
`max_answer_seconds` (150 s). **Ctrl-C** stops; the session is still saved. In drill mode, say "again",
"next" or "stop" after the feedback. Use a headset: speakers can leak into an open mic.

## The process folder

| File | What |
|---|---|
| `process.yaml` | The recruitment process. Only `company` is required; everything else falls back to a generic behavioural interview. |
| `profile.md` | Optional short profile (~120 words), shown to the interviewer every turn. Start it with `Name: <first name>.` Set `profile: ""` to go without one. |
| `questions.yaml` | Optional question bank. Without it, the tool's generic bank (`AInterviewer/questions.yaml`) is used; with `extends_default: true` it adds to that bank (`exclude: [ids]` drops default questions). |
| notes file | Optional prepared stories (`notes:` in process.yaml). Sent only with the debrief, which then names the best story for each question. |

What goes in `process.yaml` (see `templates/process/process.yaml` for a commented example):

| Key | Used for |
|---|---|
| `company`, `role`, `round`, `round_description`, `team`, `interviewer_from` | Who is interviewing and for what. |
| `format` | How the round runs, as the recruiter described it. |
| `assess`, `good_answers` | What the interviewer judges silently, and what a strong answer looks like. Drives follow-ups, drill feedback and debrief scores. |
| `candidate_level` | Calibrates questions and scoring (graduate vs senior, etc.). |
| `interviewer_rules` | Process-specific don'ts ("never ask why they left; they're still employed"). |
| `known_failure_reasons` | Why candidates are reported to fail; the debrief checks for these. |
| `team_facts` | What the panel knows about their team: used in their intro and to answer your questions. |
| `closing`, `candidate_question_ideas` | How they close the call; examples of sharper questions for the debrief to suggest. |
| `panel` | One entry per role, each with several `people` (name + voice); one person per role is picked each session. `opens: true` runs the call, `technical: true` asks technical-curiosity follow-ups, `observer: true` (a shadow or note-taker) is introduced but never speaks. |
| `values` | Optional company values: `brief` (sent every turn), `probe` (follow-ups while a question tagged with that value is open), `red_flags` (debrief). The debrief rates each value. |
| `vocabulary` | Names and terms speech recognition should spell right. |
| `interview:`, `llm:`, `stt:`, `tts:` | Override any tool setting from `config.yaml` for this process. |

Questions in the process's own `questions.yaml` are asked before the generic bank's. Questions can use
`{company}`, `{team}` and `{role}` in their wordings. Each question has a `theme`
(weak-spot tracking), optionally a `value`, and `core: true` if it should be asked first. One `motivation`
question is always placed early.

## What it does

- **Interviewer:** opens with a short intro of themselves and the team, explains the format, then asks
  you to introduce yourself (sometimes they skip that and go straight to a question). They close with
  the process's `closing` line. Panels of any size work; names and voices vary between sessions.
- **Interview:** core questions first, one per theme, weak themes earlier, plus one motivation question
  and sometimes one off-script extra (favouring a value the core questions barely cover). It's
  **time-driven**: questions continue until `candidate_minutes` are left, then you ask your questions.
- **Follow-ups:** up to `max_followups` per question, one short, plain question each, about something you
  just said. They ask for one specific time when you speak in general terms, for a different example when
  you reuse a story, and what *you* did when you say "we". Some are deliberately uncomfortable
  (`curveball_chance`), some are technical curiosity (`technical_chance`).
- **Natural wording:** each question has 2-4 human-written wordings; one is asked as written. No model
  rewording, no invented hypotheticals.
- **Debrief:** a spoken 3-4 sentence verdict, and a full report in `sessions/YYYY-MM-DD-HHMM-*.md`: system
  glitches (never scored against you), an overview table for every turn, answer-by-answer feedback with a
  model answer (only facts from your notes, `[your number]` where missing), patterns, values or
  assessed areas, self-contradictions, story choice, your questions at the end, delivery, and the top 3
  things to practise.
- **Local metrics, no tokens:** answer length, words per minute, filler words, "I" vs "we" count.
- **Weak spots:** `weak_spots.json` (in the process folder) tracks a score per theme; later sessions pick
  weak themes and rarely-asked questions more often.

## Tool settings (`config.yaml`)

Interview mechanics (length, follow-up limits, curveball/technical/intro-skip rates), audio device,
speech recognition, voice and models. A process can override any of them.

## LLM providers

| Provider | How | Key |
|---|---|---|
| `anthropic` | Anthropic API (Opus 5.5 uses server-side refusal fallbacks) | `ANTHROPIC_API_KEY`, or an `ant auth login` profile |
| `claude-cli` | `claude -p` with your Claude subscription | none |
| `openai`, `openrouter`, `groq`, `gemini`, `mistral`, `deepseek`, `together` | OpenAI-compatible API | `<NAME>_API_KEY`, e.g. `GROQ_API_KEY` |
| `ollama`, `lmstudio`, `llamacpp`, `vllm` | local server on its default port | none; nothing leaves the machine |
| your own | any OpenAI-compatible URL under `llm.providers` | `key_env` |

`llm.provider: auto` picks the Anthropic API if a key is set, else the `claude` CLI if installed, else a
running Ollama. For one run: `--provider ollama` (a local server's first model is used unless you add
`--model qwen3:14b`), or `--provider openrouter --model meta-llama/llama-3.3-70b-instruct`.

Models are set per job in `config.yaml → llm`: `model` (interviewer), `debrief_model`, `feedback_model`
(drill; `init` uses `debrief_model`). Prefix a model with a provider to mix them, e.g. a fast local
interviewer and a strong debrief:
```yaml
llm:
  model: ollama/qwen3:14b
  feedback_model: ollama/qwen3:14b
  debrief_model: anthropic/claude-opus-5-5     # or claude-cli/claude-opus-5-5
```
For OpenRouter, keep its own slash: `openrouter/anthropic/claude-opus-5-5`. A provider can also be
overridden or added per process (`llm:` in process.yaml).

Notes on other models:
- Some local servers cap every reply (one stops at 2,047 tokens and still reports a normal finish). The
  debrief therefore runs in parts when its model is local (`llm.debrief_mode: auto`; `sections` or
  `single` to force): verdict and overview, one call per main question, the closing sections, the scores.
- Small local models (tested: a 9B Qwen) run a usable interview. Their debriefs are too lenient and can
  blend stories or invent details in model answers. Keep the interviewer local and re-score with a strong
  model: `ainterviewer rescore <session file> --provider claude-cli`.
- Local models get at least 2048 output tokens per call (it costs nothing), so reasoning models (Qwen3,
  DeepSeek-R1, gpt-oss) have room to think. Their `<think>` blocks are stripped and never spoken. For
  the lowest latency, pick a non-thinking model or turn thinking off in the server.
- The interviewer follows instructions with tags ([F]/[N]/[E]). When it moves on, the next question is
  always spoken exactly as written, not in the model's words. An opening "That sounds great."-style
  evaluation is removed. If a weaker model ignores "move on" twice, the tool asks the next question
  itself, so the interview never gets stuck. The debrief needs a
  long, structured report, so it is the job where a strong model matters most.
- Measured on the Claude CLI, from the end of your answer to the first interviewer audio: Opus 5.5 about
  2.2 s, Haiku 5.5 about 1.7 s (Haiku loses the thread more).

## Troubleshooting

- **"microphone gives pure silence"**: the headset's mic-mute is on, or the wrong input is selected.
  Check with `pavucontrol` → Input Devices.
- **Wrong mic or speakers:** run `.venv/bin/python -m sounddevice` and set `audio_device`.
- **"Can't reach ollama at ..."**: start the server (`ollama serve`) and pull a model, or set
  `llm.providers.ollama.base_url` if it runs elsewhere. Same for LM Studio, llama.cpp and vLLM.
- **Silence or odd replies with a local model:** try a bigger model or turn its thinking off; check
  what it got with `ainterviewer show`.
- **Cut off while thinking:** raise `stt.silence_end_seconds`. **Transcripts wrong:** add names to
  `vocabulary`, or set `stt.model: medium.en`.

## Setup from scratch

```bash
cd ~/Projects/AInterviewer
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
mkdir -p models && cd models
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -LO https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```
The Whisper model downloads on first run (~500 MB for `small.en`).

## Testing

```bash
uv pip install --python .venv/bin/python -r requirements-dev.txt
.venv/bin/python -m pytest                     # unit + end-to-end, ~1 min, no tokens
.venv/bin/python -m pytest -m audio            # the real speech models, no microphone
AINTERVIEWER_LIVE=<provider or URL> .venv/bin/python -m pytest -m live   # a real LLM: finds misalignments
```
Plus `tests/manual/CHECKLIST.md` for the microphone, speakers and a full voice interview. `tests/README.md`
describes every layer and what it covers.
