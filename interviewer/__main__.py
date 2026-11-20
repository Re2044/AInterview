"""AInterviewer: a voice mock interviewer for any employer's behavioural / fit round.

    ainterviewer init <dir>                 # describe the recruitment process; the LLM writes the folder
    ainterviewer init <dir> -d "..." -f job_ad.txt -f cv.pdf          # ...non-interactively, from text and files
    ainterviewer init <dir> --template      # just copy the commented template
    ainterviewer interview                  # no setup: the built-in generic behavioural sample
    ainterviewer interview --role "Product Manager"     # ...narrowed to a role, still no personal data
    ainterviewer interview [-p DIR]         # realistic interview for your own process, debrief at the end
    ainterviewer drill [-p DIR]             # one question at a time, instant feedback
    ainterviewer rescore <session.md>       # re-run the debrief on a saved session
    ainterviewer show [-p DIR]              # check what the interviewer will be told
    ainterviewer check [-p DIR]             # dry-run a whole interview on the process (no tokens)
    add --text to type instead of speaking, --mute to also silence the voice
    add --provider ollama [--model qwen3:14b] to use another LLM (see config.yaml: llm)

The process folder (process.yaml, profile, questions, notes) comes from --process (a folder or a
built-in sample name), else $AINTERVIEWER_PROCESS, else the current folder, else the built-in
'behavioural' sample. Sessions and weak spots are saved in the process folder (for built-in samples:
~/.local/share/ainterviewer/<sample>/).
"""
import argparse
import os
import sys
import time
from pathlib import Path

from . import process as P
from .generate import ask_description, generate
from .llm import LLM, provider_names
from .session import (ROOT, Interview, pick_panel, TextIO, VoiceIO, debrief, load_bank, load_weak,
                      parse_session_md, run_drill, save_session, save_weak, update_weak, use_data_dir)


def main():
    try:
        return _main()
    except P.ProcessError as e:
        raise SystemExit(f"Process problem: {e}")


def _main():
    ap = argparse.ArgumentParser(prog="ainterviewer")
    ap.add_argument("mode", choices=["interview", "drill", "rescore", "init", "show", "check"],
                    help="show = print the resolved process and the interviewer's system prompt; "
                         "check = load the process and dry-run a whole interview on it (no tokens)")
    ap.add_argument("target", nargs="?", help="rescore: a saved sessions/*.md file; init: the new process folder")
    ap.add_argument("-p", "--process", help="recruitment process folder (contains process.yaml), or a built-in "
                                            f"sample: {', '.join(P.samples())}")
    ap.add_argument("--company", help="override the company name")
    ap.add_argument("--role", help="override the role, e.g. \"Data Analyst\"")
    ap.add_argument("--level", help="override the candidate level, e.g. \"graduate\" or \"8 years, team lead\"")
    ap.add_argument("--text", action="store_true", help="type answers instead of speaking")
    ap.add_argument("--mute", action="store_true", help="with --text: don't speak either")
    ap.add_argument("--minutes", type=int, help="override interview length")
    ap.add_argument("--questions", type=int, help="cap on main questions (default: as many as time allows)")
    ap.add_argument("--no-debrief", action="store_true")
    ap.add_argument("--provider", help="LLM provider for this run: " + ", ".join(provider_names())
                    + ", or the URL of any OpenAI-compatible server (http://127.0.0.1:8000/v1)")
    ap.add_argument("--model", help="model for every job this run (interviewer, feedback, debrief, init)")
    ap.add_argument("-d", "--describe", help="init: the recruitment process in your own words")
    ap.add_argument("-f", "--file", action="append", default=[],
                    help="init: a job ad, recruiter email, candidate report or CV (.txt .md .html .pdf .docx); repeatable")
    ap.add_argument("--template", action="store_true", help="init: copy the template instead of generating")
    args = ap.parse_args()

    if args.mode == "init":
        return init(args)

    if args.target:
        args.target = str(Path(args.target).expanduser().resolve())   # before the chdir below
    if args.mode == "rescore" and args.target and not args.process:
        # a session file lives in <process>/sessions/: use that process
        guess = Path(args.target).parent.parent
        if (guess / "process.yaml").exists():
            args.process = str(guess)
        elif guess.parent == P.DATA_ROOT and guess.name in P.samples():
            args.process = guess.name

    pdir = P.find_dir(args.process)
    cfg = P.load(pdir)
    use_data_dir(P.data_dir(pdir))
    for key in ("company", "role"):
        if getattr(args, key):
            cfg["process"][key] = getattr(args, key)
    if args.level:
        cfg["process"]["candidate_level"] = args.level
    if args.minutes:
        cfg["interview"]["minutes"] = args.minutes
    if args.questions:
        cfg["interview"]["max_questions"] = args.questions
    profile = P.profile(cfg)
    notes = P.notes(cfg)
    bank = load_bank(cfg)
    os.chdir(ROOT)                                    # model paths in config.yaml are relative to the tool
    p = cfg["process"]
    print(f"Process: {p['company']} · {p['round']} · {p['role']}  ({pdir})"
          + (f"\nSessions: {P.data_dir(pdir)}" if P.is_sample(pdir) else ""))

    if cfg.get("load_notes") and args.mode in ("check", "show"):
        print("Read as: " + "; ".join(dict.fromkeys(cfg["load_notes"])))
    if args.mode == "check":
        from .selftest import dry_run
        try:
            print(dry_run(cfg, bank, profile, notes))
        except Exception as e:                       # noqa: BLE001 - report any failure as a failed check
            raise SystemExit(f"Check failed: {type(e).__name__}: {e}")
        return

    if args.mode == "show":
        from .session import interviewer_system
        print(f"{len(bank['questions'])} questions ({sum(bool(q.get('core')) for q in bank['questions'])} core), "
              f"{len(cfg['values'])} values, notes: {'yes' if notes else 'no'}\n")
        print(interviewer_system(cfg, pick_panel(cfg), profile))
        return

    cfg["llm"] = llm_config(cfg["llm"], args)          # the session code reads models from here too
    llm = LLM(cfg["llm"])
    print(f"LLM: {llm.backend}")
    panel = pick_panel(cfg)

    if args.mode == "rescore":
        return rescore(args.target, cfg, llm, bank, profile, notes)

    print(f"Loading local voice models (LLM backend: {llm.backend})...", flush=True)
    t0 = time.monotonic()
    speaker = None
    keys = None
    listener = None
    if not (args.text and args.mute):
        from .audio import Speaker, setup_devices
        setup_devices(cfg.get("audio_device", "auto"))
        speaker = Speaker(cfg["tts"], voices=[m["voice"] for m in panel])
    if args.text:
        io = TextIO(speaker)
    else:
        from .audio import Keys, Listener
        keys = Keys()
        listener = Listener(cfg["stt"], keys)
        listener.warmup()
        if listener.mic_level() == 0.0:
            import sounddevice as sd
            print(f"\033[1;31mWarning: the microphone gives pure silence ({sd.query_devices(kind='input')['name']} "
                  f"-> system default source). Check the headset's mic-mute button, or pick the input in "
                  f"pavucontrol. Continuing anyway.\033[0m")
        io = VoiceIO(listener, speaker)
    print(f"Ready in {time.monotonic() - t0:.1f}s. Ctrl-C ends the session (it is still saved).\n")

    transcript: list[dict] = []
    report = ""
    try:
        if args.mode == "interview":
            print("Panel today: " + ", ".join(f"{m['name']} ({m['role']})" for m in panel))
            iv = Interview(cfg, io, llm, bank, profile, panel)
            transcript = iv.transcript
            try:
                iv.run()
            finally:
                save_weak(iv.weak)
        else:
            run_drill(cfg, io, llm, bank, profile, transcript)
    except KeyboardInterrupt:
        print("\n\nSession stopped.")
    except EOFError:
        pass

    try:
        answered = [e for e in transcript if e["who"] == "you"]
        if args.mode == "interview" and answered and not args.no_debrief:
            print("\nPreparing feedback... (Ctrl-C to skip)", flush=True)
            spoken, report, scores = debrief(cfg, llm, transcript, bank, profile, notes)
            weak = load_weak()
            for theme, s in scores.items():
                if isinstance(s, (int, float)):
                    update_weak(weak, theme, float(s))
            save_weak(weak)
            if llm.is_local(cfg["llm"]["debrief_model"]):
                print("\033[2m[the debrief was written by a local model; small models are lenient and can invent "
                      "details in model answers. For a reliable report: ainterviewer rescore <session file> "
                      "--provider claude-cli]\033[0m", flush=True)
            if spoken:
                report = f"**Verdict:** {spoken}\n\n{report}"
                print("\nThe full answer-by-answer feedback is in the session file below.", flush=True)
                io.say([spoken])
    except KeyboardInterrupt:
        print("Debrief skipped.")
    finally:
        if keys:
            keys.restore()
        if transcript:
            path = save_session(args.mode, transcript, report, llm)
            print(f"\nSaved: {path}" + ("  <- open this for feedback on every answer" if report else ""))
        tin, tout = llm.totals()
        print(f"Tokens this session ({llm.backend}): {tin} input, {tout} output, {len(llm.usage)} calls",
              flush=True)
        if listener:
            listener.close()
        # audio libraries can leave non-daemon threads behind; everything is saved, so exit hard
        os._exit(0)


def llm_config(llm_cfg: dict, args) -> dict:
    """Apply --provider / --model for this run. A provider switch without --model drops Claude model names
    for non-Claude providers (local servers then use their first model)."""
    c = dict(llm_cfg)
    if args.provider:
        c["provider"] = args.provider
    jobs = ("model", "debrief_model", "feedback_model")
    if args.model:
        for k in jobs:
            c[k] = args.model
    elif args.provider and args.provider not in ("anthropic", "claude-cli"):
        for k in jobs:
            if "/" not in str(c.get(k, "")):
                c[k] = "auto"
    return c


def init(args):
    if not args.target:
        raise SystemExit("usage: ainterviewer init <dir> [-d \"description\"] [-f file ...] [--template]")
    dst = Path(args.target).expanduser().resolve()
    if (dst / "process.yaml").exists():
        raise SystemExit(f"{dst} already has a process.yaml; not overwriting.")
    if args.template:
        return P.init(args.target)
    description = args.describe or ""
    if not description and not args.file:
        if not sys.stdin.isatty():
            description = sys.stdin.read().strip()
        else:
            description = ask_description()
    if not description and not args.file:
        print("Nothing described: copying the template instead.")
        return P.init(args.target)
    cfg = P.tool_config()
    llm = LLM(llm_config(cfg["llm"], args))
    generate(dst, description, args.file, llm, model=llm.cfg.get("debrief_model") or llm.cfg["model"])
    tin, tout = llm.totals()
    print(f"Tokens: {tin} input, {tout} output")


def rescore(session: str | None, cfg, llm, bank, profile, notes):
    """Re-run the debrief on a saved session with the current debrief model. Writes <name>-rescored.md
    next to it; the original file and weak_spots.json are left unchanged."""
    if not session:
        raise SystemExit("usage: ainterviewer rescore <process>/sessions/<file>.md")
    src = Path(session)
    transcript = parse_session_md(src.read_text())
    if not any(e["who"] == "you" for e in transcript):
        raise SystemExit(f"No answers found in {src}")
    print(f"Re-scoring {src.name} with {cfg['llm']['debrief_model']} "
          f"({sum(e['who'] == 'you' for e in transcript)} answers)...", flush=True)
    spoken, report, scores = debrief(cfg, llm, transcript, bank, profile, notes)
    out = src.with_name(src.stem + "-rescored.md")
    tin, tout = llm.totals()
    out.write_text(f"# Re-scored: {src.name} ({cfg['llm']['debrief_model']})\n\n"
                   f"## Verdict\n\n{spoken}\n\n## Debrief\n\n{report}\n\n## Theme scores\n\n{scores}\n\n"
                   f"_Tokens: {tin} in / {tout} out. Transcript: see {src.name}._\n")
    print(f"\n{spoken}\n\nSaved: {out}")


if __name__ == "__main__":
    sys.exit(main())
