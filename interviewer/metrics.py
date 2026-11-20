"""Delivery metrics computed locally (no LLM)."""
import re

FILLERS = re.compile(r"\b(um+|uh+|erm+|hmm+|ah+|you know|basically|kind of|sort of|i mean)\b", re.I)
LIKE_FILLER = re.compile(r",\s*like,|\blike,\s", re.I)


def answer_metrics(text: str, seconds: float) -> dict:
    words = re.findall(r"[A-Za-z']+", text)
    lower = [w.lower() for w in words]
    fillers = len(FILLERS.findall(text)) + len(LIKE_FILLER.findall(text))
    i_count = sum(w in ("i", "i'm", "i've", "i'd", "my", "me") for w in lower)
    we_count = sum(w in ("we", "we're", "we've", "we'd", "our", "us") for w in lower)
    return {
        "seconds": round(seconds),
        "words": len(words),
        "wpm": round(len(words) / seconds * 60) if seconds > 5 else None,
        "fillers": fillers,
        "i": i_count,
        "we": we_count,
    }


def format_metrics(m: dict) -> str:
    wpm = f", {m['wpm']} wpm" if m.get("wpm") else ""
    return f"{m['seconds']}s, {m['words']} words{wpm}, {m['fillers']} fillers, I/we {m['i']}/{m['we']}"
