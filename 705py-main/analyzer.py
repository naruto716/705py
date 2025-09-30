# analyzer.py
"""
Stream+Deepgram transcript analyzer (terminal only, no CSV).
- Reads JSONL produced by main.py (transcripts/YYYY-MM-DD-<call_id>.jsonl)
- Prints real-time sentiment for each utterance
- Maintains per-user running stats (pos/neu/neg counts, avg score)
Usage examples:
  uv run python analyzer.py --follow
  uv run python analyzer.py --follow --from-start
  uv run python analyzer.py --file transcripts/2025-09-30-abc123.jsonl --follow
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import re
from collections import defaultdict

# Optional deps with graceful fallback
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:
    SentimentIntensityAnalyzer = None

try:
    from snownlp import SnowNLP
except Exception:
    SnowNLP = None

try:
    from langdetect import detect
except Exception:
    detect = None


def looks_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text or ""))


def detect_lang(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "unk"
    if looks_chinese(text):
        return "zh"
    if detect:
        try:
            code = detect(text)
            if code.startswith("zh"):
                return "zh"
            if code.startswith("en"):
                return "en"
            return code
        except Exception:
            pass
    return "en"


class Sentiment:
    """English => VADER; Chinese => SnowNLP; falls back to neutral."""

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer() if SentimentIntensityAnalyzer else None
        self.cn_ok = SnowNLP is not None

    def score_en(self, text: str):
        if not self.vader:
            return "neu", 0.0
        comp = float(self.vader.polarity_scores(text)["compound"])
        if comp >= 0.05:
            lab = "pos"
        elif comp <= -0.05:
            lab = "neg"
        else:
            lab = "neu"
        return lab, comp

    def score_zh(self, text: str):
        if not self.cn_ok:
            return "neu", 0.0
        try:
            p = float(SnowNLP(text).sentiments)  # 0..1
            comp = 2 * p - 1.0                   # map to [-1,1]
            if comp >= 0.05:
                lab = "pos"
            elif comp <= -0.05:
                lab = "neg"
            else:
                lab = "neu"
            return lab, comp
        except Exception:
            return "neu", 0.0

    def score(self, text: str, lang_hint: str = "auto"):
        lang = detect_lang(text) if lang_hint == "auto" else lang_hint
        if lang.startswith("zh"):
            lab, score = self.score_zh(text)
            return lab, score, "zh"
        lab, score = self.score_en(text)
        return lab, score, "en"


def iter_jsonl(path: Path, follow: bool = False, from_start: bool = False):
    """Yield JSON records; if follow=True, keep watching for new lines."""
    with path.open("r", encoding="utf-8") as f:
        if follow and not from_start:
            f.seek(0, 2)  # jump to end (tail -f)
        while True:
            line = f.readline()
            if not line:
                if follow:
                    time.sleep(0.2)
                    continue
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if follow:
                    time.sleep(0.1)
                    continue


def fmt_ts(ts: float | int | None) -> str:
    if not ts:
        return ""
    try:
        return datetime.fromtimestamp(float(ts), tz=timezone.utc).astimezone().isoformat(timespec="seconds")
    except Exception:
        return ""


def print_user_summary(user_stats: dict[str, dict]):
    """Print a compact per-user summary line."""
    if not user_stats:
        return
    lines = []
    for u, s in user_stats.items():
        n = s["n"]
        avg = s["sum"] / n if n else 0.0
        lines.append(f"{u}: pos={s['pos']} neu={s['neu']} neg={s['neg']} avg={avg:+.3f} (n={n})")
    print("— User sentiment summary —")
    for ln in lines:
        print("  " + ln)


def main():
    ap = argparse.ArgumentParser(description="Terminal-only sentiment analyzer for Stream+Deepgram transcripts.")
    ap.add_argument("--file", "-f", type=str, help="Path to JSONL (default: latest in ./transcripts)")
    ap.add_argument("--follow", "-F", action="store_true", help="Follow file for new lines (tail -f)")
    ap.add_argument("--from-start", action="store_true", help="Read from file start before following")
    ap.add_argument("--include-partials", action="store_true", help="Also print 'partial' events")
    ap.add_argument("--minlen", type=int, default=2, help="Ignore utterances shorter than N chars")
    ap.add_argument("--lang", type=str, default="auto", help="Force language: auto/en/zh")
    ap.add_argument("--summary-every", type=int, default=10, help="Print per-user summary every N finals")
    args = ap.parse_args()

    # Locate file
    if args.file:
        in_path = Path(args.file)
    else:
        trans_dir = Path("transcripts")
        cands = sorted(trans_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not cands:
            print("No transcripts found in ./transcripts. Specify --file.")
            sys.exit(1)
        in_path = cands[0]

    if not in_path.exists():
        print(f"File not found: {in_path}")
        sys.exit(1)

    print(f"[Analyzer] Reading: {in_path}  (follow={args.follow}, from_start={args.from_start})")

    sent = Sentiment()
    per_user = defaultdict(lambda: {"pos": 0, "neu": 0, "neg": 0, "sum": 0.0, "n": 0})
    finals_seen = 0

    try:
        for rec in iter_jsonl(in_path, follow=args.follow, from_start=args.from_start):
            typ = rec.get("type")
            if typ not in ("final", "partial"):
                continue
            if typ == "partial" and not args.include_partials:
                continue

            text = (rec.get("text") or "").strip()
            if len(text) < args.minlen:
                continue

            lab, score, lang = sent.score(text, args.lang.lower() if args.lang else "auto")
            ts_iso = fmt_ts(rec.get("ts"))
            user = rec.get("user", "")

            tag = "FINAL" if typ == "final" else "PART "
            print(f"[{ts_iso}] ({lang}) {tag} {lab:>3} {score:+.3f} | {user}: {text}")

            if typ == "final":
                finals_seen += 1
                st = per_user[user]
                st[lab] += 1
                st["sum"] += score
                st["n"] += 1

                if args.summary_every and finals_seen % args.summary_every == 0:
                    print_user_summary(per_user)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("\n=== Final summary ===")
        print_user_summary(per_user)


if __name__ == "__main__":
    main()
