# consolidate.py
"""
Real-time topic-patient consolidation for Stream/Deepgram JSONL.

After semantic merging + light NLP polishing, prints exactly:
  user: <speaker_id>
  content: <one coherent paragraph>

Fewer, larger paragraphs:
- Debounced readiness (min duration/length/segments).
- Topic patience: require K consecutive low-similarity fragments before breaking.
- Hard boundaries: long gaps, long idle, hard caps.

Optional (improves results if available):
  uv add sentence-transformers numpy
  uv add deepmultilingualpunctuation
Otherwise falls back to fuzzy/heuristic logic.

Run:
  uv run python consolidate.py --follow
  # larger chunks:
  # uv run python consolidate.py --follow --min-duration 25 --min-chars 520 --min-segments 12 --debounce-sec 4 --idle-sec 7
"""

from __future__ import annotations
import argparse, json, re, sys, time
from pathlib import Path
from typing import Any, Dict, Tuple

# ---------- optional deps ----------
try:
    import numpy as np
except Exception:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from deepmultilingualpunctuation import PunctuationModel
except Exception:
    PunctuationModel = None

try:
    from rapidfuzz.fuzz import token_set_ratio
except Exception:
    token_set_ratio = None
# -----------------------------------

# ---------- text utils & polishing ----------
FILLERS_RE = re.compile(r"\b(uh|umm|um|erm|er|ah|eh|hmm|you know|i mean|like)\b", re.I)
SPACES_RE = re.compile(r"\s+")
REPEAT_1_RE = re.compile(r"\b(\w+)(?:\s+\1){1,4}\b", re.I)         # "word word"
REPEAT_2_RE = re.compile(r"\b(\w+\s+\w+)(?:\s+\1){1,3}\b", re.I)   # "make shape make shape"

LEADING_CONT = {"and","so","then","but","because","also"}
TRAILING_CONT = {"and","or","but","so","because","that","to","for","of","with","in","on","at","by","we","i","you","they"}

def normalize(s: str) -> str:
    s = s.strip()
    s = FILLERS_RE.sub("", s)
    s = REPEAT_1_RE.sub(r"\1", s)
    s = REPEAT_2_RE.sub(r"\1", s)
    s = SPACES_RE.sub(" ", s)
    return s.strip()

def starts_cont(s: str) -> bool:
    toks = s.lower().split()
    return bool(toks) and toks[0] in LEADING_CONT

def ends_cont(s: str) -> bool:
    toks = s.lower().split()
    return bool(toks) and toks[-1] in TRAILING_CONT

def smart_capitalize(text: str) -> str:
    text = re.sub(r"\bi\b", "I", text)
    # capitalize sentence initials
    parts = [t.strip() for t in re.split(r"(?<=[.!?…])\s+", text) if t.strip()]
    parts = [p[:1].upper() + p[1:] if p else p for p in parts]
    return " ".join(parts)

class Punctuator:
    """Use deepmultilingualpunctuation if available; otherwise simple heuristic."""
    def __init__(self):
        self.mode = "heuristic"
        self.model = None
        if PunctuationModel is not None:
            try:
                self.model = PunctuationModel()
                self.mode = "model"
            except Exception:
                self.model = None

    def heuristic(self, text: str) -> str:
        toks = text.split()
        out, cur = [], []
        for tok in toks:
            cur.append(tok)
            if (len(cur) >= 20) or (len(cur) >= 12 and tok.lower() in {"because","but","so"}):
                out.append(" ".join(cur)); cur = []
        if cur:
            out.append(" ".join(cur))
        return ". ".join(out) + "."

    def punct(self, text: str) -> str:
        if not text:
            return text
        if self.mode == "model":
            try:
                return self.model.restore_punctuation(text)
            except Exception:
                pass
        return self.heuristic(text)

def polish(text: str, punctuator: Punctuator) -> str:
    t = normalize(text)
    t = punctuator.punct(t)
    t = smart_capitalize(t)
    t = re.sub(r"\s+([?.!,…])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
# ---------------------------------------------

# ---------- similarity backend ----------
class Similarity:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.mode = "none"; self.model = None
        if SentenceTransformer is not None and np is not None:
            try:
                self.model = SentenceTransformer(model_name)
                self.mode = "sbert"
            except Exception:
                self.model = None
        if self.mode == "none" and token_set_ratio is not None:
            self.mode = "fuzzy"

    def encode(self, text: str):
        if self.mode != "sbert":
            return None
        return self.model.encode([text], normalize_embeddings=True)[0]

    def sim(self, a, b, ta: str = "", tb: str = "") -> float:
        if self.mode == "sbert":
            if a is None or b is None:
                return 0.0
            cos = float(np.dot(a, b))
            return max(0.0, min(1.0, (cos + 1.0) / 2.0))
        if self.mode == "fuzzy":
            return float(token_set_ratio(ta, tb)) / 100.0 if token_set_ratio else 0.0
        return 0.0
# ---------------------------------------

# ---------- JSONL helpers ----------
def latest_jsonl(trans_dir: Path) -> Path:
    cands = sorted(trans_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError("No transcripts in ./transcripts; use --file.")
    return cands[0]

def iter_jsonl_with_ticks(path: Path, follow: bool, from_start: bool, tick: float = 0.25):
    with path.open("r", encoding="utf-8") as f:
        if follow and not from_start:
            f.seek(0, 2)  # tail mode
        while True:
            line = f.readline()
            if not line:
                if follow:
                    time.sleep(tick)
                    yield {"_tick": time.time()}
                    continue
                break
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if follow:
                    time.sleep(tick)
                    yield {"_tick": time.time()}
                    continue

def extract_times(rec: Dict[str, Any]) -> Tuple[float, float]:
    ts = float(rec.get("ts", 0.0))
    meta = rec.get("meta") or {}
    words = meta.get("words") or []
    if words:
        try:
            return float(words[0]["start"]), float(words[-1]["end"])
        except Exception:
            pass
    return ts, ts
# -------------------------------------

# ---------- emitter ----------
def emit(user: str, buf: Dict[str, Any], punct: Punctuator):
    text = polish(buf["text"], punct)
    print(f"user: {user}\ncontent: {text}\n", flush=True)
# -------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Real-time consolidation to 'user/content' paragraphs.")
    ap.add_argument("--file","-f",type=str,help="Path to JSONL (default: latest in ./transcripts)")
    ap.add_argument("--call-id",type=str,default=None,help="Filter by call id")

    # continuity and boundaries
    ap.add_argument("--gap",type=float,default=3.5,help="Soft max seconds gap to allow merging")
    ap.add_argument("--hard-gap",type=float,default=7.0,help="Hard gap boundary (force paragraph end)")
    ap.add_argument("--idle-sec",type=float,default=6.0,help="Emit on speaker pause (no new fragments)")
    ap.add_argument("--break-patience",type=int,default=3,help="Consecutive low-sim needed to break topic")

    # similarity thresholds
    ap.add_argument("--sim-threshold",type=float,default=0.55,help="Similarity to MERGE (0..1)")
    ap.add_argument("--break-threshold",type=float,default=0.35,help="Similarity considered LOW (counts toward break)")

    # readiness + debounce (soft targets; higher => fewer/larger)
    ap.add_argument("--min-duration",type=float,default=20.0,help="Soft target duration before ready (s)")
    ap.add_argument("--min-chars",type=int,default=420,help="Soft target chars before ready")
    ap.add_argument("--min-segments",type=int,default=10,help="Soft target segments before ready")
    ap.add_argument("--debounce-sec",type=float,default=3.5,help="Wait after ready; extend if more merges come")

    # safety caps
    ap.add_argument("--hard-max-duration",type=float,default=90.0,help="Force emit at/after this duration (s)")
    ap.add_argument("--hard-max-chars",type=int,default=2000,help="Force emit at/after this many chars")

    # modes
    ap.add_argument("--follow","-F",action="store_true",help="Follow file (tail -f)")
    ap.add_argument("--from-start",action="store_true",help="Read whole file then follow")
    ap.add_argument("--model",type=str,default="sentence-transformers/all-MiniLM-L6-v2",help="SBERT model name")
    args = ap.parse_args()

    path = Path(args.file) if args.file else latest_jsonl(Path("transcripts"))
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    sim = Similarity(args.model)
    punct = Punctuator()

    # user -> buffer
    # fields: text,start,end,last_text,last_emb,mean_emb,segments,updated_at,call_id,ready_at,low_sim_streak
    buffers: Dict[str, Dict[str, Any]] = {}

    try:
        for rec in iter_jsonl_with_ticks(path, follow=True, from_start=args.from_start, tick=0.25):
            # heartbeat: debounce / idle / hard gap checks
            if "_tick" in rec:
                now = rec["_tick"]
                for u, buf in list(buffers.items()):
                    if buf.get("ready_at") and now >= buf["ready_at"]:
                        emit(u, buf, punct); del buffers[u]; continue
                    if (now - buf["updated_at"]) >= args.idle_sec and not buf.get("ready_at"):
                        emit(u, buf, punct); del buffers[u]
                continue

            if rec.get("type") != "final":
                continue
            if args.call_id and rec.get("call_id") != args.call_id:
                continue

            user = rec.get("user") or "unknown"
            text_raw = rec.get("text") or ""
            text = normalize(text_raw)
            if not text:
                continue

            start, end = extract_times(rec)
            now = time.time()
            cid = rec.get("call_id","")

            # flush other users far behind
            for ou, obuf in list(buffers.items()):
                if ou == user:
                    continue
                if start - float(obuf["end"]) > args.hard_gap:
                    emit(ou, obuf, punct); del buffers[ou]

            buf = buffers.get(user)
            if buf is None:
                last_emb = sim.encode(text) if sim.mode == "sbert" else None
                buffers[user] = {
                    "text": text, "start": start, "end": end,
                    "last_text": text, "last_emb": last_emb, "mean_emb": last_emb,
                    "segments": 1, "updated_at": now, "call_id": cid,
                    "ready_at": None, "low_sim_streak": 0,
                }
                # schedule if already long
                if (end - start) >= args.min_duration or len(text) >= args.min_chars:
                    buffers[user]["ready_at"] = now + args.debounce_sec
                continue

            # hard gap boundary
            if start - float(buf["end"]) > args.hard_gap:
                emit(user, buf, punct); del buffers[user]
                last_emb = sim.encode(text) if sim.mode == "sbert" else None
                buffers[user] = {
                    "text": text, "start": start, "end": end,
                    "last_text": text, "last_emb": last_emb, "mean_emb": last_emb,
                    "segments": 1, "updated_at": now, "call_id": cid,
                    "ready_at": None, "low_sim_streak": 0,
                }
                continue

            # similarity relation
            merge = False
            low_sim = False
            if sim.mode == "sbert":
                cur_emb = sim.encode(text)
                s_last = sim.sim(buf["last_emb"], cur_emb)
                s_mean = sim.sim(buf["mean_emb"], cur_emb) if buf["mean_emb"] is not None else s_last
                score = max(s_last, s_mean)
                if score >= args.sim_threshold:
                    merge = True
                if score <= args.break_threshold:
                    low_sim = True
            elif sim.mode == "fuzzy":
                score = float(token_set_ratio(buf["last_text"], text))/100.0 if token_set_ratio else 0.0
                if score >= args.sim_threshold or ends_cont(buf["last_text"]) or starts_cont(text):
                    merge = True
                if score <= args.break_threshold:
                    low_sim = True
            else:
                # heuristic-only
                if ends_cont(buf["last_text"]) or starts_cont(text):
                    merge = True
                low_sim = False
                cur_emb = None  # for type completeness

            # topic patience
            if low_sim:
                buf["low_sim_streak"] += 1
            else:
                buf["low_sim_streak"] = 0

            if merge or (not low_sim and (start - float(buf["end"])) <= args.gap):
                # merge
                buf["text"] = (buf["text"] + " " + text).strip()
                buf["end"] = end
                buf["last_text"] = text
                buf["segments"] += 1
                buf["updated_at"] = now
                if sim.mode == "sbert":
                    buf["last_emb"] = cur_emb
                    buf["mean_emb"] = cur_emb if buf["mean_emb"] is None else (buf["mean_emb"] + cur_emb) / 2.0

                # readiness -> debounce
                if ((buf["end"] - buf["start"]) >= args.min_duration or
                    len(buf["text"]) >= args.min_chars or
                    buf["segments"] >= args.min_segments):
                    buf["ready_at"] = now + args.debounce_sec

                # hard caps -> emit now
                if ((buf["end"] - buf["start"]) >= args.hard_max_duration or
                    len(buf["text"]) >= args.hard_max_chars):
                    emit(user, buf, punct); del buffers[user]
            else:
                # break only if patience exceeded; otherwise keep accumulating
                if buf["low_sim_streak"] >= args.break_patience:
                    emit(user, buf, punct); del buffers[user]
                    last_emb = sim.encode(text) if sim.mode == "sbert" else None
                    buffers[user] = {
                        "text": text, "start": start, "end": end,
                        "last_text": text, "last_emb": last_emb, "mean_emb": last_emb,
                        "segments": 1, "updated_at": now, "call_id": cid,
                        "ready_at": None, "low_sim_streak": 0,
                    }
                else:
                    # borderline: be patient (treat as continuation)
                    buf["text"] = (buf["text"] + " " + text).strip()
                    buf["end"] = end
                    buf["last_text"] = text
                    buf["segments"] += 1
                    buf["updated_at"] = now
                    if sim.mode == "sbert":
                        buf["last_emb"] = cur_emb
                        buf["mean_emb"] = cur_emb if buf["mean_emb"] is None else (buf["mean_emb"] + cur_emb) / 2.0
                    if buf.get("ready_at"):
                        buf["ready_at"] = now + args.debounce_sec

    except KeyboardInterrupt:
        pass
    finally:
        for u, buf in list(buffers.items()):
            emit(u, buf, punct)

if __name__ == "__main__":
    main()
