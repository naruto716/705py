# consolidate.py
"""
Real-time topic-patient consolidation for Stream/Deepgram JSONL,
and push each finalized paragraph to the frontend via WebSocket.

Console output (for debugging):
  user: <speaker_id>
  content: <paragraph>

WS payload (JSON):
  { "user": "<speaker_id>", "content": "<paragraph>", "call_id": "...",
    "start": <float>, "end": <float>, "duration": <float>, "emitted_at": <unix_ts> }

Start:
  uv run python consolidate.py --follow
"""

from __future__ import annotations
import argparse, json, re, sys, time, threading, asyncio
from pathlib import Path
from typing import Any, Dict, Tuple

# ---------------- WebSocket broadcaster ----------------
WS_HOST = "127.0.0.1"
WS_PORT = 8799

try:
    import websockets
except Exception:
    websockets = None  # will print a warning later

_WS_LOOP: asyncio.AbstractEventLoop | None = None
_WS_QUEUE: asyncio.Queue | None = None
_WS_CLIENTS: set = set()

async def _ws_handler(ws):
    _WS_CLIENTS.add(ws)
    try:
        async for _ in ws:  # ignore incoming messages
            pass
    finally:
        _WS_CLIENTS.discard(ws)

async def _ws_broadcaster():
    assert _WS_QUEUE is not None
    while True:
        msg = await _WS_QUEUE.get()
        dead = set()
        for ws in list(_WS_CLIENTS):
            try:
                await ws.send(msg)
            except Exception:
                dead.add(ws)
        for ws in dead:
            _WS_CLIENTS.discard(ws)

async def _ws_main():
    assert websockets is not None
    server = await websockets.serve(_ws_handler, WS_HOST, WS_PORT, ping_interval=20)
    print(f"[ws] listening on ws://{WS_HOST}:{WS_PORT}", file=sys.stderr)
    asyncio.create_task(_ws_broadcaster())
    await asyncio.Future()  # run forever

def start_ws_server():
    global _WS_LOOP, _WS_QUEUE
    if websockets is None:
        print("[ws] websockets not installed; frontend push disabled", file=sys.stderr)
        return
    _WS_LOOP = asyncio.new_event_loop()
    _WS_QUEUE = asyncio.Queue()
    t = threading.Thread(target=lambda: _WS_LOOP.run_until_complete(_ws_main()), daemon=True)
    t.start()

def ws_push(obj: Dict[str, Any]):
    # Enqueue JSON for broadcast (thread-safe)
    if _WS_LOOP is None or _WS_QUEUE is None:
        return
    payload = json.dumps(obj, ensure_ascii=False)
    asyncio.run_coroutine_threadsafe(_WS_QUEUE.put(payload), _WS_LOOP)

# ---------------- text utilities & polishing ----------------
FILLERS_RE = re.compile(r"\b(uh|umm|um|erm|er|ah|eh|hmm|you know|i mean|like)\b", re.I)
SPACES_RE = re.compile(r"\s+")
REPEAT_1_RE = re.compile(r"\b(\w+)(?:\s+\1){1,4}\b", re.I)
REPEAT_2_RE = re.compile(r"\b(\w+\s+\w+)(?:\s+\1){1,3}\b", re.I)

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
    parts = [t.strip() for t in re.split(r"(?<=[.!?…])\s+", text) if t.strip()]
    parts = [p[:1].upper() + p[1:] if p else p for p in parts]
    return " ".join(parts)

try:
    from deepmultilingualpunctuation import PunctuationModel
except Exception:
    PunctuationModel = None

class Punctuator:
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
        if cur: out.append(" ".join(cur))
        return ". ".join(out) + "."

    def punct(self, text: str) -> str:
        if not text: return text
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

# ---------------- similarity backend ----------------
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except Exception:
    np = None
    SentenceTransformer = None

try:
    from rapidfuzz.fuzz import token_set_ratio
except Exception:
    token_set_ratio = None

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
        if self.mode != "sbert": return None
        return self.model.encode([text], normalize_embeddings=True)[0]

    def sim(self, a, b, ta: str = "", tb: str = "") -> float:
        if self.mode == "sbert":
            if a is None or b is None: return 0.0
            cos = float(np.dot(a, b))
            return max(0.0, min(1.0, (cos + 1.0) / 2.0))
        if self.mode == "fuzzy":
            return float(token_set_ratio(ta, tb)) / 100.0 if token_set_ratio else 0.0
        return 0.0

# ---------------- JSONL helpers ----------------
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
                    time.sleep(tick); yield {"_tick": time.time()}; continue
                break
            line = line.strip()
            if not line: continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                if follow: time.sleep(tick); yield {"_tick": time.time()}; continue

def extract_times(rec: Dict[str, Any]) -> Tuple[float, float]:
    ts = float(rec.get("ts", 0.0))
    meta = rec.get("meta") or {}; words = meta.get("words") or []
    if words:
        try:
            return float(words[0]["start"]), float(words[-1]["end"])
        except Exception:
            pass
    return ts, ts

# ---------------- emitter (console + WS) ----------------
def emit(user: str, buf: Dict[str, Any], punct: Punctuator):
    text = polish(buf["text"], punct)
    # console view
    print(f"user: {user}\ncontent: {text}\n", flush=True)
    # WS payload
    payload = {
        "user": user,
        "content": text,
        "call_id": buf.get("call_id", ""),
        "start": buf["start"],
        "end": buf["end"],
        "duration": round(max(0.0, buf["end"] - buf["start"]), 3),
        "emitted_at": time.time(),
    }
    ws_push(payload)

# ---------------- main consolidation loop ----------------
def main():
    ap = argparse.ArgumentParser(description="Real-time consolidation to user/content + WebSocket push.")
    ap.add_argument("--file","-f",type=str,help="Path to JSONL (default: latest in ./transcripts)")
    ap.add_argument("--call-id",type=str,default=None,help="Filter by call id")

    # continuity & boundaries
    ap.add_argument("--gap",type=float,default=3.5,help="Soft gap to allow merging (s)")
    ap.add_argument("--hard-gap",type=float,default=7.0,help="Hard boundary (s)")
    ap.add_argument("--idle-sec",type=float,default=6.0,help="Emit on speaker pause (s)")
    ap.add_argument("--break-patience",type=int,default=3,help="Consecutive low-sim to break")

    # similarity thresholds
    ap.add_argument("--sim-threshold",type=float,default=0.55,help="Similarity to MERGE (0..1)")
    ap.add_argument("--break-threshold",type=float,default=0.35,help="Similarity counted LOW (0..1)")

    # readiness + debounce (higher => fewer/larger)
    ap.add_argument("--min-duration",type=float,default=20.0,help="Soft target duration (s)")
    ap.add_argument("--min-chars",type=int,default=420,help="Soft target length")
    ap.add_argument("--min-segments",type=int,default=10,help="Soft target segments")
    ap.add_argument("--debounce-sec",type=float,default=3.5,help="Wait after ready; extend if more merges")

    # safety caps
    ap.add_argument("--hard-max-duration",type=float,default=90.0,help="Force emit (s)")
    ap.add_argument("--hard-max-chars",type=int,default=2000,help="Force emit (chars)")

    # modes
    ap.add_argument("--follow","-F",action="store_true",help="Follow file (tail -f)")
    ap.add_argument("--from-start",action="store_true",help="Read whole file then follow")
    ap.add_argument("--model",type=str,default="sentence-transformers/all-MiniLM-L6-v2",help="SBERT model name")
    args = ap.parse_args()

    path = Path(args.file) if args.file else latest_jsonl(Path("transcripts"))
    if not path.exists():
        print(f"File not found: {path}"); sys.exit(1)

    # start WS server in background
    start_ws_server()

    # similarity + punctuation
    sim = Similarity(args.model)
    punct = Punctuator()

    buffers: Dict[str, Dict[str, Any]] = {}

    try:
        for rec in iter_jsonl_with_ticks(path, follow=True, from_start=args.from_start, tick=0.25):
            # heartbeat: debounce / idle
            if "_tick" in rec:
                now = rec["_tick"]
                for u, buf in list(buffers.items()):
                    if buf.get("ready_at") and now >= buf["ready_at"]:
                        emit(u, buf, punct); del buffers[u]; continue
                    if (now - buf["updated_at"]) >= args.idle_sec and not buf.get("ready_at"):
                        emit(u, buf, punct); del buffers[u]
                continue

            if rec.get("type") != "final": continue
            if args.call_id and rec.get("call_id") != args.call_id: continue

            user = rec.get("user") or "unknown"
            raw = rec.get("text") or ""
            text = normalize(raw)
            if not text: continue

            start, end = extract_times(rec)
            now = time.time()
            cid = rec.get("call_id","")

            # flush other users far behind
            for ou, obuf in list(buffers.items()):
                if ou == user: continue
                if start - float(obuf["end"]) > args.hard_gap:
                    emit(ou, obuf, punct); del buffers[ou]

            buf = buffers.get(user)
            if buf is None:
                buffers[user] = {
                    "text": text, "start": start, "end": end,
                    "last_text": text, "last_emb": sim.encode(text) if sim.mode=="sbert" else None,
                    "mean_emb": None, "segments": 1, "updated_at": now,
                    "call_id": cid, "ready_at": None, "low_sim_streak": 0,
                }
                if (end - start) >= args.min_duration or len(text) >= args.min_chars:
                    buffers[user]["ready_at"] = now + args.debounce_sec
                continue

            # hard gap boundary
            if start - float(buf["end"]) > args.hard_gap:
                emit(user, buf, punct); del buffers[user]
                buffers[user] = {
                    "text": text, "start": start, "end": end,
                    "last_text": text, "last_emb": sim.encode(text) if sim.mode=="sbert" else None,
                    "mean_emb": None, "segments": 1, "updated_at": now,
                    "call_id": cid, "ready_at": None, "low_sim_streak": 0,
                }
                continue

            # similarity
            merge, low_sim = False, False
            cur_emb = None
            if sim.mode == "sbert":
                cur_emb = sim.encode(text)
                from numpy import dot
                def _cos(a,b): 
                    if a is None or b is None: return 0.0
                    import numpy as _np
                    return float(max(0.0, min(1.0, (dot(a,b)+1.0)/2.0)))
                s_last = _cos(buf["last_emb"], cur_emb)
                s_mean = s_last
                score = max(s_last, s_mean)
                if score >= args.sim_threshold: merge = True
                if score <= args.break_threshold: low_sim = True
            elif token_set_ratio:
                score = float(token_set_ratio(buf["last_text"], text))/100.0
                if score >= args.sim_threshold or ends_cont(buf["last_text"]) or starts_cont(text):
                    merge = True
                if score <= args.break_threshold: low_sim = True
            else:
                if ends_cont(buf["last_text"]) or starts_cont(text):
                    merge = True

            if low_sim:
                buf["low_sim_streak"] += 1
            else:
                buf["low_sim_streak"] = 0

            if merge or (not low_sim and (start - float(buf["end"])) <= args.gap):
                buf["text"] = (buf["text"] + " " + text).strip()
                buf["end"] = end
                buf["last_text"] = text
                buf["segments"] += 1
                buf["updated_at"] = now
                if sim.mode == "sbert":
                    buf["last_emb"] = cur_emb

                if ((buf["end"] - buf["start"]) >= args.min_duration or
                    len(buf["text"]) >= args.min_chars or
                    buf["segments"] >= args.min_segments):
                    buf["ready_at"] = now + args.debounce_sec

                if ((buf["end"] - buf["start"]) >= args.hard_max_duration or
                    len(buf["text"]) >= args.hard_max_chars):
                    emit(user, buf, punct); del buffers[user]
            else:
                if buf["low_sim_streak"] >= args.break_patience:
                    emit(user, buf, punct); del buffers[user]
                    buffers[user] = {
                        "text": text, "start": start, "end": end,
                        "last_text": text, "last_emb": sim.encode(text) if sim.mode=="sbert" else None,
                        "mean_emb": None, "segments": 1, "updated_at": now,
                        "call_id": cid, "ready_at": None, "low_sim_streak": 0,
                    }
                else:
                    # be patient: still accumulate
                    buf["text"] = (buf["text"] + " " + text).strip()
                    buf["end"] = end
                    buf["last_text"] = text
                    buf["segments"] += 1
                    buf["updated_at"] = now
                    if buf.get("ready_at"):
                        buf["ready_at"] = now + args.debounce_sec

    except KeyboardInterrupt:
        pass
    finally:
        for u, buf in list(buffers.items()):
            emit(u, buf, Punctuator())

if __name__ == "__main__":
    main()
