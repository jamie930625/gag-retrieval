"""Demo web server — stdlib http.server only (no FastAPI, runs anywhere).

    python3 -m gag_retrieval.server --corpus lrc_test --port 8010
    # open http://127.0.0.1:8010

Endpoints (deliberately shaped like the future agent tools):
    GET /api/songs                          corpus browser
    GET /api/suggest?song=X&k=8&exclude=A,B stage 1 — the `suggest_next_song` tool
    GET /api/judge?song=X&k=8&exclude=A,B   stage 2 — live Gemini verdict
    GET /audio/<song_id>                    mp3 with HTTP Range (for seeking)

Gemini key: env GEMINI_API_KEY, or a gitignored file gag_retrieval/gemini.key.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sys
import traceback
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "gag_retrieval"

from . import judge as jd
from .lexical import LexicalIndex
from .lrc import Song, load_corpus
from .meta import line_meta, vocal_end
from .retrieve import retrieve
from .zhnorm import normalize

AUDIO_EXTS = (".mp3", ".m4a", ".wav", ".flac", ".ogg")
MIME = {".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".wav": "audio/wav",
        ".flac": "audio/flac", ".ogg": "audio/ogg"}


# ─────────────────────────────────────────────────────────────────────────────
# App state
# ─────────────────────────────────────────────────────────────────────────────

class App:
    def __init__(self, corpus_dir: str):
        self.corpus_dir = Path(corpus_dir)
        self.songs: list[Song] = load_corpus(corpus_dir)
        self.index = LexicalIndex(self.songs)
        self.by_id = {s.song_id: s for s in self.songs}
        self.audio = self._map_audio()
        self.offsets = self._load_offsets()   # song_id -> sec (audio = lrc + off)

    def _load_offsets(self) -> dict[str, float]:
        f = self.corpus_dir / "offsets.json"
        try:
            return {str(k): float(v)
                    for k, v in json.loads(f.read_text()).items()}
        except Exception:
            return {}

    def save_offset(self, song_id: str, offset: float):
        self.offsets[song_id] = round(float(offset), 2)
        (self.corpus_dir / "offsets.json").write_text(
            json.dumps(self.offsets, ensure_ascii=False, indent=2),
            encoding="utf-8")

    def cal_line(self, song: Song) -> dict:
        """The song's most recognizable line — what the user calibrates against."""
        if not song.lines:
            return {"text": "", "time": 0.0}
        for lg in song.lines:
            if song.title_norm and song.title_norm in lg.norm:
                return {"text": lg.text, "time": lg.times[0]}
        for lg in song.lines:
            if lg.repeats >= 3:
                return {"text": lg.text, "time": lg.times[0]}
        lg = song.lines[0]
        return {"text": lg.text, "time": lg.times[0]}

    def _map_audio(self) -> dict[str, Path]:
        """Fuzzy-match messy audio filenames to songs: the song whose normalized
        title appears in the normalized filename wins; longest title first so
        《說愛你》 claims its file before 《愛你》 can."""
        files = [f for f in self.corpus_dir.iterdir()
                 if f.suffix.lower() in AUDIO_EXTS]
        out: dict[str, Path] = {}
        for song in sorted(self.songs, key=lambda s: len(s.title_norm),
                           reverse=True):
            for f in files:
                if f in out.values():
                    continue
                if song.title_norm and song.title_norm in normalize(f.stem):
                    out[song.song_id] = f
                    break
        for s in self.songs:
            if s.song_id not in out:
                print(f"[server] ⚠ no audio matched for 《{s.title}》")
        return out

    def gemini_key(self) -> str:
        key = os.environ.get("GEMINI_API_KEY", "")
        if key:
            return key
        f = Path(__file__).parent / "gemini.key"
        return f.read_text().strip() if f.exists() else ""


APP: App | None = None


# ─────────────────────────────────────────────────────────────────────────────
# API logic
# ─────────────────────────────────────────────────────────────────────────────

def _enrich(ev, query: Song, song_b: Song) -> dict:
    d = dataclasses.asdict(ev)
    d["a_meta"] = line_meta(query, ev.a_time) if not ev.a_text.startswith("《") else {}
    d["b_meta"] = line_meta(song_b, ev.b_time) if not ev.b_is_title else {}
    return d


def api_suggest(q: dict) -> dict:
    song_id = q.get("song", "")
    query = APP.by_id.get(song_id)
    if query is None:
        return {"error": f"unknown song '{song_id}'"}
    k = int(q.get("k", "8"))
    exclude = {x for x in q.get("exclude", "").split(",") if x}
    results = retrieve(query, APP.index, topk_songs=k, exclude_ids=exclude)
    return {
        "query": song_id,
        "results": [{
            "song_id": r.song_id,
            "score": r.score,
            "vocal_end": vocal_end(APP.by_id[r.song_id]),
            "evidence": [_enrich(e, query, APP.by_id[r.song_id])
                         for e in r.evidence],
        } for r in results],
    }


_TIME_RE = re.compile(r"(\d+):(\d+(?:\.\d+)?)")


def _parse_mmss(s: str) -> float | None:
    m = _TIME_RE.search(str(s or ""))
    return int(m.group(1)) * 60 + float(m.group(2)) if m else None


def _resolve_song(name: str) -> str | None:
    """Map the judge's `next_song` (may carry 《》/spacing) to a corpus id.

    EXACT title match wins first; only then substring matches, longest title
    first — otherwise 《愛你》 hijacks 《說愛你》 ("爱你" ⊆ "说爱你") and the
    mix preview plays the wrong song's audio."""
    n = normalize(str(name or ""))
    if not n:
        return None
    for s in APP.songs:
        if s.title_norm == n:
            return s.song_id
    subs = [s for s in APP.songs
            if s.title_norm and (s.title_norm in n or n in s.title_norm)]
    if subs:
        return max(subs, key=lambda s: len(s.title_norm)).song_id
    return None


def api_judge(q: dict) -> dict:
    song_id = q.get("song", "")
    query = APP.by_id.get(song_id)
    if query is None:
        return {"error": f"unknown song '{song_id}'"}
    key = APP.gemini_key()
    if not key:
        return {"error": "GEMINI_API_KEY not set (env or gag_retrieval/gemini.key)"}
    os.environ["GEMINI_API_KEY"] = key
    k = int(q.get("k", "8"))
    exclude = {x for x in q.get("exclude", "").split(",") if x}
    results = retrieve(query, APP.index, topk_songs=k, exclude_ids=exclude)
    if not results:
        return {"error": "第一階段沒有任何候選"}
    prompt = jd.build_prompt(query, results, APP.index.songs)
    try:
        raw, parsed = jd.run_judge(prompt, "gemini")
    except Exception as e:
        msg = str(e)
        if "過載" in msg or "503" in msg or "429" in msg:
            msg = ("Gemini 免費層暫時過載（已自動重試多次仍失敗）。"
                   "請幾秒後再按一次；連續多首時把速度放慢一點即可。")
        return {"error": f"Stage-2 裁判呼叫失敗：{msg}"}
    if not parsed:
        return {"error": "回覆解析失敗", "raw": (raw or "")[:1000]}
    # resolve times/song ids so the UI can wire ▶ buttons
    b_id = _resolve_song(parsed.get("next_song"))
    parsed["_next_song_id"] = b_id
    fl, el = parsed.get("from_line") or {}, parsed.get("enter_line") or {}
    fl["sec"] = _parse_mmss(fl.get("time"))
    el["sec"] = _parse_mmss(el.get("time"))
    parsed["from_line"], parsed["enter_line"] = fl, el
    if fl.get("sec") is not None:
        # a_meta.line_end = when A's exit line finishes = the crossfade cut point
        parsed["_a_meta"] = line_meta(query, fl["sec"])
    if b_id and el.get("sec") is not None:
        parsed["_b_meta"] = line_meta(APP.by_id[b_id], el["sec"])
    for ru in parsed.get("runner_ups") or []:
        ru["_song_id"] = _resolve_song(ru.get("song"))
    return {"query": song_id, "judge": parsed}


def api_songs(_q: dict) -> dict:
    return {"songs": [{
        "id": s.song_id, "title": s.title, "n_lines": len(s.lines),
        "vocal_end": round(vocal_end(s), 1),
        "has_audio": s.song_id in APP.audio,
        "offset": APP.offsets.get(s.song_id, 0.0),
        "cal": APP.cal_line(s),
    } for s in APP.songs]}


def api_set_offset(q: dict) -> dict:
    song_id = q.get("song", "")
    if song_id not in APP.by_id:
        return {"error": f"unknown song '{song_id}'"}
    try:
        off = float(q.get("offset", "0"))
    except ValueError:
        return {"error": "bad offset"}
    APP.save_offset(song_id, off)
    return {"ok": True, "song": song_id, "offset": APP.offsets[song_id]}


# ─────────────────────────────────────────────────────────────────────────────
# HTTP plumbing
# ─────────────────────────────────────────────────────────────────────────────

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        if "/audio/" not in str(args[0] if args else ""):
            super().log_message(fmt, *args)

    def _json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path, ctype: str):
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _audio(self, song_id: str):
        path = APP.audio.get(song_id)
        if path is None or not path.exists():
            self._json({"error": "no audio"}, 404)
            return
        size = path.stat().st_size
        ctype = MIME.get(path.suffix.lower(), "application/octet-stream")
        rng = self.headers.get("Range")
        start, end = 0, size - 1
        if rng:                                   # "bytes=start-end"
            m = re.match(r"bytes=(\d*)-(\d*)", rng)
            if m:
                if m.group(1):
                    start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
                elif not m.group(1):
                    start = 0
        length = end - start + 1
        with open(path, "rb") as fh:
            fh.seek(start)
            data = fh.read(length)
        self.send_response(206 if rng else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_HEAD(self):
        # browsers may probe audio with HEAD before ranged GETs
        route = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        if route.startswith("/audio/"):
            path = APP.audio.get(route[len("/audio/"):])
            if path is None or not path.exists():
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type",
                             MIME.get(path.suffix.lower(), "application/octet-stream"))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_GET(self):
        try:
            parsed = urllib.parse.urlparse(self.path)
            q = {k: v[0] for k, v in urllib.parse.parse_qs(parsed.query).items()}
            route = urllib.parse.unquote(parsed.path)
            if route == "/" or route == "/index.html":
                self._file(Path(__file__).parent / "demo.html",
                           "text/html; charset=utf-8")
            elif route == "/api/songs":
                self._json(api_songs(q))
            elif route == "/api/suggest":
                self._json(api_suggest(q))
            elif route == "/api/judge":
                self._json(api_judge(q))
            elif route == "/api/set_offset":
                self._json(api_set_offset(q))
            elif route.startswith("/audio/"):
                self._audio(route[len("/audio/"):])
            else:
                self._json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            except Exception:
                pass


def main():
    global APP
    p = argparse.ArgumentParser(description="有梗接歌 demo server")
    p.add_argument("--corpus", required=True)
    p.add_argument("--port", type=int, default=8010)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    APP = App(args.corpus)
    print(f"[server] {len(APP.songs)} songs, "
          f"{sum(1 for _ in APP.audio)} with audio, "
          f"gemini_key={'yes' if APP.gemini_key() else 'NO'}")
    try:
        srv = ThreadingHTTPServer((args.host, args.port), Handler)
    except OSError as e:
        sys.exit(f"[server] 埠 {args.port} 已被佔用（{e}）。先殺掉舊的："
                 f"pkill -f gag_retrieval.server，或改用 --port {args.port + 1}")
    print(f"[server] http://{args.host}:{args.port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
