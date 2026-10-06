"""Agent tool surface — the same engine the demo uses, as plain functions.

Two decision layers, two functions:

  * ``suggest_next_song`` — SONG SELECTION. Lives in the orchestrator, OUTSIDE
    the transition agent: the agent never chooses songs, so this stage is
    structurally impossible to skip.
  * ``lyric_hooks`` — CUE SELECTION within a fixed (A, B) pair. Designed to be
    INJECTED into the transition agent's seed prompt (data in the prompt
    cannot be skipped), or registered as an inspect-tool gated the same way
    omni_agent gates builds behind structure_overview.

Both return JSON-ready dicts: every option carries the gag evidence (anchor,
type, saturation) AND the honest feasibility metadata (times, runway, section
position) — the tool never filters on feasibility; weighing gag vs acoustics
is the agent's job.

Example (per-run, stateless):
    from gag_retrieval.tool import GagCatalog
    cat = GagCatalog("lrc_test")
    cat.suggest_next_song("<song A>", k=5)
    cat.lyric_hooks("<song A>", "<song B>", k=5)
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from .hooks import load_popularity, theme_profile
from .lexical import LexicalIndex
from .lrc import Song, load_corpus
from .meta import line_meta, vocal_end
from .retrieve import SongResult, retrieve


class GagCatalog:
    """Loaded corpus + index. Build once per catalog, query per transition."""

    def __init__(self, corpus_dir: str | Path, popularity_json: str | None = None):
        self.songs = load_corpus(corpus_dir)
        self.index = LexicalIndex(self.songs)
        self.by_id = {s.song_id: s for s in self.songs}
        self.popularity = load_popularity(popularity_json)

    # ── tool 1: song selection (orchestrator layer) ─────────────────────────
    def suggest_next_song(self, current_song: str, k: int = 10,
                          exclude: set[str] | None = None) -> dict:
        query = self.by_id.get(current_song)
        if query is None:
            return {"error": f"unknown song '{current_song}'",
                    "available": sorted(self.by_id)}
        results = retrieve(query, self.index, topk_songs=k,
                           popularity=self.popularity,
                           exclude_ids=exclude or set())
        return {
            "current_song": current_song,
            "current_themes": [
                {"word": w, "n_lines": s, "n_sung": t}
                for w, s, t in theme_profile(query)],
            "candidates": [self._song_payload(query, r) for r in results],
            "note": ("gag_score只評歌詞層面；聲學可行性（BPM/key/mashability）"
                     "由呼叫方另行評估後綜合決策。"),
        }

    # ── tool 2: cue selection within a fixed pair (agent layer) ─────────────
    def lyric_hooks(self, song_a: str, song_b: str, k: int = 5) -> dict:
        query = self.by_id.get(song_a)
        target = self.by_id.get(song_b)
        if query is None or target is None:
            return {"error": "unknown song",
                    "available": sorted(self.by_id)}
        exclude = {s for s in self.by_id if s != song_b}
        results = retrieve(query, self.index, topk_songs=1,
                           popularity=self.popularity, max_evidence=k,
                           exclude_ids=exclude)
        if not results:
            return {"song_a": song_a, "song_b": song_b, "options": [],
                    "note": "兩首歌之間沒有字面對應的接點。"}
        payload = self._song_payload(query, results[0])
        return {
            "song_a": song_a, "song_b": song_b,
            "a_themes": [{"word": w, "n_lines": s, "n_sung": t}
                         for w, s, t in theme_profile(query)],
            "b_themes": [{"word": w, "n_lines": s, "n_sung": t}
                         for w, s, t in theme_profile(target)],
            "options": payload["options"],
            "hint": ("每個 option 的 line_a.end / line_b.start 即 a_out/b_in "
                     "cue 候選；用 mashability 評各選項窗口、親耳聽過再選。"
                     "有梗但難混的選項可考慮 generate_bridge。"),
        }

    # ── shared payload shaping ───────────────────────────────────────────────
    def _song_payload(self, query: Song, r: SongResult) -> dict:
        song_b = self.by_id[r.song_id]
        options = []
        for ev in r.evidence:
            d = dataclasses.asdict(ev)
            a_meta = ({} if ev.a_text.startswith("《")
                      else line_meta(query, ev.a_time))
            b_meta = {} if ev.b_is_title else line_meta(song_b, ev.b_time)
            options.append({
                "anchor": ev.anchor,
                "anchor_label": ev.anchor_label,
                "gag_score": ev.score,
                "a_saturation": {"n_lines": ev.a_spread, "n_sung": ev.a_sat},
                "line_a": {"text": ev.a_text, "start": ev.a_time,
                           "end": a_meta.get("line_end"),
                           **{k: a_meta[k] for k in
                              ("section_pos", "to_section_end") if k in a_meta}},
                "line_b": {"text": ev.b_text, "start": ev.b_time,
                           "end": b_meta.get("line_end"),
                           "recognizability": ev.b_label,
                           **{k: b_meta[k] for k in
                              ("section_pos", "runway", "pos_pct") if k in b_meta}},
                "all_anchors": d["anchors"],
            })
        return {"song": r.song_id, "gag_score": r.score,
                "vocal_end": round(vocal_end(song_b), 1), "options": options}
