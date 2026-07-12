"""Stage 1 — line-level retrieval: given the playing song A, rank candidate
next songs by their best (line_a ↔ line_b) anchor.

Stage 1's job is RECALL + rough ordering; taste lives in stage 2. The rough
ordering follows ONE principle: a shared word only ranks if it is an IDENTITY
WORD of one of the songs —

    strength   = identity(anchor)            歌名詞 1.0 > hook詞 0.6 > 一般詞 0.3
                 × mild length factor        (2字 0.9 ... 4字+ 1.0)
                 × position bonus            anchor at A's line END (last thing
                                             heard) / B's line edge
                 + near-duplicate bonus      同句反轉 shape
    pair_score = strength × recognizability(B line) × popularity(B)

「愛你」(B's title, line-final both sides) must outrank 「这一刻」 (nobody's
identity) regardless of anchor length — that inversion was the v1 bug.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .hooks import _song_spread, _song_tf, identity_weight, recognizability
from .lexical import _STOP_CHARS, LexicalIndex, is_noise_line, match_lines
from .lrc import Song


@dataclass
class Evidence:
    a_text: str
    a_time: float
    b_text: str
    b_time: float
    b_label: str              # 歌名句/副歌句/重複句/首句/一般句
    b_is_title: bool          # matched B's title pseudo-line (not a lyric line)
    anchor: str               # the shared word this pair is scored on
    anchor_label: str         # A/B歌名詞 | A/Bhook詞 | 一般詞 | 同句反轉
    anchors: list[str]        # all shared anchors (evidence for the judge)
    strength: float
    recog: float
    score: float
    channel: str = "lexical"
    a_sat: int = 1            # times A sings the anchor over the WHOLE song
    a_spread: int = 1         # DISTINCT A lines containing the anchor — the
                              # saturated-MEANING signal (飞 woven through 9
                              # lines is A's theme; 快乐 looped in 1 line isn't)


@dataclass
class SongResult:
    song_id: str
    score: float
    evidence: list[Evidence] = field(default_factory=list)   # best first


# Calibrated on real junk vs real gold: 「面前/在上/近我」 fragments land at
# ~0.08-0.10 (kill), while a plain-word gem like 「愿意」 lands at ~0.14 (must
# reach the judge). Ranking cannot tell 我说 from 愿意 — only the judge can.
MIN_PAIR = 0.12   # pairs below this are noise — never shown, never judged
MIN_SONG = 0.15   # a candidate song needs at least one pair this strong


def _length_factor(span: str) -> float:
    return min(1.0, 0.85 + 0.05 * (len(span) - 2))           # 2字0.85, 5字+1.0


def _position_bonus(anchor: str, ca: str, cb: str) -> float:
    """Timing physics of a live gag: the echo must land the INSTANT B enters.
    Anchor within B's first 3 chars = instant payoff (「分手快乐」 counts —
    the crowd hears 快乐 within a second). Anchor at A's line end = freshest
    in the ear. Both = true anadiplosis. Anchor buried at B's line END = the
    echo arrives 8 words late (near zero)."""
    ia, ib = ca.rfind(anchor), cb.find(anchor)
    a_end = ia >= 0 and ia + len(anchor) >= len(ca) - 1
    b_front = 0 <= ib <= 2
    bonus = 1.0 + (0.15 if a_end else 0.0) + (0.25 if b_front else 0.0)
    if a_end and b_front:
        bonus += 0.15
    elif ib > 2 and cb.endswith(anchor):
        bonus += 0.03
    return bonus


def _score_anchors(norm_a: str, norm_b: str, song_a: Song,
                   song_b: Song) -> tuple[float, str, str, list[str]] | None:
    """Best anchor of a line pair -> (strength, anchor, label, all_anchors)."""
    m = match_lines(norm_a, norm_b)
    ca, cb = norm_a.replace(" ", ""), norm_b.replace(" ", "")
    best, b_anchor, b_label = 0.0, "", ""
    anchors_all: list[str] = list(m.anchors) if m else []
    if m:
        for span in m.anchors:
            ident, label = identity_weight(span, song_a, song_b)
            s = ident * _length_factor(span) * _position_bonus(span, ca, cb)
            if s > best:
                best, b_anchor, b_label = s, span, label
    # 頂真接龍: B's first char echoes A's tail WORD (「…说爱你」→「爱就是有我
    # 常烦着你」). A single char is only PERCEIVABLE when it is a TITLE-grade
    # identity char (爱 ⊆《愛你》, 飛 ⊆《我要飛》) — generic chars (心/会/表)
    # flood the ranks with echoes no crowd would ever register.
    head = cb[0] if cb else ""
    if head and not head.isascii() and head not in _STOP_CHARS \
            and head in ca[-3:]:
        ident, label = identity_weight(head, song_a, song_b)
        if "歌名詞" in label:
            exact_tail = ca.endswith(head)
            s = ident * 0.8 * (1.55 if exact_tail else 1.45)
            if s > best:
                best, b_anchor, b_label = s, head, f"接龍·{label}"
            if head not in anchors_all:
                anchors_all.append(head)
    if m and m.near_dup > 0:
        nd = 0.5 + 0.5 * m.near_dup                 # 反轉句自身就是強對應
        if nd > best:
            best, b_anchor = nd, (b_anchor or "(近乎同句)")
            b_label = "同句反轉"
        else:
            best = min(1.0, best + 0.10 * m.near_dup)
    if best <= 0:
        return None
    return min(1.0, best), b_anchor, b_label, anchors_all


def _query_lines(song: Song) -> list[tuple[str, str, float]]:
    """(display, norm, time) for each of A's line groups + its title."""
    out = [(lg.text, lg.norm, lg.times[0]) for lg in song.lines
           if not is_noise_line(lg.norm)]
    if song.title_norm and len(song.title_norm) >= 2:
        out.append((f"《{song.title}》(歌名)", song.title_norm, 0.0))
    return out


def retrieve(query: Song, index: LexicalIndex, *, topk_songs: int = 10,
             popularity: dict[str, float] | None = None,
             semantic=None, max_evidence: int = 5,
             exclude_ids: set[str] | None = None) -> list[SongResult]:
    popularity = popularity or {}
    exclude = {query.song_id} | (exclude_ids or set())
    best: dict[tuple[str, str, str], Evidence] = {}   # (song, a_norm, b_norm)

    queries = _query_lines(query)
    for a_text, a_norm, a_time in queries:
        for song_id, b_norm in index.candidates(a_norm, query.song_id):
            if song_id in exclude or is_noise_line(b_norm):
                continue
            song_b = index.songs[song_id]
            scored = _score_anchors(a_norm, b_norm, query, song_b)
            if scored is None:
                continue
            strength, anchor, anchor_label, anchors = scored
            if strength < 0.2:
                continue
            lg = song_b.line_by_norm(b_norm)
            if lg is not None:
                recog, label = recognizability(song_b, lg)
                b_text, b_time, b_is_title = lg.text, lg.times[0], False
            else:                                   # title pseudo-line
                recog, label = 1.0, "歌名"
                b_text, b_time, b_is_title = f"《{song_b.title}》(歌名)", 0.0, True
            # the gag needs BOTH ends heard — but LRC repetition is a weak
            # proxy for A-side fame (宣誓句 sung once is still iconic), so
            # this factor stays GENTLE or it buries gold like 「愿意」.
            lg_a = query.line_by_norm(a_norm)
            recog_a = recognizability(query, lg_a)[0] if lg_a else 1.0
            # saturated-MEANING factor: what matters is the anchor being WOVEN
            # through many distinct A lines (the crowd's retained meaning),
            # not one chorus line looped — spread drives the boost, raw
            # repetition of a single line adds almost nothing.
            plain = anchor and "(" not in anchor
            sat = _song_tf(query, anchor) if plain else 1
            spread = _song_spread(query, anchor) if plain else 1
            sat_factor = (1.0 + 0.10 * min(max(spread - 1, 0), 4)
                          + (0.05 if spread == 1 and sat >= 3 else 0.0))
            pop = popularity.get(song_id, 1.0)
            ev = Evidence(a_text=a_text, a_time=a_time, b_text=b_text,
                          b_time=b_time, b_label=label, b_is_title=b_is_title,
                          anchor=anchor, anchor_label=anchor_label,
                          anchors=anchors, strength=round(strength, 3),
                          recog=recog, a_sat=sat, a_spread=spread,
                          score=round(strength * recog * sat_factor
                                      * (0.85 + 0.15 * recog_a) * pop, 4))
            key = (song_id, a_norm, b_norm)
            if key not in best or ev.score > best[key].score:
                best[key] = ev

    # optional semantic supplement over lines that lexical didn't already pair
    if semantic is not None and getattr(semantic, "available", False):
        cand_lines = [(s.song_id, lg) for s in index.songs.values()
                      if s.song_id not in exclude for lg in s.lines]
        hits = semantic.line_hits([q[1] for q in queries],
                                  [lg.text for _, lg in cand_lines])
        for qi, cj, strength in hits:
            a_text, a_norm, a_time = queries[qi]
            song_id, lg = cand_lines[cj]
            key = (song_id, a_norm, lg.norm)
            if key in best:
                continue
            song_b = index.songs[song_id]
            recog, label = recognizability(song_b, lg)
            pop = popularity.get(song_id, 1.0)
            best[key] = Evidence(
                a_text=a_text, a_time=a_time, b_text=lg.text, b_time=lg.times[0],
                b_label=label, b_is_title=False, anchor="(語意對應)",
                anchor_label="語意", anchors=[], channel="semantic",
                strength=round(strength, 3), recog=recog,
                score=round(strength * recog * pop, 4))

    # one row per (anchor, B line): three A lines echoing the same B hook is
    # ONE option, not three — keep only the strongest telling of it.
    dedup: dict[tuple[str, str, str], Evidence] = {}
    for (song_id, _, b_norm), ev in best.items():
        if ev.score < MIN_PAIR:
            continue
        key = (song_id, ev.anchor, b_norm)
        if key not in dedup or ev.score > dedup[key].score:
            dedup[key] = ev

    by_song: dict[str, list[Evidence]] = {}
    for (song_id, _, _), ev in dedup.items():
        by_song.setdefault(song_id, []).append(ev)

    results = []
    for song_id, evs in by_song.items():
        evs.sort(key=lambda e: e.score, reverse=True)
        if evs[0].score < MIN_SONG:
            continue                       # no real hook — don't offer the song
        extra = sum(0.05 for e in evs[1:4] if e.score >= 0.3)
        results.append(SongResult(song_id=song_id,
                                  score=round(evs[0].score + extra, 4),
                                  evidence=evs[:max_evidence]))
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:topk_songs]
