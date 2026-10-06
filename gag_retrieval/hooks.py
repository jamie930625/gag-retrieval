"""Line recognizability — the ranking key that makes the gag LAND.

A lexical match only works live if the audience instantly recognizes the
incoming line ("happy breakup, I wish you happiness" — everyone knows it by
beat one). All four
golden examples land on the B-song's most iconic line, so recognizability is
the primary sort key among equally-matching candidates:

    title line > chorus (repeated) line > opening line > any other line

All of it is computable from LRC structure alone — no LLM, no annotation.
An optional per-song popularity file scales the whole song (a perfect line of
a song nobody knows is still a dead gag).
"""

from __future__ import annotations

import json
from pathlib import Path

from .lrc import LineGroup, Song

W_TITLE = 1.00      # line contains the song title (or is the title)
W_CHORUS3 = 0.80    # sung >=3 times — almost surely the hook
W_CHORUS2 = 0.70    # sung twice
W_OPENING = 0.60    # the very first lyric line
W_PLAIN = 0.35      # any other line


def recognizability(song: Song, lg: LineGroup) -> tuple[float, str]:
    """(weight, label) for one line of a song."""
    w, label = W_PLAIN, "一般句"
    if lg.first_idx == 0:
        w, label = W_OPENING, "首句"
    if lg.repeats >= 3:
        w, label = max(w, W_CHORUS3), "副歌句"
    elif lg.repeats == 2:
        w, label = max(w, W_CHORUS2), "重複句"
    tn = song.title_norm
    if tn and len(tn) >= 2 and (tn in lg.norm or lg.norm in tn):
        w, label = W_TITLE, "歌名句"
    return w, label


# ─────────────────────────────────────────────────────────────────────────────
# Anchor identity — is this shared word an IDENTITY WORD of the song?
#
# The user's golden examples all anchor on a word that IS one song's identity
# ("love you" = B's title, "happy" ⊂ "Happy Breakup", "want to fly" = A's
# repeated hook), while junk anchors ("this moment", "what") are nobody's
# identity. This is the
# primary rank signal — anchor LENGTH is deliberately secondary.
# ─────────────────────────────────────────────────────────────────────────────

W_ID_TITLE = 1.00     # anchor ⊆ title AND covers >=1/3 of the title's content
W_ID_HOOK = 0.60      # WOVEN through the song: appears in >=2 DISTINCT lines
W_ID_PLAIN = 0.30     # ordinary content word

_TF_CACHE: dict[int, list[tuple[str, int]]] = {}   # id(Song) -> [(norm, repeats)]


def _song_lines(song: Song) -> list[tuple[str, int]]:
    key = id(song)
    if key not in _TF_CACHE:
        _TF_CACHE[key] = [(lg.norm, lg.repeats) for lg in song.lines]
    return _TF_CACHE[key]


def _song_tf(song: Song, span: str) -> int:
    return sum(norm.count(span) * reps for norm, reps in _song_lines(song))


def _song_spread(song: Song, span: str) -> int:
    """How many DISTINCT lines contain the span — the theme signal. "fly" woven
    through 9 different lines of "I Want to Fly" is the song's MEANING; "happy"
    looped 3x inside one single line of "Secretly" is just a passing word."""
    return sum(1 for norm, _ in _song_lines(song) if span in norm)


def _title_grade(span: str, title_norm: str) -> bool:
    """Substring alone is too loose for LONG titles: "give me" ⊆ "Marry Me
    Today" is a corner, not the identity. Require the anchor to cover >=1/3 of
    the title's CONTENT characters ("love you" covers "love" = 1/2 of the
    content chars of "Say Love You" -> yes; "give me" covers "give" = 1/5 of
    the content chars of "Marry Me Today" -> no, falls through to hook/plain)."""
    from .lexical import _STOP_CHARS
    if not title_norm or len(title_norm) < 2:
        return False
    if span not in title_norm and title_norm not in span:
        return False
    content = {ch for ch in title_norm if ch not in _STOP_CHARS}
    if not content:
        return False
    return len(set(span) & content) / len(content) >= 1 / 3


def identity_weight(span: str, song_a: Song, song_b: Song) -> tuple[float, str]:
    """(weight, label) — how strongly `span` identifies EITHER song. Hook grade
    requires SPREAD (>=2 distinct lines), not raw repetition — one chorus line
    looped 3x does not make its words the song's theme."""
    best, label = W_ID_PLAIN, "一般詞"
    for song, side in ((song_b, "B"), (song_a, "A")):
        if _title_grade(span, song.title_norm):
            return W_ID_TITLE, f"{side}歌名詞"
        if _song_spread(song, span) >= 2 and best < W_ID_HOOK:
            best, label = W_ID_HOOK, f"{side}主題詞"
    return best, label


def theme_profile(song: Song, k: int = 6) -> list[tuple[str, int, int]]:
    """The song's saturated meanings: top (word, spread, tf) by how widely the
    word is WOVEN through distinct lines. This is what the crowd's ear retains
    — fed to the judge so it can match SEMANTIC fields (fly ↔ wings/clouds), not
    just repeated strings."""
    from .lexical import _STOP_CHARS, _valid_anchor
    stats: dict[str, list[int]] = {}       # word -> [spread, tf]
    for norm, reps in _song_lines(song):
        core = norm.replace(" ", "")
        grams = set()
        for n in (2, 3):
            for i in range(len(core) - n + 1):
                g = core[i:i + n]
                if _valid_anchor(g):
                    grams.add(g)
        for ch in set(song.title_norm):    # title chars count as 1-char themes
            if ch not in _STOP_CHARS and not ch.isascii() and ch in core:
                grams.add(ch)
        for g in grams:
            st = stats.setdefault(g, [0, 0])
            st[0] += 1
            st[1] += core.count(g) * reps
    cands = [(g, s, t) for g, (s, t) in stats.items() if s >= 2]
    cands.sort(key=lambda x: (-x[1], -x[2], -len(x[0])))
    # drop substrings of an already-kept equal-or-wider word ("want to fly" ⊂ "I want to fly")
    keep: list[tuple[str, int, int]] = []
    for g, s, t in cands:
        if not any(g in kg and ks >= s for kg, ks, _ in keep):
            keep.append((g, s, t))
        if len(keep) >= k:
            break
    return keep


def load_popularity(path: str | Path | None) -> dict[str, float]:
    """Optional {song_id: 0..1} popularity map; missing songs default to 1.0."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return {str(k): float(v) for k, v in d.items()}
    except Exception as e:
        print(f"[hooks] popularity file unreadable ({e}); ignoring")
        return {}
