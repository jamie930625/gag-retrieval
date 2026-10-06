"""Lexical channel — character-bigram inverted index + anchor extraction.

"Lexical correspondence" is a string property, so this channel is the DEFINITION of the
task, not an approximation of it. The inverted index makes query cost depend on
posting-list sizes, not corpus size — the same structure scales from 6 songs to
a search-engine-sized catalog (swap the dict for Elasticsearch/Tantivy).

An ANCHOR is a contiguous word both lines share, cleanly aligned to word
boundaries: no grammatical particle inside (kills "de kan" (of-look) and
"toutou de" (secretly-of) — the clean core "toutou" (secretly) survives as its
own anchor) and at least one content character (kills pure pronoun glue
"wo de" (my), "ni shi" (you are)).

Anchor extraction here is deliberately identity-blind: WHICH anchor matters
(title word > hook word > ordinary word) needs song context and lives in retrieve.py. This
module only answers "what clean words do these two lines share".
"""

from __future__ import annotations

from dataclasses import dataclass

from .lrc import Song

# Grammatical particles: an anchor may not CONTAIN any of these ("de kan" is a
# segmentation artifact, never a word).
_PARTICLES = set("的了着著吧啊呢吗嗎嘛么麼哦喔哟喲耶欸唉哎啦呀嘿哈之乎者矣焉哉")

# Function characters (pronouns/copulas/etc.): an anchor made ONLY of these has
# no content — "ai ni" (love you) stands ("ai" is content), "wo ni" (I you),
# "jiu shi" (just is), "gei wo" (give me) do not.
_STOP_CHARS = _PARTICLES | set("我你他她它您们們这這那哪个個是就也都很再才在有"
                               "和跟与與及而或於于把被向往到从從对對又还還只没沒不给給一能要")

# Weak-semantics words: time/logic/degree scaffolding. Sharing one of these is
# not a PERCEIVABLE gag ("today", "together" — the crowd hears nothing), so they
# cannot stand ALONE as the anchor; they may still sit inside a longer anchor
# ("today you will marry" is fine). Curated — extend as bad examples show up.
_WEAK_ANCHORS = {
    "今天", "明天", "昨天", "现在", "現在", "时候", "時候", "一刻", "此刻",
    "一起", "一样", "一樣", "这样", "這樣", "那样", "那樣", "怎样", "怎樣",
    "过去", "過去", "以后", "以後", "后来", "後來", "可以", "知道", "因为",
    "因為", "所以", "如果", "真的", "已经", "已經", "曾经", "曾經", "一点",
    "一點", "点点", "點點", "总是", "總是", "开始", "開始", "最后", "最後",
    "一切", "一天", "每天", "刻我",   # "ke wo": cross-word fragment "yi ke" (moment) | "wo" (I)
    "关于", "關於", "对于", "對於", "然后", "然後", "还有", "還有",
}

_LEAD_STRIP = set("这這那哪")        # demonstratives glued onto a fragment
_TAIL_STRIP = set("我你他她它")      # pronouns glued onto a fragment


def _weak(span: str) -> bool:
    """True if the span — or the span minus demonstrative/pronoun glue — is a
    weak-semantics word ("this moment I" → strip "this"/"I" → "moment" → weak)."""
    if span in _WEAK_ANCHORS:
        return True
    core = span
    while core and core[0] in _LEAD_STRIP:
        core = core[1:]
    while core and core[-1] in _TAIL_STRIP:
        core = core[:-1]
    return core in _WEAK_ANCHORS

# Interjection noise ("Yeah", "Oh baby", "Oh Oh Oh") — not a hook on either
# side. _NOISE_RE catches arbitrary repetitions of the interjection units.
_ASCII_NOISE = {"yeah", "oh", "ohh", "baby", "la", "lala", "lalala", "wo",
                "woo", "hey", "ha", "haha", "na", "nana", "comeon", "ohyeah"}
import re as _re
_NOISE_RE = _re.compile(r"(?:oh+|yeah|la|na+|wo+|hey|ha|hoo?|baby|uh|eh)+")

MAX_ANCHOR = 6      # anchors longer than this add no discrimination


def _valid_anchor(span: str) -> bool:
    if " " in span:                       # never bridge an ASCII word boundary
        return False
    if len(span) < 2:
        return False
    if span.isascii():                    # latin: whole-word-sized runs only
        return len(span) >= 5 and span not in _ASCII_NOISE
    if any(ch in _PARTICLES for ch in span):
        return False
    if _weak(span):                       # time/logic words carry no gag
        return False
    return any(ch not in _STOP_CHARS for ch in span)


def is_noise_line(norm: str) -> bool:
    """Lines that are pure interjection ("Yeah", "Oh Oh Oh") — they match
    everything and mean nothing; skipped on BOTH the query and candidate side."""
    if not norm.isascii():
        return False
    core = norm.replace(" ", "")
    return (len(core) < 4 or core in _ASCII_NOISE
            or bool(_NOISE_RE.fullmatch(core)))


def _bigrams(norm: str) -> set[str]:
    return {norm[i:i + 2] for i in range(len(norm) - 1)
            if " " not in norm[i:i + 2]}


def find_anchors(norm_a: str, norm_b: str) -> list[str]:
    """All maximal clean anchors the two lines share, longest first.

    Enumerates A's valid n-grams (2..MAX_ANCHOR) present in B, then drops any
    that is a substring of a longer kept anchor. The particle rule does the
    word-boundary alignment: a cross-word artifact always carries the particle
    that glued it together ("toutou de | kan" -> "de kan" contains the particle
    "de" -> dead)."""
    found = set()
    la = len(norm_a)
    for n in range(min(MAX_ANCHOR, la), 1, -1):
        for i in range(la - n + 1):
            span = norm_a[i:i + n]
            if span in found or not _valid_anchor(span):
                continue
            if span in norm_b:
                found.add(span)
    keep: list[str] = []
    for s in sorted(found, key=len, reverse=True):
        if not any(s in k for k in keep):
            keep.append(s)
    return keep


def _levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def near_dup_ratio(norm_a: str, norm_b: str) -> float:
    """Whole-line similarity for the same-line-reversed shape ("today you will
    marry me" ↔ "tomorrow I will marry you")."""
    if abs(len(norm_a) - len(norm_b)) > max(3, min(len(norm_a), len(norm_b)) // 2):
        return 0.0
    ratio = 1.0 - _levenshtein(norm_a, norm_b) / max(len(norm_a), len(norm_b))
    return ratio if ratio >= 0.5 else 0.0


@dataclass
class RawMatch:
    anchors: list[str]        # clean shared words, longest first
    near_dup: float           # 0..1 whole-line similarity


def match_lines(norm_a: str, norm_b: str) -> RawMatch | None:
    anchors = find_anchors(norm_a, norm_b)
    nd = near_dup_ratio(norm_a, norm_b)
    if not anchors and nd < 0.5:
        return None
    return RawMatch(anchors=anchors, near_dup=round(nd, 3))


class LexicalIndex:
    """Bigram inverted index over every line group (+ a title pseudo-line)."""

    def __init__(self, songs: list[Song]):
        self.songs = {s.song_id: s for s in songs}
        self.postings: dict[str, set[tuple[str, str]]] = {}  # bigram -> {(song, norm)}
        self.heads: dict[str, set[tuple[str, str]]] = {}     # first char -> lines
        for s in songs:
            entries = [lg.norm for lg in s.lines]
            if s.title_norm and len(s.title_norm) >= 2:
                entries.append(s.title_norm)
            for norm in entries:
                for bg in _bigrams(norm):
                    if all(ch in _STOP_CHARS for ch in bg):
                        continue
                    self.postings.setdefault(bg, set()).add((s.song_id, norm))
                core = norm.replace(" ", "")
                if core and not core[0].isascii() and core[0] not in _STOP_CHARS:
                    self.heads.setdefault(core[0], set()).add((s.song_id, norm))

    def candidates(self, query_norm: str, exclude_song: str) -> set[tuple[str, str]]:
        """Bigram overlap ∪ tail-head continuation (an anadiplosis chain pair
        like "…say I love you" → "love is…" shares NO bigram — it needs its own
        channel)."""
        out: set[tuple[str, str]] = set()
        for bg in _bigrams(query_norm):
            for song_id, norm in self.postings.get(bg, ()):
                if song_id != exclude_song:
                    out.add((song_id, norm))
        core = query_norm.replace(" ", "")
        for ch in set(core[-3:]):          # tail WORD, not just the last char
            if ch.isascii() or ch in _STOP_CHARS:
                continue
            for song_id, norm in self.heads.get(ch, ()):
                if song_id != exclude_song:
                    out.add((song_id, norm))
        return out
