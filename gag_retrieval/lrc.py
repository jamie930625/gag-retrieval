"""LRC / timestamped-txt lyric parsing into line groups.

A corpus file is one song: ``[mm:ss.xx]lyric line`` per row (plain .txt export
of an LRC works as-is; multiple leading time tags per row are accepted).
The song TITLE comes from the filename stem.

Identical lines (after normalization) collapse into one LineGroup carrying all
occurrence times — repetition count is the chorus signal, and the first time is
the natural DJ entry point for that line.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .zhnorm import normalize

_TIME_RE = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
# Credit/metadata rows ("Lyricist: ...", "Mixing : ..."), not lyrics.
_META_RE = re.compile(
    r"^\s*(?:作词|作詞|作曲|编曲|編曲|监制|監製|制作|製作|出品|发行|發行|词|曲|"
    r"和声|和聲|吉他|贝斯|贝司|键盘|鍵盤|鼓|弦乐|弦樂|录音|錄音|混音|母带|母帶|"
    r"OP|SP|ISRC)[^:：]{0,10}[:：]")


@dataclass
class LineGroup:
    text: str                 # original display text (first occurrence)
    norm: str                 # normalized matching form
    times: list[float]        # every occurrence (sec), sorted
    first_idx: int            # order-of-appearance index of first occurrence

    @property
    def repeats(self) -> int:
        return len(self.times)


@dataclass
class Song:
    song_id: str              # filename stem = title
    title: str
    title_norm: str
    path: str
    lines: list[LineGroup] = field(default_factory=list)   # by first appearance

    def line_by_norm(self, norm: str) -> LineGroup | None:
        for lg in self.lines:
            if lg.norm == norm:
                return lg
        return None


def parse_song(path: str | Path) -> Song:
    path = Path(path)
    title = path.stem
    song = Song(song_id=title, title=title, title_norm=normalize(title),
                path=str(path))
    groups: dict[str, LineGroup] = {}
    order = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        tags = _TIME_RE.findall(raw)
        text = _TIME_RE.sub("", raw).strip()
        if not text or _META_RE.match(text):
            continue
        # header row "artist - title" (usually the very first tagged row)
        if order == 0 and " - " in text:
            continue
        norm = normalize(text)
        if len(norm) < 2:      # "Oh", "Ye" — nothing to match on
            continue
        times = [int(m) * 60 + float(s) for m, s in tags] or [0.0]
        if norm in groups:
            groups[norm].times.extend(times)
        else:
            groups[norm] = LineGroup(text=text, norm=norm, times=list(times),
                                     first_idx=order)
            order += 1
    song.lines = sorted(groups.values(), key=lambda g: g.first_idx)
    for lg in song.lines:
        lg.times.sort()
    return song


def load_corpus(corpus_dir: str | Path, exts=(".txt", ".lrc")) -> list[Song]:
    corpus_dir = Path(corpus_dir)
    songs = []
    for f in sorted(corpus_dir.iterdir()):
        if f.suffix.lower() in exts and f.is_file():
            try:
                songs.append(parse_song(f))
            except Exception as e:
                print(f"[lrc] skip {f.name}: {e}")
    return songs
