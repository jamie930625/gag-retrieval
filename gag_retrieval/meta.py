"""Positional metadata ("line-position science") — honest feasibility facts per line.

The retrieval layer must NOT filter on feasibility; it attaches these facts so
the agent (or the demo viewer) weighs them itself:

    line_end        when this line finishes (next vocal event, capped)
    section_pos     section start / middle / end — position within its lyric section
                    (sections split on >=4s vocal gaps: verse/chorus breaks)
    to_section_end  seconds until the section ends (natural exit points)
    runway          seconds of song remaining after this line (entering B on a
                    line with 20s runway = the party dies in 20s)
    pos_pct         where in the song this moment sits (0..1)
"""

from __future__ import annotations

from .lrc import Song

SECTION_GAP = 4.0     # vocal silence >= this splits sections
LINE_CAP = 6.0        # a line never "lasts" longer than this
TAIL_PAD = 4.0        # assumed duration of the very last line


def song_events(song: Song) -> list[tuple[float, str]]:
    """Every sung occurrence (time, text), chronological."""
    ev = [(t, lg.text) for lg in song.lines for t in lg.times]
    ev.sort(key=lambda x: x[0])
    return ev


def vocal_end(song: Song) -> float:
    ev = song_events(song)
    return (ev[-1][0] + TAIL_PAD) if ev else 0.0


def _sections(times: list[float]) -> list[tuple[int, int]]:
    """(start_idx, end_idx) inclusive index ranges over the event list."""
    if not times:
        return []
    out, s = [], 0
    for i in range(1, len(times)):
        if times[i] - times[i - 1] >= SECTION_GAP:
            out.append((s, i - 1))
            s = i
    out.append((s, len(times) - 1))
    return out


def line_meta(song: Song, t: float) -> dict:
    """Positional facts for the vocal event at (or nearest to) time t."""
    ev = song_events(song)
    if not ev:
        return {}
    times = [e[0] for e in ev]
    idx = min(range(len(times)), key=lambda i: abs(times[i] - t))
    nxt = times[idx + 1] if idx + 1 < len(times) else times[idx] + TAIL_PAD
    line_end = round(min(nxt, times[idx] + LINE_CAP), 2)
    end = vocal_end(song)

    sec_label, to_sec_end = "一般", None
    for s, e in _sections(times):
        if s <= idx <= e:
            sec_end = min(times[e] + LINE_CAP, end)
            to_sec_end = round(max(0.0, sec_end - times[idx]), 1)
            n = e - s + 1
            if n <= 2:
                sec_label = "短段"
            elif idx - s == 0:
                sec_label = "段首"
            elif e - idx <= 1:
                sec_label = "段尾"
            else:
                sec_label = "段中"
            break

    return {
        "line_end": line_end,
        "section_pos": sec_label,
        "to_section_end": to_sec_end,
        "runway": round(max(0.0, end - times[idx]), 1),
        "pos_pct": round(times[idx] / end, 3) if end > 0 else 0.0,
    }
