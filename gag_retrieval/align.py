"""Timestamp refinement — official lyrics get Whisper's clock. (SKELETON —
functional but not yet wired into the demo; enable per-song when LRC times
prove inaccurate.)

Principle: Whisper's TEXT is unreliable on sung Mandarin (「嫁給我」→「駕給我」
kills a lexical gag) but its TIMELINE is roughly right; scraped official lyrics
are the reverse. So align the two character streams and transfer times onto the
official lines — ASR misheard characters still occupy the right slots, so the
mapping survives them.

Input Whisper JSON (produced by whisper/faster-whisper/the project's stems
server; both shapes accepted):
    {"words": [{"word": "嫁", "start": 81.2, "end": 81.4}, ...]}
    {"segments": [{"words": [...]}, ...]}          # openai-whisper default

Usage:
    python3 -m gag_retrieval.align --lrc lrc_test/偷偷.txt \
        --whisper 偷偷.whisper.json --out lrc_test/偷偷.aligned.txt
The output is a normal LRC txt (same format the corpus loader reads) with
refined line times, plus a per-line delta report on stdout.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path

from .lrc import parse_song
from .zhnorm import normalize


def load_whisper_chars(path: str | Path) -> list[tuple[str, float]]:
    """Flatten a Whisper JSON into a [(normalized_char, time)] stream."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    words = d.get("words") or [w for seg in d.get("segments", [])
                               for w in seg.get("words", [])]
    stream: list[tuple[str, float]] = []
    for w in words:
        text = normalize(str(w.get("word", "")))
        t0 = float(w.get("start", 0.0))
        t1 = float(w.get("end", t0))
        if not text:
            continue
        step = (t1 - t0) / max(1, len(text))     # spread time across chars
        for i, ch in enumerate(text):
            if ch != " ":
                stream.append((ch, t0 + i * step))
    return stream


def align_song(lrc_path: str | Path, whisper_json: str | Path,
               max_shift: float = 15.0) -> list[dict]:
    """Refined per-line times. Returns [{text, old, new, delta, coverage}].

    coverage = fraction of the line's characters that found a Whisper match;
    low coverage (<0.4) means the refinement is untrustworthy — keep old time.
    max_shift guards against pathological matches (a chorus line matching a
    different chorus repeat): refuse moves larger than this many seconds.
    """
    song = parse_song(lrc_path)
    wchars = load_whisper_chars(whisper_json)
    wtext = "".join(c for c, _ in wchars)

    # official stream: chars tagged with (line occurrence order)
    events = sorted(((t, lg) for lg in song.lines for t in lg.times),
                    key=lambda x: x[0])
    ochars: list[tuple[str, int]] = []           # (char, event_idx)
    for idx, (_, lg) in enumerate(events):
        for ch in lg.norm:
            if ch != " ":
                ochars.append((ch, idx))
    otext = "".join(c for c, _ in ochars)

    sm = difflib.SequenceMatcher(None, otext, wtext, autojunk=False)
    hit_times: dict[int, list[float]] = {}       # event_idx -> matched times
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            _, ev_idx = ochars[blk.a + k]
            hit_times.setdefault(ev_idx, []).append(wchars[blk.b + k][1])

    report = []
    for idx, (old_t, lg) in enumerate(events):
        times = sorted(hit_times.get(idx, []))
        cov = len(times) / max(1, len(lg.norm.replace(" ", "")))
        new_t = old_t
        if times and cov >= 0.4:
            cand = times[len(times) // 4]        # robust early quantile
            if abs(cand - old_t) <= max_shift:
                new_t = round(cand, 2)
        report.append({"text": lg.text, "old": round(old_t, 2), "new": new_t,
                       "delta": round(new_t - old_t, 2),
                       "coverage": round(cov, 2)})
    return report


def write_lrc(report: list[dict], out_path: str | Path):
    lines = []
    for r in sorted(report, key=lambda x: x["new"]):
        m, s = divmod(r["new"], 60.0)
        lines.append(f"[{int(m):02d}:{s:05.2f}]{r['text']}")
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="LRC 時間戳 Whisper 精修（骨架）")
    p.add_argument("--lrc", required=True)
    p.add_argument("--whisper", required=True, help="Whisper 詞級 JSON")
    p.add_argument("--out", required=True, help="精修後的 LRC txt")
    args = p.parse_args()
    report = align_song(args.lrc, args.whisper)
    moved = [r for r in report if abs(r["delta"]) > 0.3]
    for r in report:
        flag = " *" if abs(r["delta"]) > 0.3 else ""
        print(f"  {r['old']:7.2f} -> {r['new']:7.2f} ({r['delta']:+5.2f}s "
              f"cov {r['coverage']:.2f}){flag}  {r['text']}")
    write_lrc(report, args.out)
    print(f"[align] wrote {args.out} ({len(moved)}/{len(report)} lines moved >0.3s)")


if __name__ == "__main__":
    main()
