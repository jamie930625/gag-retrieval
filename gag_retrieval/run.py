"""CLI — run the two-stage 字面對應 next-song retrieval.

    python3 -m gag_retrieval.run --corpus lrc_test --query 偷偷
    python3 -m gag_retrieval.run --corpus lrc_test --query 偷偷 --llm gemini
    python3 -m gag_retrieval.run --corpus lrc_test --all          # every song

Stage-1 ranking always prints; --llm picks the stage-2 backend (default
'prompt' writes the judge prompt to out/ for manual pasting into Gemini).
Reports land in gag_retrieval/out/.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

if __package__ in (None, ""):                      # allow `python3 gag_retrieval/run.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from gag_retrieval import judge as jd
    from gag_retrieval.hooks import load_popularity
    from gag_retrieval.lexical import LexicalIndex
    from gag_retrieval.lrc import load_corpus, parse_song
    from gag_retrieval.retrieve import retrieve
else:
    from . import judge as jd
    from .hooks import load_popularity
    from .lexical import LexicalIndex
    from .lrc import load_corpus, parse_song
    from .retrieve import retrieve


def _fmt_time(t: float) -> str:
    return f"{int(t) // 60:02d}:{t % 60:05.2f}"


def _print_stage1(query, results):
    print(f"\n════ 歌曲 A：《{query.title}》 → 第一階段候選 ════")
    if not results:
        print("  (沒有任何字面對應的候選)")
        return
    for rank, r in enumerate(results, 1):
        print(f"\n  #{rank} 《{r.song_id}》  score={r.score}")
        for ev in r.evidence:
            print(f"      A[{_fmt_time(ev.a_time)}] {ev.a_text}")
            print(f"      ↔ B[{_fmt_time(ev.b_time)}] {ev.b_text}"
                  f"   [{ev.b_label}] 錨點「{ev.anchor}」({ev.anchor_label})"
                  f"  ({ev.score})")


def main():
    p = argparse.ArgumentParser(description="字面對應接歌檢索（兩階段）")
    p.add_argument("--corpus", required=True, help="歌詞資料夾（.txt/.lrc，帶時間戳）")
    p.add_argument("--query", help="歌曲 A：語料內的歌名，或外部歌詞檔路徑")
    p.add_argument("--all", action="store_true", help="把語料每首歌都當一次 A")
    p.add_argument("--topk", type=int, default=10, help="進第二階段的候選歌數")
    p.add_argument("--llm", default="prompt",
                   choices=["prompt", "gemini", "openai", "vllm"],
                   help="第二階段後端（prompt=只輸出提示詞檔）")
    p.add_argument("--semantic", action="store_true",
                   help="啟用語意通道（需要 sentence-transformers）")
    p.add_argument("--popularity", default=None,
                   help="選配 {歌名: 0..1} 知名度 JSON")
    p.add_argument("--out", default=str(Path(__file__).parent / "out"))
    args = p.parse_args()

    songs = load_corpus(args.corpus)
    if not songs:
        sys.exit(f"corpus 是空的：{args.corpus}")
    print(f"[corpus] {len(songs)} 首歌，共 {sum(len(s.lines) for s in songs)} 個句組")

    index = LexicalIndex(songs)
    popularity = load_popularity(args.popularity)
    semantic = None
    if args.semantic:
        from gag_retrieval.semantic import SemanticChannel
        semantic = SemanticChannel()
        print("[semantic] 已啟用" if semantic.available
              else f"[semantic] 無法載入，改用純字面通道（{semantic.reason}）")

    if args.all:
        queries = songs
    elif args.query:
        by_id = {s.song_id: s for s in songs}
        if args.query in by_id:
            queries = [by_id[args.query]]
        elif Path(args.query).exists():
            q = parse_song(args.query)          # external song A, not in corpus
            queries = [q]
        else:
            sys.exit(f"找不到 query：{args.query}（語料內有：{sorted(by_id)}）")
    else:
        sys.exit("請給 --query <歌名|歌詞檔> 或 --all")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for query in queries:
        results = retrieve(query, index, topk_songs=args.topk,
                           popularity=popularity, semantic=semantic)
        _print_stage1(query, results)

        report = {"query": query.title,
                  "stage1": [dataclasses.asdict(r) for r in results]}
        if results:
            prompt = jd.build_prompt(query, results, index.songs)
            prompt_path = out_dir / f"prompt_{query.title}.md"
            prompt_path.write_text(prompt, encoding="utf-8")
            if args.llm == "prompt":
                print(f"\n  [stage2] 提示詞已寫到 {prompt_path} — 貼給任何 LLM 即可")
            else:
                print(f"\n  [stage2] 呼叫 {args.llm} ...")
                try:
                    raw, parsed = jd.run_judge(prompt, args.llm)
                    report["stage2_raw"] = raw
                    report["stage2"] = parsed
                    if parsed:
                        print(f"  ✚ 下一首：《{parsed.get('next_song')}》"
                              f" [{parsed.get('type')}] 信心 {parsed.get('confidence')}")
                        fl, el = parsed.get("from_line", {}), parsed.get("enter_line", {})
                        print(f"    A 出點 [{fl.get('time')}]「{fl.get('text')}」")
                        print(f"    B 進點 [{el.get('time')}]「{el.get('text')}」")
                        print(f"    why: {parsed.get('why')}")
                    else:
                        print(f"  (回覆解析失敗，raw 已存進報告) {str(raw)[:200]}")
                except Exception as e:
                    print(f"  [stage2] 失敗：{e}")
                    report["stage2_error"] = str(e)
        report_path = out_dir / f"report_{query.title}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        print(f"  [report] {report_path}")


if __name__ == "__main__":
    main()
