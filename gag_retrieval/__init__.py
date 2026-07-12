"""gag_retrieval — 字面對應接歌檢索（兩階段）.

Stage 1: line-level lexical (+optional semantic) retrieval over a lyric corpus.
Stage 2: LLM judge over the top-K candidate songs (full lyrics + evidence).

Self-contained: stdlib only. Does NOT import from or modify the main project.
"""
