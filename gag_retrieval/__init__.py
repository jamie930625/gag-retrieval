"""gag_retrieval — lexical-correspondence song-transition retrieval (two-stage).

Stage 1: line-level lexical (+optional semantic) retrieval over a lyric corpus.
Stage 2: LLM judge over the top-K candidate songs (full lyrics + evidence).

Self-contained: stdlib only. Does NOT import from or modify the main project.
"""
