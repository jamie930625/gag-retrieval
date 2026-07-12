"""Optional semantic channel (dense line embeddings).

Catches the small slice lexical matching can't ("嫁給我" ↔ "娶你回家") — a
SUPPLEMENT, never the main channel. Requires sentence-transformers with a
multilingual model; when unavailable the channel silently reports disabled and
stage 1 runs lexical-only (all four golden examples are lexical).

High threshold on purpose: paraphrase-level only. Topic-level similarity
("both are love songs") is the boring opposite of a gag.
"""

from __future__ import annotations

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
SIM_THRESHOLD = 0.72        # paraphrase-level, NOT topic-level
STRENGTH_SCALE = 0.6        # semantic hits rank below a solid lexical hit


class SemanticChannel:
    def __init__(self):
        self.model = None
        self.reason = ""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME)
        except Exception as e:
            self.reason = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        return self.model is not None

    def line_hits(self, query_texts: list[str], cand_texts: list[str],
                  threshold: float = SIM_THRESHOLD) -> list[tuple[int, int, float]]:
        """(query_idx, cand_idx, strength) for pairs above threshold."""
        if not self.available or not query_texts or not cand_texts:
            return []
        import numpy as np
        eq = self.model.encode(query_texts, convert_to_numpy=True,
                               normalize_embeddings=True)
        ec = self.model.encode(cand_texts, convert_to_numpy=True,
                               normalize_embeddings=True)
        sims = eq @ ec.T
        hits = []
        for i, j in zip(*np.where(sims >= threshold)):
            sim = float(sims[i, j])
            strength = STRENGTH_SCALE * (sim - threshold) / (1.0 - threshold)
            hits.append((int(i), int(j), strength))
        return hits
