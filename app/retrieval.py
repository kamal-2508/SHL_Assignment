"""
Retrieval over the SHL Individual Test Solutions catalog.

Strategy:
  1. LLM rewrites the conversation into a rich search query (no hardcoded aliases).
  2. TF-IDF cosine similarity over name + description + keys + job levels.
  3. Substring boost on name tokens from the LLM-expanded query.

The LLM query rewrite is the key upgrade: instead of a hardcoded alias table,
we ask the model to expand abbreviations and extract intent from the full
conversation naturally. "OPQ" becomes "Occupational Personality Questionnaire
personality behavior workplace" automatically, without us having to anticipate
every possible abbreviation.
"""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"


class Catalog:
    def __init__(self, path: Path = DATA_PATH):
        self.items = json.loads(path.read_text(encoding="utf-8"))
        self.by_id = {it["id"]: it for it in self.items}
        self._build_index()

    def _doc_text(self, it):
        parts = [
            it["name"], it["name"],  # name weighted twice
            it.get("description", ""),
            " ".join(it.get("keys_full", [])),
            " ".join(it.get("job_levels", [])),
            " ".join(it.get("languages", [])),
        ]
        return " ".join(p for p in parts if p)

    def _build_index(self):
        self.corpus = [self._doc_text(it) for it in self.items]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=20000
        )
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def search(self, query: str, top_k: int = 30):
        if not query.strip():
            return []

        qvec = self.vectorizer.transform([query])
        sims = cosine_similarity(qvec, self.matrix)[0]

        # substring boost: any token from the query that appears
        # literally in an item name gets a score bump
        tokens = set(re.findall(r"[a-z0-9\+\.\#]+", query.lower()))
        boosted = list(sims)
        for i, it in enumerate(self.items):
            name_lower = it["name"].lower()
            if any(len(tok) >= 2 and tok in name_lower for tok in tokens):
                boosted[i] += 0.4

        ranked = sorted(
            range(len(self.items)), key=lambda i: boosted[i], reverse=True
        )
        return [self.items[i] for i in ranked[:top_k] if boosted[i] > 0]

    def get_by_name(self, name_fragment: str):
        frag = name_fragment.lower()
        return [it for it in self.items if frag in it["name"].lower()]

    def validate_urls(self, urls):
        valid = {it["url"] for it in self.items}
        return [u for u in urls if u in valid]


_catalog_singleton = None


def get_catalog() -> Catalog:
    global _catalog_singleton
    if _catalog_singleton is None:
        _catalog_singleton = Catalog()
    return _catalog_singleton