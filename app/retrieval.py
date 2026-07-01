"""
Lightweight retrieval over the SHL Individual Test Solutions catalog.

Strategy (kept deliberately simple/fast for the 30s per-call budget):
  1. TF-IDF cosine similarity over name + description + keys + job levels.
  2. Exact / substring name matching boost (handles short queries like
     "OPQ", "GSA", "DSI" which TF-IDF alone underweights).
  3. A small alias table for the most common abbreviations seen in the
     domain, so "OPQ" reliably surfaces "Occupational Personality
     Questionnaire OPQ32r", etc.

The result is a compact candidate list handed to the LLM as grounding
context -- the LLM never invents catalog items, it only ever chooses
from / discusses what retrieval surfaces (plus the conversation history,
which may reference earlier items already shown by name).
"""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "catalog.json"

ALIASES = {
    "opq": "occupational personality questionnaire opq32r",
    "opq32r": "occupational personality questionnaire opq32r",
    "gsa": "global skills assessment",
    "dsi": "dependability and safety instrument",
    "mq": "motivation questionnaire",
    "sjt": "situational judgement situational judgment",
    "jfa": "job focused assessment",
    "g+": "verify g+ general ability",
    "verify g+": "shl verify interactive g+",
    "svar": "svar spoken",
    "ucf": "universal competency report",
}


class Catalog:
    def __init__(self, path: Path = DATA_PATH):
        self.items = json.loads(path.read_text(encoding="utf-8"))
        self.by_id = {it["id"]: it for it in self.items}
        self._build_index()

    def _doc_text(self, it):
        parts = [
            it["name"], it["name"],  # weight name higher
            it.get("description", ""),
            " ".join(it.get("keys_full", [])),
            " ".join(it.get("job_levels", [])),
        ]
        return " ".join(p for p in parts if p)

    def _build_index(self):
        self.corpus = [self._doc_text(it) for it in self.items]
        self.vectorizer = TfidfVectorizer(
            stop_words="english", ngram_range=(1, 2), max_features=20000
        )
        self.matrix = self.vectorizer.fit_transform(self.corpus)

    def _expand_query(self, query: str) -> str:
        q = query.lower()
        extra = []
        for alias, expansion in ALIASES.items():
            if re.search(r"\b" + re.escape(alias) + r"\b", q):
                extra.append(expansion)
        return query + " " + " ".join(extra)

    def search(self, query: str, top_k: int = 25):
        if not query.strip():
            return []
        expanded = self._expand_query(query)
        qvec = self.vectorizer.transform([expanded])
        sims = cosine_similarity(qvec, self.matrix)[0]

        # substring / alias boost on name
        q_lower = query.lower()
        tokens = set(re.findall(r"[a-z0-9\+\.]+", q_lower))
        for alias in ALIASES:
            if alias in q_lower:
                tokens.add(alias)

        boosted = list(sims)
        for i, it in enumerate(self.items):
            name_lower = it["name"].lower()
            if any(tok and len(tok) >= 2 and tok in name_lower for tok in tokens):
                boosted[i] += 0.5

        ranked = sorted(range(len(self.items)), key=lambda i: boosted[i], reverse=True)
        results = [self.items[i] for i in ranked[:top_k] if boosted[i] > 0]
        return results

    def get_by_name(self, name_fragment: str):
        """Direct case-insensitive substring lookup, used to force-include
        items explicitly named by the user (e.g. for comparisons)."""
        frag = name_fragment.lower()
        return [it for it in self.items if frag in it["name"].lower()]

    def validate_urls(self, urls):
        """Returns the subset of urls that exist in the catalog."""
        valid = {it["url"] for it in self.items}
        return [u for u in urls if u in valid]


_catalog_singleton = None


def get_catalog() -> Catalog:
    global _catalog_singleton
    if _catalog_singleton is None:
        _catalog_singleton = Catalog()
    return _catalog_singleton
