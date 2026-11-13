"""Felles korpus og søk for RAG-pipelinen og agenten.

Korpuset bygges ÉN gang for alle spørsmålene i et utvalg, og begge betingelsene bruker
samme `search`:
    from src.korpus import Korpus
    korpus = Korpus(examples)                       # chunking + embeddinger, én gang
    hits = korpus.search("net revenue 2015", doc_id="AWK/2014/page_121.pdf-2")

- RAG-pipelinen kaller `search` én gang per spørsmål, med spørsmålet som query.
- Agenten får `search` som verktøy med doc_id bundet til spørsmålet, så modellen
  bare bestemmer query. Den kan søke flere ganger med ulike formuleringer.

Designvalg (forelesning 4, «Dokumentgrunnlag og chunks er designvalg»):
- RETRIEVAL_SCOPE = "dokument": søk bare i spørsmålets egen rapportside.
- Chunk-ID-ene er FinQAs egne (text_i, table_i), så treff kan sammenlignes med gold_inds.
- Tabellrader skrives om til tekst med kolonnenavn fra rad 0:
  "leased facilities2: unitedstates = 2.1; othercountries = 6.0"
- Negative tall som "-32 ( 32 )" ryddes til "-32", så kalkulatoren kan lese dem.
- Tomme setninger (bare ".") fjernes.
"""
import re

import numpy as np

from src.config import CHUNK_STRATEGY, EMBEDDING_MODEL, INDEX_BACKEND, SENTENCES_PER_CHUNK, TOP_K

# --- Chunking ---

# Parentes med KUN tall inni ("( 32 )"), så "amount ( in millions )" blir stående.
_NEG_PAREN_RE = re.compile(r"\s*\(\s*[\d.,]+\s*\)")


def clean_cell(cell: str) -> str:
    """'-32 ( 32 )' -> '-32', '$ -4614 ( 4614 )' -> '$ -4614'."""
    return _NEG_PAREN_RE.sub("", cell).strip()


def row_to_text(row: list, header: list) -> str:
    """['leased facilities2', '2.1', '6.0'] -> 'leased facilities2: unitedstates = 2.1; othercountries = 6.0'"""
    label = row[0].strip()
    parts = []
    for col_name, value in zip(header[1:], row[1:]):
        value, col_name = clean_cell(value), col_name.strip()
        parts.append(f"{col_name} = {value}" if col_name else value)
    return f"{label}: " + "; ".join(parts) if label else "; ".join(parts)


def make_chunks(example: dict, strategy: str = CHUNK_STRATEGY,
                sentences_per_chunk: int = SENTENCES_PER_CHUNK) -> list:
    """Del én rapportside i chunks: {chunk_id, source_ids, kind, text}."""
    chunks = []
    # Tekst: FinQA nummererer pre_text og post_text fortløpende (text_0, text_1, ...).
    # Hver del chunkes for seg, så en setningsgruppe aldri går på tvers av tabellen.
    n_pre = len(example["pre_text"])
    size = 1 if strategy == "setning" else sentences_per_chunk
    for part, offset in [(example["pre_text"], 0), (example["post_text"], n_pre)]:
        sentences = [(offset + i, s.strip()) for i, s in enumerate(part) if re.search(r"[a-z0-9]", s)]
        for start in range(0, len(sentences), size):
            group = sentences[start:start + size]
            ids = [f"text_{i}" for i, _ in group]
            chunk_id = ids[0] if len(ids) == 1 else f"text_{group[0][0]}-{group[-1][0]}"
            chunks.append({"chunk_id": chunk_id, "source_ids": ids, "kind": "tekst",
                           "text": " ".join(s for _, s in group)})
    # Tabell: rad 0 er kolonnenavn (slik FinQA tolker den), men kan også være data,
    # så den blir også en egen chunk.
    table = example["table"]
    if table:
        header = table[0]
        for i, row in enumerate(table):
            text = "kolonner: " + " | ".join(clean_cell(c) for c in row) if i == 0 else row_to_text(row, header)
            chunks.append({"chunk_id": f"table_{i}", "source_ids": [f"table_{i}"], "kind": "tabell", "text": text})
    return chunks


# --- Korpus og søk ---

class Korpus:
    """Alle chunks for et sett FinQA-eksempler, med indeks. Bygges én gang."""

    def __init__(self, examples: dict, backend: str = INDEX_BACKEND, chunk_strategy: str = CHUNK_STRATEGY):
        # Steg 1: Chunk hvert dokument. Nøkkelen er spørsmålets id (hvert FinQA-eksempel
        # har sin egen rapportside). chunk_strategy kan endres for eksperimenter på dev.
        self.chunks = {doc_id: make_chunks(ex, strategy=chunk_strategy) for doc_id, ex in examples.items()}
        self.backend = backend

        # Steg 2: Lag vektorer. Embeddinger regnes ut for alle chunks på én gang (raskt),
        # TF-IDF bygges per dokument ved første søk (IDF skal komme fra dokumentet selv).
        if backend == "embedding":
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer(EMBEDDING_MODEL)
            texts = [c["text"] for doc in self.chunks.values() for c in doc]
            matrix = self.encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False,
                                         batch_size=128)
            self.vectors, start = {}, 0
            for doc_id, doc in self.chunks.items():
                self.vectors[doc_id] = matrix[start:start + len(doc)]
                start += len(doc)
        elif backend == "tfidf":
            self._tfidf = {}
        else:
            raise ValueError(f"Ukjent backend: {backend}")

    def _scores(self, query: str, doc_id: str) -> np.ndarray:
        if self.backend == "embedding":
            q = self.encoder.encode([query], normalize_embeddings=True, show_progress_bar=False)
            return (self.vectors[doc_id] @ q.T).ravel()
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        if doc_id not in self._tfidf:
            vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True, token_pattern=r"(?u)\b\w+\b")
            self._tfidf[doc_id] = (vec, vec.fit_transform([c["text"] for c in self.chunks[doc_id]]))
        vec, matrix = self._tfidf[doc_id]
        return cosine_similarity(vec.transform([query]), matrix).ravel()

    def search(self, query: str, doc_id: str, k: int = TOP_K) -> list:
        """De k chunkene i dokumentet doc_id som ligner mest på query (cosinuslikhet)."""
        scores = self._scores(query, doc_id)
        top = np.argsort(-scores)[:k]
        return [{"rank": r, "score": round(float(scores[i]), 3), **self.chunks[doc_id][i]}
                for r, i in enumerate(top, start=1)]


def gold_coverage(hits: list, gold_inds: list) -> tuple:
    """(alle gold_inds hentet?, minst én hentet?) for en liste treff."""
    hentet = {sid for h in hits for sid in h["source_ids"]}
    gold = set(gold_inds)
    return gold <= hentet, bool(gold & hentet)
