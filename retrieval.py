"""Índice invertido y recuperación léxica y semántica."""

from collections import Counter, defaultdict
import hashlib
from heapq import nsmallest
import math
import os
from pathlib import Path
import sys
import re
from tempfile import NamedTemporaryFile
import unicodedata

from corpus import document_text

LEXICAL = ("jaccard", "tfidf", "bm25")
METHODS = (*LEXICAL, "semantic")
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
WORD = re.compile(r"[^\W_]+")
STOPWORDS = frozenset("""
a an and are as at be been being but by can could did do does doing for from had has have
having he her hers him his how i if in into is it its me my of on or our ours she should
so than that the their theirs them then there these they this those to too up us was we
were what when where which who whom why will with would you your yours
""".split())


def tokenize(text):
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [word for word in WORD.findall(normalized) if word not in STOPWORDS]


class InvertedIndex:
    def __init__(self, documents):
        self.postings = defaultdict(dict)
        self.lengths = {}
        self.unique_terms = {}
        for identifier, row in documents.items():
            counts = Counter(tokenize(document_text(row)))
            self.lengths[identifier] = counts.total()
            self.unique_terms[identifier] = len(counts)
            for term, frequency in counts.items():
                self.postings[term][identifier] = frequency

        size = len(documents)
        self.mean_length = sum(self.lengths.values()) / size
        self.tfidf_idf = {}
        self.bm25_idf = {}
        squared_norm = Counter()
        for term, posting in self.postings.items():
            df = len(posting)
            idf = math.log((size + 1) / (df + 1)) + 1
            self.tfidf_idf[term] = idf
            self.bm25_idf[term] = math.log1p((size - df + 0.5) / (df + 0.5))
            for identifier, frequency in posting.items():
                squared_norm[identifier] += (frequency * idf) ** 2
        self.norms = {key: math.sqrt(value) for key, value in squared_norm.items()}

    def search(self, query, method, limit):
        terms = Counter(tokenize(query))
        scores = Counter()
        if method == "jaccard":
            for term in terms:
                scores.update(self.postings.get(term, {}).keys())
            for key, overlap in scores.items():
                scores[key] = overlap / (len(terms) + self.unique_terms[key] - overlap)
        elif method == "tfidf":
            query_weights = {term: tf * self.tfidf_idf[term]
                             for term, tf in terms.items() if term in self.tfidf_idf}
            query_norm = math.sqrt(sum(weight ** 2 for weight in query_weights.values()))
            for term, weight in query_weights.items():
                for key, tf in self.postings[term].items():
                    scores[key] += weight * tf * self.tfidf_idf[term]
            for key in scores:
                scores[key] /= query_norm * self.norms[key]
        elif method == "bm25":
            k1, b = 1.2, 0.75
            for term in terms:
                for key, tf in self.postings.get(term, {}).items():
                    length_factor = 1 - b + b * self.lengths[key] / self.mean_length
                    scores[key] += self.bm25_idf[term] * tf * (k1 + 1) / (tf + k1 * length_factor)
        else:
            raise ValueError(f"Método léxico desconocido: {method}")
        return nsmallest(limit, scores.items(), key=lambda hit: (-hit[1], hit[0]))


class SemanticIndex:
    def __init__(self, documents, directory):
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        faiss.omp_set_num_threads(1)
        self.ids = sorted(documents)
        self.model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
        dimension = self.model.get_sentence_embedding_dimension()
        digest = hashlib.sha256(f"v1:{MODEL}:{MODEL_REVISION}".encode())
        for identifier in self.ids:
            for field in (identifier, document_text(documents[identifier])):
                data = field.encode("utf-8")
                digest.update(len(data).to_bytes(8, "big"))
                digest.update(data)
        path = directory / f"documents-{digest.hexdigest()}.faiss"
        if path.exists():
            self.index = faiss.read_index(str(path))
            if (self.index.ntotal != len(self.ids) or self.index.d != dimension
                    or self.index.metric_type != faiss.METRIC_INNER_PRODUCT):
                raise ValueError(f"Índice FAISS incompatible: {path}")
            print(f"Índice FAISS reutilizado: {path}", file=sys.stderr)
            return

        self.index = faiss.IndexFlatIP(dimension)
        for offset in range(0, len(self.ids), 256):
            batch = self.ids[offset:offset + 256]
            self.index.add(self.encode([document_text(documents[key]) for key in batch]))
        with NamedTemporaryFile(dir=directory, delete=False) as output:
            temporary = Path(output.name)
        try:
            faiss.write_index(self.index, str(temporary))
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"Índice FAISS guardado: {path}", file=sys.stderr)

    def encode(self, texts):
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )

    def search_many(self, queries, limit):
        # Buscar todos permite aplicar el mismo desempate también en el límite del top-n.
        for offset in range(0, len(queries), 32):
            vectors = self.encode(queries[offset:offset + 32])
            scores, positions = self.index.search(vectors, len(self.ids))
            for row_scores, row_positions in zip(scores, positions, strict=True):
                hits = ((self.ids[position], float(score))
                        for score, position in zip(row_scores, row_positions, strict=True))
                yield nsmallest(limit, hits, key=lambda hit: (-hit[1], hit[0]))

