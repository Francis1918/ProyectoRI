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

import numpy as np

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
        terms = Counter(tokenize(query)) #convertimos la consulta en palabras y puntuamos cada domuento con el metodo elegido
        scores = Counter() #y devolvemos los ID con mayor puntaje
        if method == "jaccard":
            for term in terms: #palabras en comun con las palabras distintas
                scores.update(self.postings.get(term, {}).keys())
            for key, overlap in scores.items():
                scores[key] = overlap / (len(terms) + self.unique_terms[key] - overlap)
        elif method == "tfidf":
            query_weights = {term: tf * self.tfidf_idf[term] #palabras freq en el doc y no comun el corpus
                             for term, tf in terms.items() if term in self.tfidf_idf}
            query_norm = math.sqrt(sum(weight ** 2 for weight in query_weights.values()))
            for term, weight in query_weights.items():
                for key, tf in self.postings[term].items():
                    scores[key] += weight * tf * self.tfidf_idf[term]
            for key in scores:
                scores[key] /= query_norm * self.norms[key]
        elif method == "bm25":
            k1, b = 1.2, 0.75 #palabras raras modera cuntas suma repeirlas y lo ajsutas po rla longitud del docuemtno
            for term in terms:
                for key, tf in self.postings.get(term, {}).items():
                    length_factor = 1 - b + b * self.lengths[key] / self.mean_length
                    scores[key] += self.bm25_idf[term] * tf * (k1 + 1) / (tf + k1 * length_factor)
        else:
            raise ValueError(f"Método léxico desconocido: {method}")
        return nsmallest(limit, scores.items(), key=lambda hit: (-hit[1], hit[0]))
# devuelve hasta un limite de pares con el ilmite de puntaje de mayor a menor si empatan se orden por ID
    # --- Parte 2: serialización compacta para el despliegue serverless -------------------------
    # La puntuación sigue siendo `search` (arriba): solo cambia dónde viven los postings. Se
    # guardan como arrays (CSR) y cada lista de postings se materializa al consultar el término.

    def export_arrays(self):
        ids = list(self.lengths)
        position = {identifier: index for index, identifier in enumerate(ids)}
        terms = list(self.postings)
        offsets = np.zeros(len(terms) + 1, dtype=np.int64)
        for index, term in enumerate(terms):
            offsets[index + 1] = offsets[index] + len(self.postings[term])
        doc_pos = np.empty(offsets[-1], dtype=np.int32)
        frequency = np.empty(offsets[-1], dtype=np.int32)
        for index, term in enumerate(terms):
            posting = self.postings[term]
            span = slice(offsets[index], offsets[index + 1])
            doc_pos[span] = [position[identifier] for identifier in posting]
            frequency[span] = list(posting.values())
        return {
            "ids": np.frombuffer("\n".join(ids).encode("utf-8"), dtype=np.uint8),
            "terms": np.frombuffer("\n".join(terms).encode("utf-8"), dtype=np.uint8),
            "offsets": offsets, "doc_pos": doc_pos, "tf": frequency,
            "lengths": np.array([self.lengths[key] for key in ids], dtype=np.int64),
            "unique_terms": np.array([self.unique_terms[key] for key in ids], dtype=np.int64),
            "norms": np.array([self.norms.get(key, np.nan) for key in ids], dtype=np.float64),
            "tfidf_idf": np.array([self.tfidf_idf[term] for term in terms], dtype=np.float64),
            "bm25_idf": np.array([self.bm25_idf[term] for term in terms], dtype=np.float64),
            "mean_length": np.array(self.mean_length, dtype=np.float64),
        }

    @classmethod
    def from_arrays(cls, arrays):
        self = cls.__new__(cls)
        ids = arrays["ids"].tobytes().decode("utf-8").split("\n")
        terms = arrays["terms"].tobytes().decode("utf-8").split("\n")
        self.postings = CompactPostings(ids, terms, arrays["offsets"], arrays["doc_pos"], arrays["tf"])
        self.lengths = dict(zip(ids, arrays["lengths"].tolist()))
        self.unique_terms = dict(zip(ids, arrays["unique_terms"].tolist()))
        self.norms = {key: value for key, value in zip(ids, arrays["norms"].tolist()) if value == value}
        self.tfidf_idf = dict(zip(terms, arrays["tfidf_idf"].tolist()))
        self.bm25_idf = dict(zip(terms, arrays["bm25_idf"].tolist()))
        self.mean_length = float(arrays["mean_length"])
        return self


class CompactPostings:
    """Mapa término → {documento: frecuencia} respaldado por arrays; solo lo que usa `search`."""

    def __init__(self, ids, terms, offsets, doc_pos, frequency):
        self.ids = ids
        self.term_index = {term: index for index, term in enumerate(terms)}
        self.offsets, self.doc_pos, self.frequency = offsets, doc_pos, frequency

    def __len__(self):
        return len(self.term_index)

    def __contains__(self, term):
        return term in self.term_index

    def __getitem__(self, term):
        index = self.term_index[term]
        span = slice(self.offsets[index], self.offsets[index + 1])
        ids = self.ids
        return dict(zip((ids[position] for position in self.doc_pos[span].tolist()),
                        self.frequency[span].tolist()))

    def get(self, term, default=None):
        return self[term] if term in self.term_index else default


def faiss_path(documents, directory):
    """Ruta del índice FAISS de este corpus y modelo: el nombre lleva el hash del contenido."""
    #ruta del archivo que esta guardado enel disk usa el nombre modelo y el directorio corupus
    digest = hashlib.sha256(f"v1:{MODEL}:{MODEL_REVISION}".encode())
    for identifier in sorted(documents):
        for field in (identifier, document_text(documents[identifier])):
            data = field.encode("utf-8")
            digest.update(len(data).to_bytes(8, "big"))
            digest.update(data)
    return directory / f"documents-{digest.hexdigest()}.faiss"
# esa funcion devuleve la ruta


class SemanticIndex:
    def __init__(self, documents, directory):
        import faiss
        import torch
        from sentence_transformers import SentenceTransformer

        self.encoder = None
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        faiss.omp_set_num_threads(1)
        self.ids = sorted(documents)
        self.model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
        dimension = self.model.get_sentence_embedding_dimension()
        path = faiss_path(documents, directory)
        if path.exists():
            self.index = faiss.read_index(str(path)) #llama esta ruta para leer el archivo que esta guardado en el disk
            if (self.index.ntotal != len(self.ids) or self.index.d != dimension
                    or self.index.metric_type != faiss.METRIC_INNER_PRODUCT): # lo carga y lo guarda en memoria
                raise ValueError(f"Índice FAISS incompatible: {path}")
            print(f"Índice FAISS reutilizado: {path}", file=sys.stderr)
            return #lo usamos para buscar los vectroes mas parecidos en la consulta

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

    @classmethod
    def open(cls, ids, path, encoder):
        """Abre un índice FAISS ya construido sin cargar el corpus ni PyTorch (despliegue web).

        `encoder` debe ser el mismo modelo y revisión con que se indexó (ver `onnx_encoder`).
        """
        import faiss

        faiss.omp_set_num_threads(1)
        self = cls.__new__(cls)
        self.ids, self.encoder, self.model = ids, encoder, None
        self.index = faiss.read_index(str(path))
        if (self.index.ntotal != len(ids) or self.index.d != encoder.dimension
                or self.index.metric_type != faiss.METRIC_INNER_PRODUCT):
            raise ValueError(f"Índice FAISS incompatible: {path}")
        return self

    def encode(self, texts):
        if self.encoder is not None:
            return self.encoder.encode(texts)
        return self.model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )

    def search_many(self, queries, limit):
        # Buscar todos permite aplicar el mismo desempate también en el límite del top-n.
        # La búsqueda de un solo query es más rápida, pero el desempate se hace por query.
        for offset in range(0, len(queries), 32):
            vectors = self.encode(queries[offset:offset + 32]) # convvierte la consulta en un vector
            scores, positions = self.index.search(vectors, len(self.ids))# faiss usa el vector para encotrar los doc mas cercanos
            for row_scores, row_positions in zip(scores, positions, strict=True):
                hits = ((self.ids[position], float(score))
                        for score, position in zip(row_scores, row_positions, strict=True))
                yield nsmallest(limit, hits, key=lambda hit: (-hit[1], hit[0]))
# para convertir la consulta en un vct buscamos los vct mas cercanos
#la consuta se convierte en un VCT y este mismo los buscamos en los vectores de los docuemento y ver cual son los mas parecidos

