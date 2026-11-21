"""Motor de Lente: carga una sola vez los artefactos y responde consultas.

Toda la recuperación sigue en retrieval.py (`InvertedIndex.search`, `SemanticIndex.search_many`).
Aquí solo se abren los artefactos precalculados (ver build_artifacts.py) y se presentan resultados:

* arranque: leer lexical.npz, abrir el índice FAISS de 57 638 vectores, crear la sesión ONNX;
  no se recalculan embeddings de documentos ni se reconstruye el índice léxico;
* por consulta nueva: un solo embedding (el de la consulta) y la búsqueda.
"""

import json
import mmap
import os
from pathlib import Path
import time
import zlib

import numpy as np

from onnx_encoder import OnnxEncoder
from retrieval import METHODS, InvertedIndex, SemanticIndex

ARTIFACTS = Path(os.environ.get("LENTE_ARTIFACTS", Path(__file__).parent / "artifacts"))
LABELS = {"jaccard": "Jaccard", "tfidf": "TF-IDF", "bm25": "BM25", "semantic": "Semántico"}
FAMILIES = {"jaccard": "Binario", "tfidf": "Vectorial (TF-IDF)", "bm25": "Probabilístico",
            "semantic": "Vectorial (embeddings)"}
MAX_TEXT = 6000
SNIPPET = 380
PAIRS = (("semantic", "bm25"), ("semantic", "tfidf"), ("bm25", "tfidf"))


def snippet(text, limit=SNIPPET):
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + "…"


def normalize_query(text):
    return " ".join(text.split()).casefold()


def memory_mb():
    """Memoria residente actual y pico del proceso (Linux); None donde no hay /proc."""
    try:
        values = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS", "VmHWM")):
                values[line.split(":")[0]] = round(int(line.split()[1]) / 1024)
        return {"rss": values.get("VmRSS"), "peak": values.get("VmHWM")}
    except OSError:
        return None


class Engine:
    def __init__(self, directory=ARTIFACTS):
        self.directory = Path(directory)
        steps, last = {}, time.perf_counter()
        started = last

        def mark(name):
            nonlocal last
            now = time.perf_counter()
            steps[name] = round(now - last, 3)
            last = now

        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        for item in self.manifest["files"]:   # existencia y tamaño; el hash lo comprueba build_artifacts
            path = self.directory / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise FileNotFoundError(f"Artefacto ausente o alterado: {item['path']}")
        mark("manifiesto")

        arrays = dict(np.load(self.directory / "lexical.npz"))
        self.lexical = InvertedIndex.from_arrays(arrays)
        self.ids = self.lexical.postings.ids
        self.position = {identifier: index for index, identifier in enumerate(self.ids)}
        mark("indice_lexico")

        offsets = np.load(self.directory / "docs.offsets.npy")
        if len(offsets) != len(self.ids) + 1:
            raise ValueError("docs.bin no corresponde al índice léxico")
        self.doc_offsets = offsets.tolist()
        self._docs_file = (self.directory / "docs.bin").open("rb")
        self._docs = mmap.mmap(self._docs_file.fileno(), 0, access=mmap.ACCESS_READ)
        mark("documentos")

        model = self.directory / "model"
        self.encoder = OnnxEncoder(model / "model.onnx", model / "vocab.txt",
                                   threads=max(1, min(2, os.cpu_count() or 1)))
        mark("modelo_onnx")

        self.semantic = SemanticIndex.open(self.ids, self.directory / self.manifest["faiss"], self.encoder)
        mark("indice_faiss")

        self.report = json.loads((self.directory / "report_k10.json").read_text(encoding="utf-8"))
        judged = json.loads((self.directory / "judgments.json").read_text(encoding="utf-8"))
        self.judged = {normalize_query(item["text"]): (key, set(item["relevant"]))
                       for key, item in judged.items()}
        self.relevant_total = sum(len(item["relevant"]) for item in judged.values())
        self._evaluation = self._build_evaluation()
        mark("evaluacion")

        # Calentamiento: recorre el índice y el modelo una vez para que la primera consulta real
        # no pague las lecturas de página ni la inicialización de ONNX Runtime.
        next(self.semantic.search_many(["calentamiento del indice"], 1))
        mark("calentamiento")

        self.load_seconds = round(time.perf_counter() - started, 3)
        self.load_steps = steps

    # --- documentos ------------------------------------------------------------------------

    def document(self, identifier):
        index = self.position[identifier]
        record = json.loads(zlib.decompress(self._docs[self.doc_offsets[index]:self.doc_offsets[index + 1]]))
        text = record["text"]
        truncated = len(text) > MAX_TEXT
        return {"id": identifier, "title": record["title"], "snippet": snippet(text),
                "text": text[:MAX_TEXT] + ("…" if truncated else ""), "truncated": truncated}

    def judgment(self, query):
        return self.judged.get(normalize_query(query))

    # --- búsqueda --------------------------------------------------------------------------

    def _rank(self, query, method, n):
        started = time.perf_counter()
        if method == "semantic":
            hits = next(self.semantic.search_many([query], n))
        else:
            hits = self.lexical.search(query, method, n)
        return hits, round((time.perf_counter() - started) * 1000, 1)

    def search(self, query, method, n):
        hits, took = self._rank(query, method, n)
        judged = self.judgment(query)
        results = []
        for rank, (identifier, score) in enumerate(hits, start=1):
            item = {"rank": rank, "score": score, **self.document(identifier)}
            if judged:
                item["relevant"] = identifier in judged[1]
            results.append(item)
        return {"query": query, "method": method, "label": LABELS[method], "n": n,
                "took_ms": took, "count": len(results), **self._judged_info(judged), "hits": results}

    def compare(self, query, n):
        judged = self.judgment(query)
        methods, ranks, order = [], {}, []
        for method in METHODS:
            hits, took = self._rank(query, method, n)
            methods.append({"id": method, "label": LABELS[method], "family": FAMILIES[method],
                            "took_ms": took,
                            "hits": [{"rank": rank, "id": identifier, "score": score}
                                     for rank, (identifier, score) in enumerate(hits, start=1)]})
            for rank, (identifier, _) in enumerate(hits, start=1):
                if identifier not in ranks:
                    ranks[identifier] = {}
                    order.append(identifier)
                ranks[identifier][method] = rank
        documents = {}
        for identifier in order:
            documents[identifier] = {**self.document(identifier), "ranks": ranks[identifier]}
            if judged:
                documents[identifier]["relevant"] = identifier in judged[1]
        sets = {item["id"]: {hit["id"] for hit in item["hits"]} for item in methods}
        overlap = {a: {b: len(sets[a] & sets[b]) for b in METHODS} for a in METHODS}
        return {"query": query, "n": n, "took_ms": round(sum(item["took_ms"] for item in methods), 1),
                **self._judged_info(judged), "methods": methods, "documents": documents,
                "order": order, "overlap": overlap,
                "in_all_methods": len(set.intersection(*sets.values())), "distinct": len(order)}

    @staticmethod
    def _judged_info(judged):
        if not judged:
            return {"judged": None}
        return {"judged": {"query_id": judged[0], "relevant_total": len(judged[1])}}

    # --- evaluación (valores reales de report_k10.json) -------------------------------------

    def evaluation(self):
        return self._evaluation

    def _build_evaluation(self):
        report = self.report
        count = report["query_count"]
        by_method, table = [], {}
        for method in METHODS:
            rows = report["methods"][method]["queries"]
            if len(rows) != count:
                raise ValueError(f"report_k10.json incompleto para {method}")
            mean_ap = sum(row["ap"] for row in rows) / count
            if abs(mean_ap - report["methods"][method]["map"]) > 1e-12:
                raise ValueError(f"MAP inconsistente en report_k10.json para {method}")
            by_method.append({
                "id": method, "label": LABELS[method], "family": FAMILIES[method],
                "map": report["methods"][method]["map"],
                "mean_precision": sum(row["precision"] for row in rows) / count,
                "mean_recall": sum(row["recall"] for row in rows) / count,
                "queries_with_hit": sum(1 for row in rows if row["recall"] > 0),
            })
            for row in rows:
                entry = table.setdefault(row["query_id"], {"id": row["query_id"], "text": row["query"]})
                entry[method] = [round(row["precision"], 6), round(row["recall"], 6), round(row["ap"], 6)]
        queries = list(table.values())
        head_to_head = []
        for a, b in PAIRS:
            ap = lambda entry, m: entry[m][2]  # noqa: E731
            better = sum(1 for q in queries if ap(q, a) > ap(q, b) + 1e-12)
            worse = sum(1 for q in queries if ap(q, b) > ap(q, a) + 1e-12)
            both_zero = sum(1 for q in queries if ap(q, a) == 0 and ap(q, b) == 0)
            head_to_head.append({"a": a, "b": b, "a_better": better, "b_better": worse,
                                 "tie": count - better - worse, "both_zero": both_zero})
        return {"k": report["k"], "documents": report["documents"], "query_count": count,
                "relevant_total": self.relevant_total,
                "source": "report_k10.json", "methods": by_method,
                "head_to_head": head_to_head, "queries": queries}

    # --- estado ----------------------------------------------------------------------------

    def info(self):
        return {"documents": len(self.ids), "model": self.manifest["model"],
                "corpus": self.manifest["corpus"], "faiss_vectors": self.semantic.index.ntotal,
                "load_seconds": self.load_seconds, "load_steps": self.load_steps}
