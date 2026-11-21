"""Pruebas de la API y del motor de Lente (parte 2). Requieren los artefactos generados:

    python build_artifacts.py
    python -m unittest test_lente -v
"""

import json
from pathlib import Path
import unittest
from unittest import mock

from fastapi.testclient import TestClient

import app as lente_app
from lente_engine import Engine
from main import metrics
import retrieval

REPORT = json.loads(Path("data/fiqa/report_k10.json").read_text(encoding="utf-8"))
HOME_OFFICE = "can I deduct a home office if I work remotely as a freelancer"
METHODS = ("jaccard", "tfidf", "bm25", "semantic")


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(lente_app.app)
        cls.client.__enter__()          # ejecuta el lifespan: carga los artefactos una vez
        cls.engine = lente_app.get_engine()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def get(self, path, **params):
        return self.client.get(path, params=params)

    def test_health_reports_loaded_engine(self):
        body = self.get("/api/health").json()
        self.assertTrue(body["ready"])
        self.assertEqual(body["documents"], 57638)
        self.assertEqual(body["faiss_vectors"], 57638)
        self.assertEqual(body["model"]["revision"], retrieval.MODEL_REVISION)

    def test_each_method_returns_ordered_top_n(self):
        for method in METHODS:
            with self.subTest(method=method):
                body = self.get("/api/search", q=HOME_OFFICE, method=method, n=7).json()
                self.assertEqual(body["count"], 7)
                self.assertEqual([hit["rank"] for hit in body["hits"]], list(range(1, 8)))
                scores = [hit["score"] for hit in body["hits"]]
                self.assertEqual(scores, sorted(scores, reverse=True))
                self.assertTrue(all(hit["text"] and hit["snippet"] for hit in body["hits"]))

    def test_semantic_matches_part_one_results(self):
        body = self.get("/api/search", q=HOME_OFFICE, method="semantic", n=3).json()
        self.assertEqual([hit["id"] for hit in body["hits"]], ["295509", "288537", "377621"])
        for hit, expected in zip(body["hits"], (0.7468, 0.6529, 0.6165)):
            self.assertAlmostEqual(hit["score"], expected, places=4)

    def test_default_method_is_semantic(self):
        self.assertEqual(self.get("/api/search", q="stock options").json()["method"], "semantic")

    def test_compare_is_consistent_with_search(self):
        body = self.get("/api/compare", q=HOME_OFFICE, n=10).json()
        self.assertEqual([m["id"] for m in body["methods"]], list(METHODS))
        for method in body["methods"]:
            single = self.get("/api/search", q=HOME_OFFICE, method=method["id"], n=10).json()
            self.assertEqual([h["id"] for h in method["hits"]], [h["id"] for h in single["hits"]])
            for hit in method["hits"]:      # posición en el documento = posición en la lista del método
                self.assertEqual(body["documents"][hit["id"]]["ranks"][method["id"]], hit["rank"])
        self.assertEqual(body["distinct"], len(body["documents"]))
        self.assertEqual(body["overlap"]["bm25"]["semantic"], body["overlap"]["semantic"]["bm25"])

    def test_empty_results_for_stopword_only_lexical_query(self):
        for method in ("jaccard", "tfidf", "bm25"):
            self.assertEqual(self.get("/api/search", q="the of a", method=method).json()["count"], 0)
        self.assertEqual(self.get("/api/search", q="the of a", method="semantic", n=5).json()["count"], 5)

    def test_validation_errors_are_json_and_in_spanish(self):
        cases = [dict(q=""), dict(q="x" * 301), dict(q="ok", n=0), dict(q="ok", n=51), dict(q="ok", method="tf")]
        for params in cases:
            with self.subTest(params=params):
                response = self.get("/api/search", **params)
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["error"]["code"], "invalid_request")
        self.assertEqual(self.get("/api/search", q="   ").status_code, 422)

    def test_unicode_and_odd_queries_do_not_crash(self):
        for query in ("¿Cómo declarar impuestos? 日本語 😀", "[CLS] [SEP]", "͸", "a" * 300, "%$&?=#"):
            with self.subTest(query=query):
                self.assertEqual(self.get("/api/search", q=query, n=3).status_code, 200)

    def test_judged_query_marks_relevant_documents(self):
        text = REPORT["methods"]["semantic"]["queries"][3]["query"]
        body = self.get("/api/search", q=text.lower(), method="semantic", n=10).json()
        self.assertIsNotNone(body["judged"])
        self.assertTrue(all("relevant" in hit for hit in body["hits"]))
        self.assertIsNone(self.get("/api/search", q="a query outside FiQA test", n=3).json()["judged"])

    def test_evaluation_values_come_from_report(self):
        body = self.get("/api/evaluation").json()
        self.assertEqual(body["k"], 10)
        self.assertEqual(body["query_count"], 648)
        self.assertEqual(len(body["queries"]), 648)
        for method in body["methods"]:
            self.assertEqual(method["map"], REPORT["methods"][method["id"]]["map"])
        row = body["queries"][0]
        source = REPORT["methods"]["bm25"]["queries"][0]
        self.assertEqual(row["bm25"], [round(source["precision"], 6), round(source["recall"], 6), round(source["ap"], 6)])
        self.assertIn("max-age", self.client.get("/api/evaluation").headers["cache-control"])


class ReuseTests(unittest.TestCase):
    """Los artefactos se cargan una vez y solo se embebe la consulta nueva."""

    @classmethod
    def setUpClass(cls):
        cls.engine = Engine()

    def test_no_index_rebuild_and_one_embedding_per_query(self):
        calls = []
        real_encode = self.engine.encoder.encode

        def counting_encode(texts, *args, **kwargs):
            calls.append(len(texts))
            return real_encode(texts, *args, **kwargs)

        with mock.patch.object(retrieval.InvertedIndex, "__init__", side_effect=AssertionError("reconstruye el índice")), \
                mock.patch.object(retrieval.SemanticIndex, "__init__", side_effect=AssertionError("reconstruye FAISS")), \
                mock.patch.object(self.engine.encoder, "encode", side_effect=counting_encode):
            for method in ("jaccard", "tfidf", "bm25"):
                self.engine.search("deduct home office", method, 5)
            self.assertEqual(calls, [], "las búsquedas léxicas no deben codificar nada")
            self.engine.search("deduct home office", "semantic", 5)
            self.assertEqual(calls, [1])
            self.engine.compare("deduct home office", 5)
            self.assertEqual(calls, [1, 1])      # comparar = un solo embedding de la consulta

    def test_semantic_ranking_reproduces_report_metrics(self):
        """El camino de producción (ONNX + FAISS) reproduce las métricas por consulta del informe."""
        judged = json.loads((self.engine.directory / "judgments.json").read_text(encoding="utf-8"))
        rows = REPORT["methods"]["semantic"]["queries"][:60]
        identical = 0
        for row in rows:
            relevant = set(judged[row["query_id"]]["relevant"])
            hits, _ = self.engine._rank(row["query"], "semantic", 10)
            got = metrics(hits, relevant, 10)
            identical += abs(got["ap"] - row["ap"]) < 1e-9
            self.assertAlmostEqual(got["recall"], row["recall"], places=9)
        self.assertGreaterEqual(identical, len(rows) - 1)

    def test_lexical_ranking_reproduces_report_metrics_exactly(self):
        judged = json.loads((self.engine.directory / "judgments.json").read_text(encoding="utf-8"))
        for method in ("jaccard", "tfidf", "bm25"):
            for row in REPORT["methods"][method]["queries"][:80]:
                relevant = set(judged[row["query_id"]]["relevant"])
                hits, _ = self.engine._rank(row["query"], method, 10)
                self.assertEqual(metrics(hits, relevant, 10), {k: row[k] for k in ("precision", "recall", "ap")})


if __name__ == "__main__":
    unittest.main()
