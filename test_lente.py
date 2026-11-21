"""Pruebas de la API y del motor de Lente (parte 2). Requieren los artefactos generados:

    python build_artifacts.py
    python -m unittest test_lente -v
"""

import json
from pathlib import Path
import random
import unittest
from unittest import mock

import numpy as np
from fastapi.testclient import TestClient

import app as lente_app
from lente_engine import Engine
from main import metrics
import retrieval
import spelling

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

    # --- sugerencia ortográfica en «Sin resultados» (solo métodos léxicos) --------------------------

    LEXICAL = ("jaccard", "tfidf", "bm25")

    def test_misspelling_suggests_correction_in_lexical_empty_state(self):
        for method in self.LEXICAL:
            with self.subTest(method=method):
                body = self.get("/api/search", q="moretgage", method=method, n=10).json()
                self.assertEqual((body["count"], body["hits"]), (0, []))
                self.assertEqual(body["suggestion"]["query"], "mortgage")
                self.assertEqual(body["suggestion"]["changes"][0]["from"], "moretgage")
                self.assertEqual(body["query"], "moretgage")      # la consulta original no se toca

    def test_suggested_query_runs_with_same_method_and_top_n(self):
        for method in self.LEXICAL:
            with self.subTest(method=method):
                first = self.get("/api/search", q="moretgage", method=method, n=7).json()
                again = self.get("/api/search", q=first["suggestion"]["query"], method=method, n=7).json()
                self.assertEqual((again["method"], again["n"], again["count"]), (method, 7, 7))
                self.assertIsNone(again["suggestion"])

    def test_correct_word_gives_normal_results_without_suggestion(self):
        for method in METHODS:
            with self.subTest(method=method):
                body = self.get("/api/search", q="mortgage", method=method, n=5).json()
                self.assertEqual(body["count"], 5)
                self.assertIsNone(body["suggestion"])

    def test_misspelling_that_exists_in_the_corpus_is_not_an_empty_state(self):
        """«morgage» aparece en 5 documentos de FiQA: la búsqueda léxica la encuentra y no hay «Sin
        resultados». El corrector, por sí solo, sí propone «mortgage» (se enchufa solo en el estado vacío)."""
        for method in self.LEXICAL:
            body = self.get("/api/search", q="morgage", method=method, n=10).json()
            self.assertEqual(body["count"], 5)
            self.assertIsNone(body["suggestion"])
        self.assertEqual(self.engine.speller.suggest("morgage")["query"], "mortgage")

    def test_no_suggestion_without_a_clear_candidate(self):
        for query in ("xqzvwkjh", "asdfghjkl", "the of a", "moretgage xqzvwkjh"):
            for method in self.LEXICAL:
                with self.subTest(query=query, method=method):
                    body = self.get("/api/search", q=query, method=method).json()
                    self.assertEqual(body["count"], 0)
                    self.assertIsNone(body["suggestion"])

    def test_semantic_never_has_a_suggestion_and_keeps_its_flow(self):
        body = self.get("/api/search", q="moretgage", method="semantic", n=5).json()
        self.assertEqual(body["count"], 5)
        self.assertIsNone(body["suggestion"])

    def test_speller_is_not_consulted_for_non_empty_results_or_semantic(self):
        with mock.patch.object(self.engine.speller, "suggest", side_effect=AssertionError("no debe consultarse")):
            for method in METHODS:
                self.assertEqual(self.get("/api/search", q=HOME_OFFICE, method=method, n=3).json()["count"], 3)
            self.assertEqual(self.get("/api/search", q="moretgage", method="semantic", n=3).json()["count"], 3)
            self.assertEqual(self.get("/api/compare", q="moretgage", n=3).status_code, 200)

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


class SpellingTests(unittest.TestCase):
    """Corrector ortográfico con un vocabulario pequeño y frecuencias controladas (sin artefactos)."""

    TERMS = {"mortgage": 2000, "mortgages": 480, "payment": 2700, "payments": 900, "stock": 5000, "stop": 100,
             "tax": 6000, "taxes": 3700, "investment": 4300, "receive": 2000, "zebrafish": 3, "wit": 50,
             "bank": 900, "back": 900, "text": 300, "next": 280, "recieve": 10}

    @classmethod
    def setUpClass(cls):
        cls.speller = spelling.Speller(list(cls.TERMS), np.array(list(cls.TERMS.values())))

    def fix(self, query):
        found = self.speller.suggest(query)
        return found and found["query"]

    def test_edit_distance_counts_insertion_omission_substitution_and_transposition(self):
        cases = [("moretgage", "mortgage", 1),    # inserción de una letra
                 ("morgage", "mortgage", 1),      # omisión
                 ("mortgaje", "mortgage", 1),     # sustitución
                 ("mortgaeg", "mortgage", 1),     # transposición de dos letras contiguas = 1 edición
                 ("ab", "ba", 1), ("kitten", "sitting", 3), ("", "abc", 3), ("same", "same", 0)]
        for a, b, expected in cases:
            with self.subTest(a=a, b=b):
                self.assertEqual(spelling.edit_distance(a, b), expected)
                self.assertEqual(spelling.edit_distance(b, a), expected)

    def test_vectorized_distance_matches_reference_implementation(self):
        rng = random.Random(3)
        for length in (3, 5, 8):
            words = ["".join(rng.choice("abcd") for _ in range(length)) for _ in range(200)]   # alfabeto corto: muchas coincidencias y transposiciones
            columns = np.ascontiguousarray(np.frombuffer("".join(words).encode(), dtype=np.uint8).reshape(len(words), length).T)
            for probe in ("".join(rng.choice("abcd") for _ in range(rng.randint(2, 10))) for _ in range(15)):
                expected = [spelling.edit_distance(probe, word) for word in words]
                self.assertEqual(spelling._distances(probe, columns).tolist(), expected)

    def test_each_kind_of_typo_is_corrected(self):
        for typo in ("moretgage", "morgage", "mortgaje", "mortgaeg", "investmnet"):
            with self.subTest(typo=typo):
                self.assertEqual(self.fix(typo), "mortgage" if typo.startswith("mo") else "investment")

    def test_reports_distance_and_frequency_of_the_target(self):
        change = self.speller.suggest("moretgage")["changes"][0]
        self.assertEqual(change, {"from": "moretgage", "to": "mortgage", "distance": 1, "documents": 2000})

    def test_known_word_and_non_words_are_left_alone(self):
        for query in ("mortgage", "xqzvwkjh", "asdfghjkl", "", "   ", "the of a", "mortgage2", "1234", "日本語"):
            with self.subTest(query=query):
                self.assertIsNone(self.fix(query))

    def test_ambiguous_candidates_give_no_suggestion(self):
        self.assertIsNone(self.fix("banck"))     # bank y back a 1 edición y con la misma frecuencia
        self.assertIsNone(self.fix("uext"))      # text y next: casi empate
        self.assertEqual(self.fix("stok"), "stock")                      # stock (5000) domina a stop (100)

    def test_target_must_appear_in_enough_documents(self):
        self.assertIsNone(self.fix("zebrafsh"))  # «zebrafish» solo está en 3 documentos: no es evidencia suficiente

    def test_closer_candidate_wins_when_frequencies_are_comparable(self):
        # «paymnts» está a 1 de «payments» (900) y a 2 de «payment» (2700): gana la más cercana
        self.assertEqual(self.fix("paymnts"), "payments")

    def test_much_more_frequent_word_beats_a_rare_closer_one(self):
        # «recievee» está a 1 de «recieve» (una errata que existe en 10 documentos) y a 2 de «receive» (2000)
        self.assertEqual(spelling.edit_distance("recievee", "recieve"), 1)
        self.assertEqual(spelling.edit_distance("recievee", "receive"), 2)
        self.assertEqual(self.fix("recievee"), "receive")

    def test_short_words_are_not_corrected(self):
        self.assertIsNone(self.fix("tac"))       # a 1 edición de «tax»

    def test_words_close_to_a_stopword_are_ambiguous(self):
        self.assertIsNone(self.fix("wiht"))      # «with» es palabra vacía; «wit» no basta como destino

    def test_multi_word_queries_keep_the_rest_and_need_every_word_corrected(self):
        self.assertEqual(self.fix("moretgage paymnt"), "mortgage payment")
        self.assertEqual(self.fix("the moretgage"), "the mortgage")          # las palabras vacías se conservan
        self.assertEqual(self.fix("Moretgage?"), "mortgage?")                # mayúsculas y signos, como `tokenize`
        self.assertIsNone(self.fix("moretgage xqzvwkjh"))                    # nunca una corrección a medias
        self.assertIsNone(self.fix(" ".join(["moretgage"] * (spelling.MAX_WORDS + 1))))

    def test_dictionary_is_built_lazily_and_only_once(self):
        fresh = spelling.Speller(list(self.TERMS), np.array(list(self.TERMS.values())))
        self.assertIsNone(fresh._groups)
        fresh.suggest("moretgage")
        built = fresh._groups
        fresh.suggest("morgage")
        self.assertIs(fresh._groups, built)


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
