"""Recalcula MAP@10, Precision y Recall con el camino de PRODUCCIÓN (artefactos + ONNX + FAISS) y
los compara, consulta por consulta, con data/fiqa/report_k10.json (parte 1, PyTorch).

    python verify_engine.py
"""

import json
from pathlib import Path
import sys

from lente_engine import Engine
from main import metrics

report = json.loads(Path("data/fiqa/report_k10.json").read_text(encoding="utf-8"))
engine = Engine()
judged = json.loads((engine.directory / "judgments.json").read_text(encoding="utf-8"))
ok = True
print(f"{'método':<10}{'MAP prod.':>11}{'MAP informe':>13}{'consultas con AP idéntico':>28}{'|ΔAP| máx':>12}")
for method in ("jaccard", "tfidf", "bm25", "semantic"):
    rows = report["methods"][method]["queries"]
    same, worst, total = 0, 0.0, 0.0
    for row in rows:
        hits, _ = engine._rank(row["query"], method, report["k"])
        got = metrics(hits, set(judged[row["query_id"]]["relevant"]), report["k"])
        same += got == {k: row[k] for k in ("precision", "recall", "ap")}
        worst = max(worst, abs(got["ap"] - row["ap"]))
        total += got["ap"]
    print(f"{method:<10}{total / len(rows):>11.6f}{report['methods'][method]['map']:>13.6f}"
          f"{f'{same}/{len(rows)}':>28}{worst:>12.2e}")
    ok &= same == len(rows) if method != "semantic" else abs(total / len(rows) - report["methods"][method]["map"]) < 5e-4
print("RESULTADO:", "OK" if ok else "HAY DIFERENCIAS")
sys.exit(0 if ok else 1)
