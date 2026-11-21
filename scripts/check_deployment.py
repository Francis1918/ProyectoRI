"""Comprueba una URL desplegada contra el motor local: mismas posiciones y puntuaciones.

    python scripts/check_deployment.py https://lente-fiqa.vercel.app
    python scripts/check_deployment.py https://lente-fiqa.vercel.app "consulta nueva" --n 10

Solo usa la biblioteca estándar para llamar a la API; el motor local necesita los artefactos.
"""

import argparse
import json
import sys
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, ".")
from lente_engine import LABELS, Engine  # noqa: E402

DEFAULT_QUERY = "how should two roommates split rent and utilities when filing taxes separately"


def fetch(base, path, params=None):
    url = f"{base.rstrip('/')}{path}" + (f"?{urlencode(params)}" if params else "")
    started = time.perf_counter()
    with urlopen(Request(url, headers={"Accept": "application/json", "User-Agent": "lente-check/1.0"}), timeout=120) as response:
        body = json.load(response)
    return body, round((time.perf_counter() - started) * 1000)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument("--n", type=int, default=10)
    args = parser.parse_args()

    local = Engine()
    ok = True
    print(f"consulta: {args.query!r}  (n={args.n})")
    for method in LABELS:
        remote, wall_ms = fetch(args.base, "/api/search", {"q": args.query, "method": method, "n": args.n})
        mine = local.search(args.query, method, args.n)
        same_ids = [h["id"] for h in remote["hits"]] == [h["id"] for h in mine["hits"]]
        worst = max((abs(a["score"] - b["score"]) for a, b in zip(remote["hits"], mine["hits"])), default=0)
        ok &= same_ids and worst < 1e-5 and remote["count"] == args.n
        print(f"  {LABELS[method]:<10} {remote['count']:>2} resultados | servidor {remote['took_ms']:>6} ms | "
              f"total {wall_ms:>5} ms | mismas posiciones que local: {same_ids} | |Δscore| máx {worst:.1e}")
    remote, wall_ms = fetch(args.base, "/api/compare", {"q": args.query, "n": args.n})
    mine = local.compare(args.query, args.n)
    same = all([h["id"] for h in a["hits"]] == [h["id"] for h in b["hits"]]
               for a, b in zip(remote["methods"], mine["methods"])) and remote["overlap"] == mine["overlap"]
    ok &= same
    print(f"  comparar   {remote['distinct']} documentos distintos, {remote['in_all_methods']} en los cuatro | "
          f"servidor {remote['took_ms']} ms | total {wall_ms} ms | igual que local: {same}")
    report = json.loads(open("data/fiqa/report_k10.json", encoding="utf-8").read()) if __import__("os").path.exists("data/fiqa/report_k10.json") else None
    evaluation, wall_ms = fetch(args.base, "/api/evaluation")
    summary = {m["id"]: (round(m["map"], 4), round(m["mean_precision"], 4), round(m["mean_recall"], 4)) for m in evaluation["methods"]}
    print(f"  evaluación k={evaluation['k']}, {evaluation['query_count']} consultas, {len(evaluation['queries'])} filas | total {wall_ms} ms")
    for method, (map_, precision, recall) in summary.items():
        print(f"    {LABELS[method]:<10} MAP@10 {map_}  P@10 {precision}  R@10 {recall}")
    if report:
        same_report = all(abs(report["methods"][m]["map"] - evaluation_m["map"]) < 1e-12
                          for m, evaluation_m in ((x["id"], x) for x in evaluation["methods"]))
        ok &= same_report
        print(f"    MAP idéntico al de data/fiqa/report_k10.json: {same_report}")
    print("RESULTADO:", "OK" if ok else "HAY DIFERENCIAS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
