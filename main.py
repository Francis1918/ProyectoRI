"""Buscador local de FiQA. Ejecutar con Python 3.13."""

import argparse
import json
from pathlib import Path

from corpus import load_documents, load_evaluation, prepare
from retrieval import LEXICAL, METHODS, InvertedIndex, SemanticIndex


def metrics(hits, relevant, k):
    found = 0
    precision_sum = 0.0
    for rank, (identifier, _) in enumerate(hits[:k], start=1):
        if identifier in relevant:
            found += 1
            precision_sum += found / rank
    return {
        "precision": found / k,
        "recall": found / len(relevant),
        "ap": precision_sum / len(relevant),
    }


def evaluate(directory, documents, k):
    queries, relevant = load_evaluation(directory, documents)
    query_ids = sorted(relevant)
    texts = [queries[key]["text"] for key in query_ids]
    lexical = InvertedIndex(documents)
    semantic = SemanticIndex(documents, directory)
    report = {"k": k, "documents": len(documents), "query_count": len(texts), "methods": {}}
    for method in METHODS:
        if method == "semantic":
            rankings = semantic.search_many(texts, k)
        else:
            rankings = (lexical.search(query, method, k) for query in texts)
        rows = [
            {"query_id": key, "query": queries[key]["text"],
             **metrics(hits, relevant[key], k)}
            for key, hits in zip(query_ids, rankings, strict=True)
        ]
        report["methods"][method] = {
            "map": sum(row["ap"] for row in rows) / len(rows),
            "queries": rows,
        }
    return report


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("se requiere un entero positivo")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fiqa"), help="directorio de datos")
    commands = parser.add_subparsers(dest="command", required=True)
    preparation = commands.add_parser("prepare", help="descargar y validar FiQA")
    preparation.add_argument("--sample-queries", type=positive)
    preparation.add_argument("--distractors", type=positive, default=100)
    commands.add_parser("index", help="precalcular el índice FAISS")
    search = commands.add_parser("search", help="buscar una consulta libre")
    search.add_argument("query")
    search.add_argument("--method", choices=METHODS, required=True)
    search.add_argument("--top", type=positive, default=10)
    evaluation = commands.add_parser("evaluate", help="evaluar los cuatro métodos")
    evaluation.add_argument("--k", type=positive, default=10)
    evaluation.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        if args.command == "prepare":
            prepare(args.data, args.sample_queries, args.distractors)
            return
        documents = load_documents(args.data)
        if not documents:
            raise ValueError("El corpus está vacío")
        if args.command == "index":
            SemanticIndex(documents, args.data)
            return
        if args.command == "search":
            if not args.query.strip():
                raise ValueError("La consulta está vacía")
            if args.method in LEXICAL:
                hits = InvertedIndex(documents).search(args.query, args.method, args.top)
            else:
                index = SemanticIndex(documents, args.data)
                hits = next(index.search_many([args.query], args.top))
            result = [
                {"rank": rank, "id": key, "score": score, "title": documents[key]["title"],
                 "text": documents[key]["text"]}
                for rank, (key, score) in enumerate(hits, 1)
            ]
        else:
            result = evaluate(args.data, documents, args.k)
            if args.output:
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
                result = {method: values["map"] for method, values in result["methods"].items()}
                result = {"k": args.k, "map": result, "report": str(args.output)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except FileNotFoundError as error:
        parser.exit(1, f"Archivo ausente: {error.filename}. Ejecuta prepare con el mismo --data.\n")
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
