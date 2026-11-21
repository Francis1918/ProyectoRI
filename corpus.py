"""Carga de FiQA y exportación local en JSON Lines UTF-8."""

from collections import defaultdict
import json
from pathlib import Path
import random
from tempfile import NamedTemporaryFile

DATASET = "BeIR/fiqa"


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
    ) as output:
        temporary = Path(output.name)
        try:
            for row in rows:
                print(json.dumps(row, ensure_ascii=False), file=output)
            output.close()
            temporary.replace(path)
        finally:
            output.close()
            temporary.unlink(missing_ok=True)


def read_jsonl(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def records(rows):
    indexed = {}
    for row in rows:
        identifier = str(row["_id"])
        if identifier in indexed:
            raise ValueError(f"ID duplicado: {identifier}")
        indexed[identifier] = row
    return dict(sorted(indexed.items()))


def load_documents(directory):
    return records(read_jsonl(directory / "corpus.jsonl"))


def load_evaluation(directory, documents):
    queries = records(read_jsonl(directory / "queries.jsonl"))
    judgments = read_jsonl(directory / "qrels.jsonl")
    return queries, validate_qrels(documents, queries, judgments)


def validate_qrels(documents, queries, judgments):
    relevant = defaultdict(set)
    for row in judgments:
        query_id = str(row["query-id"])
        document_id = str(row["corpus-id"])
        if query_id not in queries or document_id not in documents:
            raise ValueError(f"Qrel sin referencia válida: {query_id} → {document_id}")
        if row["score"] > 0:
            relevant[query_id].add(document_id)
    if not relevant:
        raise ValueError("El conjunto de evaluación no tiene qrels positivos")
    return relevant


def normalize_records(rows):
    for row in rows:
        yield {"_id": str(row["_id"]), "title": row.get("title") or "", "text": row["text"]}


def prepare(directory, sample_queries=None, distractors=100):
    from datasets import load_dataset

    documents = records(normalize_records(load_dataset(DATASET, "corpus", split="corpus")))
    queries = records(normalize_records(load_dataset(DATASET, "queries", split="queries")))
    judgments = [
        {"query-id": str(row["query-id"]), "corpus-id": str(row["corpus-id"]), "score": row["score"]}
        for row in load_dataset(f"{DATASET}-qrels", split="test")
    ]
    relevant = validate_qrels(documents, queries, judgments)
    print(f"FiQA validado: {len(documents)} documentos, {len(queries)} consultas totales, "
          f"{len(relevant)} consultas de test y {len(judgments)} qrels compatibles.")

    if sample_queries is not None:
        rng = random.Random(42)
        chosen = set(rng.sample(sorted(relevant), min(sample_queries, len(relevant))))
        judgments = [row for row in judgments if row["query-id"] in chosen]
        required = {row["corpus-id"] for row in judgments}
        pool = sorted(documents.keys() - required)
        retained = required | set(rng.sample(pool, min(distractors, len(pool))))
        documents = {key: documents[key] for key in sorted(retained)}

    query_ids = {row["query-id"] for row in judgments}
    write_jsonl(directory / "corpus.jsonl", documents.values())
    write_jsonl(directory / "queries.jsonl", (queries[key] for key in sorted(query_ids)))
    write_jsonl(directory / "qrels.jsonl", judgments)
    print(f"{directory}: {len(documents)} documentos, {len(query_ids)} consultas, "
          f"{len(judgments)} qrels.")


def document_text(row):
    return "\n".join(part for part in (row.get("title"), row["text"]) if part)
