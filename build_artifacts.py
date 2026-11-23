"""Empaqueta los artefactos que necesita la web (Lente) a partir de lo ya calculado en la parte 1.

No recalcula embeddings ni métricas: copia el índice FAISS de 57 638 vectores y report_k10.json
tal cual, y deriva tres cosas baratas y deterministas:

* lexical.npz   índice invertido en arrays (mismos números que `InvertedIndex`, verificado),
* docs.bin      título y texto de cada documento comprimidos uno a uno (acceso aleatorio),
* model/        el grafo ONNX y vocab.txt de la MISMA revisión de all-MiniLM-L6-v2.

    python build_artifacts.py              # genera artifacts/ (≈ 2 minutos)

`artifacts/` está en .gitignore salvo manifest.json; se sube al desplegar con la CLI de Vercel.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import zlib

import numpy as np

from corpus import document_text, load_documents, load_evaluation
from onnx_encoder import OnnxEncoder
from retrieval import MODEL, MODEL_REVISION, InvertedIndex, SemanticIndex, faiss_path

LEXICAL_CHECK_TOP = 100


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def file_entry(path, base):
    return {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size,
            "sha256": sha256(path)}


def write_docstore(documents, ids, directory):
    offsets = np.zeros(len(ids) + 1, dtype=np.int64)
    with (directory / "docs.bin").open("wb") as output:
        for index, identifier in enumerate(ids):
            row = documents[identifier]
            record = json.dumps({"title": row.get("title") or "", "text": row["text"]},
                                ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            packed = zlib.compress(record, 9)
            output.write(packed)
            offsets[index + 1] = offsets[index] + len(packed)
    np.save(directory / "docs.offsets.npy", offsets)


def check_docstore(documents, ids, directory):
    blob = (directory / "docs.bin").read_bytes()
    offsets = np.load(directory / "docs.offsets.npy")
    for index in range(0, len(ids), 97):
        record = json.loads(zlib.decompress(blob[offsets[index]:offsets[index + 1]]))
        row = documents[ids[index]]
        assert record == {"title": row.get("title") or "", "text": row["text"]}, ids[index]


def check_lexical(live, restored, queries):
    """El índice cargado de arrays debe dar EXACTAMENTE los mismos rankings y puntuaciones."""
    for method in ("jaccard", "tfidf", "bm25"):
        for query in queries:
            if live.search(query, method, LEXICAL_CHECK_TOP) != restored.search(query, method, LEXICAL_CHECK_TOP):
                raise AssertionError(f"Índice léxico distinto en {method}: {query!r}")

#se arma el paquete que vercel se lleva, emqueta el copus los indices el modelo onnx y reporte para la WEB
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fiqa"))
    parser.add_argument("--out", type=Path, default=Path("artifacts"))
    parser.add_argument("--report", type=Path, default=None, help="por defecto <data>/report_k10.json")
    args = parser.parse_args()
    report_path = args.report or args.data / "report_k10.json"
    started = time.perf_counter()

    documents = load_documents(args.data)
    ids = sorted(documents)
    assert list(documents) == ids, "el corpus debe estar ordenado por id"
    queries, relevant = load_evaluation(args.data, documents)
    source_faiss = faiss_path(documents, args.data)
    if not source_faiss.exists():
        sys.exit(f"Falta el índice FAISS {source_faiss}. Ejecuta: python main.py index")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["documents"] == len(documents), "report_k10.json es de otro corpus"
    print(f"corpus: {len(documents):,} documentos; consultas de prueba: {len(relevant)}; k={report['k']}")

    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "model").mkdir(parents=True)

    from huggingface_hub import hf_hub_download

    for name, target in (("onnx/model.onnx", "model.onnx"), ("vocab.txt", "vocab.txt")):
        shutil.copyfile(hf_hub_download(MODEL, name, revision=MODEL_REVISION), out / "model" / target)
    faiss_name = source_faiss.name
    shutil.copyfile(source_faiss, out / faiss_name)
    shutil.copyfile(report_path, out / "report_k10.json")
    print(f"[{time.perf_counter() - started:5.1f}s] modelo ONNX, índice FAISS y report_k10.json copiados")

    live = InvertedIndex(documents)
    arrays = live.export_arrays()
    np.savez_compressed(out / "lexical.npz", **arrays)
    restored = InvertedIndex.from_arrays(dict(np.load(out / "lexical.npz")))
    sample = [queries[key]["text"] for key in sorted(relevant)]
    check_lexical(live, restored, sample)
    print(f"[{time.perf_counter() - started:5.1f}s] índice léxico: {len(arrays['tfidf_idf']):,} términos, "
          f"{len(arrays['doc_pos']):,} postings; idéntico al original en {len(sample)} consultas x 3 métodos")

    write_docstore(documents, ids, out)
    check_docstore(documents, ids, out)
    print(f"[{time.perf_counter() - started:5.1f}s] docs.bin escrito y verificado")

    judged = {key: {"text": queries[key]["text"], "relevant": sorted(relevant[key])}
              for key in sorted(relevant)}
    (out / "judgments.json").write_text(json.dumps(judged, ensure_ascii=False, separators=(",", ":")),
                                        encoding="utf-8")

    # El índice FAISS y el codificador ONNX deben ser compatibles antes de publicar nada.
    encoder = OnnxEncoder(out / "model" / "model.onnx", out / "model" / "vocab.txt")
    semantic = SemanticIndex.open(ids, out / faiss_name, encoder)
    top = next(semantic.search_many([queries[sorted(relevant)[0]]["text"]], 3))
    print(f"[{time.perf_counter() - started:5.1f}s] FAISS + ONNX abren y buscan: {top[0]}")

    files = sorted(path for path in out.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest = {
        "format": 1,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": {"id": MODEL, "revision": MODEL_REVISION, "dimension": encoder.dimension,
                  "runtime": "onnxruntime (grafo onnx/model.onnx de la misma revisión)"},
        "corpus": {"name": "BeIR/fiqa", "documents": len(documents),
                   "test_queries": len(relevant), "k": report["k"]},
        "faiss": faiss_name,
        "files": [file_entry(path, out) for path in files],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                       encoding="utf-8")
    total = sum(item["bytes"] for item in manifest["files"])
    print("\nartefactos:")
    for item in manifest["files"]:
        print(f"  {item['bytes'] / 2**20:8.2f} MB  {item['path']}")
    print(f"  {total / 2**20:8.2f} MB  TOTAL  ({time.perf_counter() - started:.0f} s)")


if __name__ == "__main__":
    main()
