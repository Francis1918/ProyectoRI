"""Verifica que el codificador ONNX (onnx_encoder.py) equivale al de la parte 1 (PyTorch).

Herramienta de desarrollo: usa tokenizers, sentence-transformers y FAISS, que NO van al despliegue.

    python verify_encoder.py                 # tokenización (todos los code points + corpus) y vectores
    python verify_encoder.py --skip-unicode  # omite el barrido de code points
"""

import argparse
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

from corpus import document_text, load_documents, load_evaluation
from onnx_encoder import MAX_LENGTH, OnnxEncoder, WordPiece
from retrieval import MODEL, MODEL_REVISION

EDGE_CASES = [
    "", " ", "Hello, World!", "can I deduct a home office if I work remotely as a freelancer",
    "[CLS] literal [SEP] [MASK] [PAD] [UNK] tokens", "[cls] minúsculas no son especiales",
    "ΑΣ ΟΔΥΣΣΕΥΣ final sigma", "İstanbul İ I ı", "Café déjà vu naïve façade Ångström",
    "東京タワー 한국어 中文测试", "emoji 😀👍🏽 ok", "a​b zero‍width", "tab\tnew\nline\r\nmix",
    "x" * 150, "y" * 99 + " " + "z" * 101, "1,000.50 $ + 20% = ?", "state-of-the-art re-balancing 401(k)",
    "word " * 400, "\u0085nel ls ps nbsp", "\x00null�repl\x7fdel\x1bbesc",
]


def official_tokenizer(snapshot):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    tokenizer.enable_truncation(max_length=MAX_LENGTH)
    tokenizer.no_padding()
    return tokenizer


# Code points reclasificados entre las tablas Unicode del tokenizador oficial y las de Python 3.13.
RECLASSIFIED = {0x166D, 0x1734, 0x111C9}


def compare_ids(mine, official, texts, label, unicode_sweep=False):
    """Compara los ids. En el barrido de code points se aceptan solo las diferencias que se explican
    porque el tokenizador oficial usa tablas Unicode más antiguas: trata como "sin asignar"
    (mismos ids que U+0378) un carácter que Python ya conoce, o es uno de los reclasificados."""
    unassigned = official.encode("a͸b").ids
    bad, explained = [], 0
    for start in range(0, len(texts), 2000):
        chunk = texts[start:start + 2000]
        for text, reference in zip(chunk, official.encode_batch(chunk)):
            if mine.encode(text) == reference.ids:
                continue
            char = text.replace("a", "").replace("b", "") if unicode_sweep else ""
            if unicode_sweep and len(char) == 1 and (
                    ord(char) in RECLASSIFIED or official.encode(f"a{char}b").ids == unassigned):
                explained += 1
            else:
                bad.append(text)
    note = f" (+{explained} explicadas por la versión de Unicode)" if explained else ""
    print(f"  tokenización {label}: {len(texts):,} textos -> "
          f"{'OK' if not bad else f'{len(bad)} DIFERENCIAS'}{note}")
    for text in bad[:5]:
        print("    ejemplo:", repr(text[:80]), mine.encode(text)[:12], official.encode(text).ids[:12])
    return not bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/fiqa"))
    parser.add_argument("--skip-unicode", action="store_true")
    parser.add_argument("--sample", type=int, default=300)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    def fetch(name):
        return Path(hf_hub_download(MODEL, name, revision=MODEL_REVISION))

    snapshot = fetch("tokenizer.json").parent
    manifest_dir = Path("artifacts/model")
    model_path = manifest_dir / "model.onnx"
    vocab_path = manifest_dir / "vocab.txt"
    if not model_path.exists():
        model_path, vocab_path = fetch("onnx/model.onnx"), fetch("vocab.txt")
    print("modelo:", MODEL, MODEL_REVISION[:8], "| onnx:", model_path)
    ok = True

    mine = WordPiece(vocab_path)
    official = official_tokenizer(snapshot)
    official_vocab = json.loads((snapshot / "tokenizer.json").read_text(encoding="utf-8"))["model"]["vocab"]
    same_vocab = mine.vocab == official_vocab
    print(f"vocabulario idéntico al de tokenizer.json ({len(official_vocab):,} tokens): {same_vocab}")
    ok &= same_vocab

    print("tokenización:")
    ok &= compare_ids(mine, official, EDGE_CASES, "casos límite")
    if not args.skip_unicode:
        points = [chr(code) for code in range(0x110000) if not 0xD800 <= code <= 0xDFFF]
        started = time.perf_counter()
        ok &= compare_ids(mine, official, [f"a{c}b" for c in points], "todos los code points (a·c·b)",
                          unicode_sweep=True)
        print(f"  ({time.perf_counter() - started:.0f} s)")
    documents = load_documents(args.data)
    queries, relevant = load_evaluation(args.data, documents)
    texts = [document_text(row) for row in documents.values()]
    ok &= compare_ids(mine, official, texts, "corpus FiQA completo (título + texto)")
    query_texts = [queries[key]["text"] for key in sorted(relevant)]
    ok &= compare_ids(mine, official, query_texts, "648 consultas de prueba")

    print("vectores:")
    import faiss
    from sentence_transformers import SentenceTransformer

    encoder = OnnxEncoder(model_path, vocab_path, threads=2)
    index_path = next(args.data.glob("documents-*.faiss"))
    index = faiss.read_index(str(index_path))
    ids = sorted(documents)
    rng = random.Random(0)
    positions = sorted(rng.sample(range(len(ids)), args.sample))
    stored = np.stack([index.reconstruct(position) for position in positions])
    fresh = encoder.encode([document_text(documents[ids[position]]) for position in positions])
    cosine = (stored * fresh).sum(axis=1)
    print(f"  ONNX vs vector guardado en FAISS ({args.sample} documentos al azar): "
          f"coseno mín {cosine.min():.7f}, |Δ| máx {np.abs(stored - fresh).max():.2e}")
    ok &= bool(cosine.min() > 0.99999)

    torch_model = SentenceTransformer(MODEL, revision=MODEL_REVISION, device="cpu")
    reference = torch_model.encode(query_texts, batch_size=32, normalize_embeddings=True,
                                   convert_to_numpy=True, show_progress_bar=False)
    candidate = encoder.encode(query_texts)
    cosine = (reference * candidate).sum(axis=1)
    print(f"  ONNX vs PyTorch (648 consultas): coseno mín {cosine.min():.7f}, "
          f"|Δ| máx {np.abs(reference - candidate).max():.2e}")
    ok &= bool(cosine.min() > 0.99999)

    print("RESULTADO:", "TODO OK" if ok else "HAY DIFERENCIAS")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
