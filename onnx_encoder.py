"""Codificador de consultas all-MiniLM-L6-v2 con ONNX Runtime, sin PyTorch.

Es el mismo modelo y la misma revisión que usa `SemanticIndex` en el CLI
(`retrieval.MODEL`, `retrieval.MODEL_REVISION`); aquí solo cambia el motor de inferencia, para que
quepa en una función serverless. Replica los tres pasos de la tubería de sentence-transformers:

1. tokenización BERT uncased (WordPiece, `vocab.txt` de esa revisión) truncada a 256 tokens,
2. Transformer (grafo `onnx/model.onnx` de la misma revisión del repositorio del modelo),
3. promedio de tokens con la máscara de atención y normalización L2.

El WordPiece se implementa aquí para no arrastrar `tokenizers` y `huggingface_hub` (~35 MB) al
despliegue. `verify_encoder.py` comprueba que produce los mismos ids que el tokenizador oficial y
los mismos vectores que ya están en el índice FAISS.
"""

from pathlib import Path
import re
import unicodedata

import numpy as np

MAX_LENGTH = 256          # sentence_bert_config.json: max_seq_length
MAX_WORD_CHARS = 100      # tokenizer.json: max_input_chars_per_word
SPECIAL = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")
SPLIT_SPECIAL = re.compile("(" + "|".join(re.escape(token) for token in SPECIAL) + ")")
# Unicode White_Space (lo que usa el tokenizador oficial), no str.isspace().
WHITE_SPACE = frozenset(
    "\t\n\x0b\x0c\r \x85\xa0           "
    "     　"
)
CJK = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F),
       (0x2B740, 0x2B81F), (0x2B920, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F))


def _is_cjk(char):
    code = ord(char)
    return any(low <= code <= high for low, high in CJK)


def _is_control(char):
    # Como el tokenizador oficial: Cc, Cf, Co y Cs; los sin asignar (Cn) NO se descartan (acaban en [UNK]).
    return char not in "\t\n\r" and unicodedata.category(char) in ("Cc", "Cf", "Co", "Cs")


def _is_punctuation(char):
    code = ord(char)
    ascii_punctuation = 33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126
    return ascii_punctuation or unicodedata.category(char)[0] == "P"


class WordPiece:
    """BertNormalizer (clean, CJK, sin acentos, minúsculas) + BertPreTokenizer + WordPiece."""

    def __init__(self, vocab_path, max_length=MAX_LENGTH):
        lines = Path(vocab_path).read_text(encoding="utf-8").split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        self.vocab = {line.rstrip("\r"): index for index, line in enumerate(lines)}
        self.max_length = max_length
        self.cls, self.sep, self.unk = (self.vocab[token] for token in ("[CLS]", "[SEP]", "[UNK]"))
        self.special = {token: self.vocab[token] for token in SPECIAL}

    @staticmethod
    def normalize(text):
        cleaned = []
        for char in text:
            if char == "\0" or char == "�" or _is_control(char):
                continue
            char = " " if char in WHITE_SPACE else char
            cleaned.append(f" {char} " if _is_cjk(char) else char)
        decomposed = unicodedata.normalize("NFD", "".join(cleaned))
        stripped = (char for char in decomposed if unicodedata.category(char) != "Mn")
        # Carácter a carácter, como el tokenizador oficial (sin la regla de sigma final de str.lower).
        return "".join(char.lower() for char in stripped)

    @staticmethod
    def words(normalized):
        word = []
        for char in normalized:
            if char in WHITE_SPACE:
                if word:
                    yield "".join(word)
                    word = []
            elif _is_punctuation(char):
                if word:
                    yield "".join(word)
                    word = []
                yield char
            else:
                word.append(char)
        if word:
            yield "".join(word)

    def pieces(self, word):
        if len(word) > MAX_WORD_CHARS:
            return [self.unk]
        found, start = [], 0
        while start < len(word):
            end = len(word)
            current = None
            while start < end:
                candidate = word[start:end] if start == 0 else "##" + word[start:end]
                if candidate in self.vocab:
                    current = self.vocab[candidate]
                    break
                end -= 1
            if current is None:
                return [self.unk]
            found.append(current)
            start = end
        return found

    def encode(self, text):
        body = []
        for part in SPLIT_SPECIAL.split(text):
            if part in self.special:
                body.append(self.special[part])
                continue
            for word in self.words(self.normalize(part)):
                body.extend(self.pieces(word))
        return [self.cls, *body[: self.max_length - 2], self.sep]


class OnnxEncoder:
    dimension = 384

    def __init__(self, model_path, vocab_path, threads=1):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self.session = ort.InferenceSession(
            str(model_path), options, providers=["CPUExecutionProvider"]
        )
        self.input_names = {item.name for item in self.session.get_inputs()}
        self.tokenizer = WordPiece(vocab_path)

    def encode(self, texts, batch_size=32):
        batches = []
        for offset in range(0, len(texts), batch_size):
            batches.append(self._encode_batch(texts[offset:offset + batch_size]))
        if not batches:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.concatenate(batches)

    def _encode_batch(self, texts):
        encoded = [self.tokenizer.encode(text) for text in texts]
        width = max(len(ids) for ids in encoded)
        input_ids = np.zeros((len(encoded), width), dtype=np.int64)
        mask = np.zeros((len(encoded), width), dtype=np.int64)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = ids
            mask[row, : len(ids)] = 1
        feeds = {"input_ids": input_ids, "attention_mask": mask,
                 "token_type_ids": np.zeros_like(input_ids)}
        hidden = self.session.run(None, {k: v for k, v in feeds.items() if k in self.input_names})[0]
        weights = mask[:, :, None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1e-9)
        norms = np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
        return (pooled / norms).astype(np.float32)
