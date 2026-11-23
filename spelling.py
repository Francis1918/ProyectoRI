"""Sugerencia ortográfica opcional para el estado «Sin resultados» de Jaccard, TF-IDF y BM25.

Solo reutiliza lo que el índice léxico ya tiene: el vocabulario de FiQA y la frecuencia de cada
término (en cuántos documentos aparece). No toca los rankings: el motor la consulta únicamente
cuando una búsqueda léxica ya devolvió cero resultados, y nunca para el método semántico.

Regla de decisión, deliberadamente conservadora (sin candidata clara no se sugiere nada):

1. Solo se corrigen palabras ASCII de 4 a 24 letras. El diccionario de sugerencias son los
   términos del vocabulario que aparecen en al menos ``MIN_DF`` documentos: FiQA tiene erratas
   propias («mortage» en 19 documentos, «morgage» en 5), y no deben servir de destino.
2. Candidatas: palabras del diccionario a distancia de edición <= 1 (palabras de 4 a 6 letras) o
   <= 2 (de 7 en adelante). La distancia es Damerau-Levenshtein restringida (OSA): inserción,
   omisión, sustitución y transposición de dos letras contiguas cuestan 1 cada una.
3. Puntuación de canal ruidoso: ``documentos / PENALTY ** distancia``. Una edición «cuesta»
   ``PENALTY`` veces en frecuencia, así que una palabra muy usada a distancia 2 puede ganar a una
   rara a distancia 1, pero no a otra común a distancia 1.
4. Solo se sugiere si la mejor puntuación supera en ``CLEAR_RATIO`` veces a la segunda.
5. Si la palabra está tan cerca de una palabra vacía (the, with…) como de la mejor candidata, se
   considera ambigua: el índice descarta las palabras vacías y no hay a qué corregir.
6. Una consulta de varias palabras se corrige solo si **todas** sus palabras tienen una candidata
   clara; si no, no hay sugerencia (nunca se ofrece una corrección a medias).

Los umbrales se fijaron midiendo sobre el vocabulario real de FiQA (78 180 términos; 14 537 en el
diccionario): erratas sintéticas (omisión, inserción, sustitución, transposición) de palabras
frecuentes y cadenas aleatorias. Con estos valores, una errata recibe sugerencia en ~83 % de los
casos y acierta ~98 %; una cadena aleatoria recibe sugerencia ~1,4 % de las veces (casi siempre de
4 a 6 letras, a una edición de una palabra común). Con 2 ediciones desde 6 letras, «jwriki» daba
«wiki» y «mikelf» daba «likely» (8 % de las cadenas de 6 letras): de ahí LONG_WORD = 7.
"""

import threading
import unicodedata

import numpy as np

from retrieval import STOPWORDS, WORD

MIN_DF = 10          # documentos mínimos para que un término sea destino de una corrección
MIN_LENGTH = 4       # no se corrigen palabras más cortas: casi cualquier cadena corta está a 1 edición de algo
MAX_LENGTH = 24
LONG_WORD = 7        # desde esta longitud se admiten 2 ediciones; antes, 1
MAX_WORDS = 6        # consultas con más palabras fuera del vocabulario no se corrigen (acota el coste)
PENALTY = 25.0       # una edición equivale a perder esta razón de frecuencia
CLEAR_RATIO = 6.0    # la mejor candidata debe puntuar al menos esto por encima de la segunda


def edit_distance(a, b):
    """Distancia Damerau-Levenshtein restringida (OSA): inserción, omisión, sustitución y transposición."""
    previous2, previous = None, list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        current = [i]
        for j in range(1, len(b) + 1):
            best = min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                best = min(best, previous2[j - 2] + 1)
            current.append(best)
        previous2, previous = previous, current
    return previous[-1]


def _distances(word, columns):
    """`edit_distance(word, c)` para todas las palabras `c` de un grupo de igual longitud.

    `columns` es una matriz (longitud, k): la letra j de las k palabras. Misma recurrencia que
    `edit_distance`, vectorizada sobre las k palabras (los tests comprueban que coinciden).
    """
    a = word.encode("ascii")
    length, count = columns.shape
    previous2, previous = None, [np.full(count, j, dtype=np.int8) for j in range(length + 1)]
    for i in range(1, len(a) + 1):
        current = [np.full(count, i, dtype=np.int8)]
        for j in range(1, length + 1):
            column = columns[j - 1]
            best = np.minimum(np.minimum(previous[j], current[j - 1]) + 1, previous[j - 1] + (column != a[i - 1]))
            if i > 1 and j > 1:
                swapped = (column == a[i - 2]) & (columns[j - 2] == a[i - 1])
                best = np.where(swapped, np.minimum(best, previous2[j - 2] + 1), best)
            current.append(best)
        previous2, previous = previous, current
    return previous[length]


class Speller:
    """Sugiere una única consulta corregida a partir del vocabulario y la frecuencia por documento."""
# carga un diccionario y sugiere correcciones otortograficos simples
    def __init__(self, terms, frequencies):
        """`terms` y `frequencies` van en el mismo orden (el del índice léxico)."""
        self._terms, self._frequencies = terms, frequencies
        self._groups = None
        self._lock = threading.Lock()
        self._stopwords = sorted(word for word in STOPWORDS if word.isascii() and word.isalpha())

    def _load(self):
        """Agrupa el diccionario por longitud, una sola vez y la primera vez que hace falta."""
        if self._groups is not None:
            return self._groups
        with self._lock:
            if self._groups is None:
                by_length = {}
                for term, frequency in zip(self._terms, np.asarray(self._frequencies).tolist()):
                    if frequency >= MIN_DF and term.isascii() and term.isalpha() and 2 <= len(term) <= MAX_LENGTH + 2:
                        by_length.setdefault(len(term), []).append((term, frequency))
                groups = {}
                for length, items in by_length.items():
                    words = [term for term, _ in items]
                    matrix = np.frombuffer("".join(words).encode("ascii"), dtype=np.uint8).reshape(len(words), length)
                    groups[length] = (words, np.array([frequency for _, frequency in items], dtype=np.int64),
                                      np.ascontiguousarray(matrix.T))
                self._groups = groups
        return self._groups

    def correct(self, word):
        """Corrección clara de una palabra, o None. Devuelve (palabra, distancia, documentos);
        si la palabra ya está en el diccionario devuelve ella misma con distancia 0."""
        size = len(word)
        if not (MIN_LENGTH <= size <= MAX_LENGTH and word.isascii() and word.isalpha()):
            return None
        groups = self._load()
        limit = 1 if size < LONG_WORD else 2
        found = []                                   # (puntuación, distancia, documentos, palabra)
        for length in range(size - limit, size + limit + 1):
            if length not in groups:
                continue
            words, frequencies, columns = groups[length]
            distances = _distances(word, columns)
            for index in np.flatnonzero(distances <= limit).tolist():
                distance, frequency = int(distances[index]), int(frequencies[index])
                if distance == 0:
                    return word, 0, frequency
                found.append((frequency / PENALTY ** distance, distance, frequency, words[index]))
        if not found:
            return None
        found.sort(key=lambda item: (-item[0], item[1], item[3]))
        best = found[0]
        if len(found) > 1 and best[0] < CLEAR_RATIO * found[1][0]:
            return None
        if any(edit_distance(word, stop) <= best[1] for stop in self._stopwords):
            return None
        return best[3], best[1], best[2]

    def suggest(self, query):
        """Consulta corregida como {"query", "changes"}, o None si no hay una candidata clara."""
        text = unicodedata.normalize("NFKC", query).casefold()    # el mismo preprocesado que `tokenize`
        words = [match for match in WORD.finditer(text) if match.group() not in STOPWORDS]
        if not words or len(words) > MAX_WORDS:
            return None
        changes, pieces, cursor = [], [], 0
        for match in words:
            fixed = self.correct(match.group())
            if fixed is None:
                return None
            target, distance, frequency = fixed
            if distance:
                changes.append({"from": match.group(), "to": target, "distance": distance, "documents": frequency})
                pieces += [text[cursor:match.start()], target]
                cursor = match.end()
        if not changes:
            return None
        pieces.append(text[cursor:])
        return {"query": "".join(pieces).strip(), "changes": changes}
