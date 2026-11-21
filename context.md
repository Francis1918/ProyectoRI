# Contexto — Proyecto RI, parte 1 (motor local y evaluación)

Última actualización: 2026-11-21. Enunciado: `proy01.md`. Entrega: 23-nov-2026.

## 1. Lo que ya existía (código dejado por Codex, conservado sin cambios)
- `corpus.py`: carga/validación de FiQA (`BeIR/fiqa`, qrels de test), IDs normalizados a `str`, exporta JSONL a `data/fiqa/`. Opción `--sample-queries` para muestras.
- `retrieval.py`: `tokenize` (NFKC, casefold, regex de palabras, stopwords en inglés), `InvertedIndex` (postings término→{doc: frecuencia}) con Jaccard binario, coseno TF-IDF y BM25 (k1=1.2, b=0.75); `SemanticIndex` (sentence-transformers `all-MiniLM-L6-v2`, revisión fijada, CPU, FAISS `IndexFlatIP` con embeddings normalizados; mismo modelo para documentos y consultas; el índice se guarda y se reutiliza por hash del contenido).
- `main.py`: subcomandos `prepare`, `index`, `search`, `evaluate`.
- `requirements.txt`, `.gitignore` (ignora `.venv/`, `data/`). No hay repositorio git. No existía `context.md`.

## 2. Qué hice en esta sesión
- Comprobé el entorno: `.venv\Scripts\python.exe` (3.13.9) arranca sin "acceso denegado" y importa faiss/torch/sentence_transformers/datasets. Hugging Face responde (HTTP 200). **Los dos bloqueos del agente anterior no se reprodujeron**; no cambié entorno ni dependencias. La caché solo tenía SciFact, así que descargué FiQA.
- No modifiqué ningún archivo de código; todo lo pedido ya estaba implementado.
- Descargué FiQA completo y construí el índice FAISS completo (proceso en segundo plano, ~35 min en CPU).

## 3. Qué verifiqué
- FiQA: 57 638 documentos, 648 consultas de test, 1 706 qrels; todas las referencias de qrels existen en corpus y consultas.
- Índice FAISS `data/fiqa/documents-180307632d0f….faiss` (88 532 013 bytes): carga, `ntotal=57638`, `d=384`, métrica producto interno.
- Búsqueda léxica (Jaccard, TF-IDF, BM25) y semántica con consultas libres nuevas.
- Evaluación completa de los cuatro métodos, corpus completo, k=10 (101 s con índice ya construido).

## 4. Resultados (FiQA completo, k=10, 648 consultas)
| Método | MAP@10 | Precision@10 media | Recall@10 medio |
|---|---|---|---|
| Jaccard binario | 0.0298 | 0.0144 | 0.0589 |
| Coseno TF-IDF | 0.1159 | 0.0492 | 0.2060 |
| BM25 | 0.1726 | 0.0656 | 0.2969 |
| Semántico (MiniLM + FAISS) | 0.2914 | 0.1045 | 0.4413 |

Precision y Recall por consulta están en `data/fiqa/report_k10.json` (carpeta ignorada por git).

Consulta nueva de prueba (semántico): «can I deduct a home office if I work remotely as a freelancer» → top-3 ids 295509 (0.7468), 288537 (0.6529), 377621 (0.6165); los tres tratan deducción de oficina en casa.

## 5. Significado del corte k
k = número de documentos del ranking que se evalúan (por defecto 10, `--k`).
- Precision@k = relevantes en el top-k / k.
- Recall@k = relevantes en el top-k / total de relevantes de la consulta.
- AP@k = suma de las precisiones en cada posición con acierto dentro del top-k, dividida entre el **total de relevantes** (no entre min(k, relevantes)). MAP = media de AP@k sobre las consultas. Por eso es "MAP@10" y es conservador cuando hay más de 10 relevantes.
- Como FiQA tiene pocos relevantes por consulta (~2.6) y k=10, la Precision máxima alcanzable es baja; compara métodos entre sí, no con 1.0.

## 6. Comandos exactos (desde la raíz del proyecto, PowerShell)
```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
.\.venv\Scripts\python.exe main.py --data data/fiqa prepare                  # descarga y valida FiQA completo
.\.venv\Scripts\python.exe main.py index                                      # construye el índice FAISS (~35 min CPU, una sola vez)
.\.venv\Scripts\python.exe main.py search "consulta libre" --method bm25 --top 10   # métodos: jaccard | tfidf | bm25 | semantic
.\.venv\Scripts\python.exe main.py evaluate --k 10 --output data/fiqa/report_k10.json
```
Muestra pequeña para pruebas rápidas: `main.py --data data/fiqa_muestra prepare --sample-queries 20` y después usar el mismo `--data` en el resto de comandos.

## 7. Pendiente / notas
- Parte 2 (no hecha, fuera de alcance): interfaz web, comparación entre modelos en la UI, despliegue en Vercel (la dependencia `fastapi`/`uvicorn` ya está en requirements). Ojo: el índice FAISS pesa 88 MB y cargar el modelo puede exceder límites de funciones serverless; evaluarlo antes.
- Informe técnico (≤5 páginas) y README no escritos; sección f del enunciado (análisis de en qué consultas gana/pierde el semántico frente a BM25/TF-IDF) pendiente: se puede hacer con `report_k10.json`.
- Cada búsqueda léxica reconstruye el índice invertido en memoria (unos segundos); para la web conviene construirlo una vez al arrancar.
- `hf_xet` no está instalado (solo avisa y usa HTTP normal); no es necesario.
