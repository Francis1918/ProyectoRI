# Lente — buscador de FiQA con modelos clásicos y semánticos

Proyecto de Recuperación de Información (Prof. Iván Carrera, entrega 23-nov-2026).

- **Repositorio:** <https://github.com/Francis1918/ProyectoRI>
- **Clonado:** `git clone https://github.com/Francis1918/ProyectoRI.git`
- **Web pública:** <https://lente-fiqa.vercel.app>

## Qué hace Lente

Lente permite escribir una consulta en lenguaje natural y ver qué documentos de **FiQA** recupera cada modelo de recuperación. Compara cuatro métodos sobre el mismo corpus:

| Método en la interfaz | Modelo del enunciado | Cómo puntúa |
|---|---|---|
| **Jaccard** | Binario | Similitud de Jaccard entre conjuntos de términos (vectores binarios). |
| **TF-IDF** | Vectorial con TF-IDF | Coseno entre vectores TF-IDF. |
| **BM25** | Probabilístico | Okapi BM25 (k1 = 1,2; b = 0,75). |
| **Semántico** | Vectorial con embeddings | Embeddings de `all-MiniLM-L6-v2` y búsqueda vectorial con FAISS (producto interno sobre vectores normalizados). |

Los tres primeros son modelos basados en términos sobre un índice invertido propio (`retrieval.py`); el cuarto usa embeddings y una base vectorial FAISS. No se usa ningún motor de búsqueda completo (Elasticsearch, Solr, Whoosh).

### Qué es FiQA

FiQA es un conjunto de preguntas y respuestas del ámbito financiero (BEIR/FiQA en Hugging Face). Tiene **57 638 documentos** y **648 consultas de prueba** con juicios de relevancia (qrels), con unos 2,6 documentos relevantes por consulta. Las métricas se calculan contra esos juicios.

## Cómo probar la interfaz

Abre <https://lente-fiqa.vercel.app> (o la versión local, ver más abajo) y prueba:

1. **Los cuatro métodos.** Escribe una consulta, por ejemplo `can I deduct a home office if I work remotely as a freelancer`. El menú cambia entre Jaccard, TF-IDF, BM25 y Semántico, y el *top-n* (5–50) ajusta cuántos resultados se muestran.
2. **Comparar.** El enlace **«Comparar los cuatro métodos»** (o la vista *Comparar* de la lista de resultados) muestra, para cada documento, su posición en los cuatro métodos y una matriz de solapamiento. Se compara por **posiciones**, no por puntuaciones, porque las puntuaciones de cada método no son comparables.
3. **Leer completo.** El botón **«Leer completo»** de cada resultado abre el texto íntegro del documento.
4. **Sugerencia ortográfica.** Prueba `moretgage` con Jaccard, TF-IDF o BM25: como no hay resultados, la página muestra **«¿Quisiste decir mortgage?»**. Al pulsarla se ejecuta la consulta corregida con el mismo método. Con Semántico no se sugiere nada, porque ese método no lo necesita. Limitación conocida: las erratas que existen en FiQA (p. ej. `morgage`, que aparece en 5 documentos) devuelven resultados normales y no muestran sugerencia.
5. **Evaluación.** El enlace **«Evaluación FiQA»** (arriba a la derecha) muestra las métricas del informe, dónde gana cada método y una tabla de las 648 consultas, con filtros y enlace a *Comparar*.

## Resultados de la evaluación (FiQA completo, k = 10, 648 consultas)

Valores tomados de `data/fiqa/report_k10.json` (se generan con `main.py evaluate`, ver más abajo):

| Método | MAP@10 | Precision@10 media | Recall@10 medio |
|---|---|---|---|
| Jaccard binario | 0,0298 | 0,0144 | 0,0589 |
| Coseno TF-IDF | 0,1159 | 0,0492 | 0,2060 |
| BM25 | 0,1726 | 0,0656 | 0,2969 |
| Semántico (MiniLM + FAISS) | **0,2914** | **0,1045** | **0,4413** |

- **Precision@k** = relevantes en el top-k / k. **Recall@k** = relevantes en el top-k / total de relevantes de la consulta.
- **AP@k** = suma de las precisiones en cada acierto dentro del top-k, dividida entre el **total de relevantes** de la consulta. **MAP@k** es la media de AP@k sobre las consultas.
- Con k = 10 y ~2,6 relevantes por consulta, los valores sirven para comparar métodos entre sí, no como precisión absoluta.

## Requisitos

- **Python 3.13** (indicado en `.python-version`; `pyproject.toml` acepta ≥ 3.12).
- Conexión a internet para descargar FiQA y el modelo de Hugging Face la primera vez.
- Espacio: el entorno con PyTorch ocupa varios GB; el índice FAISS completo pesa unos 88 MB.
- Para construir el índice FAISS, unos **35 minutos** de CPU (una sola vez).

## Instalación desde un clon limpio

```powershell
git clone https://github.com/Francis1918/ProyectoRI.git
cd ProyectoRI
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

En Linux o macOS: `python3.13 -m venv .venv` y `source .venv/bin/activate`.

`requirements.txt` instala la parte 1 completa (`datasets`, `sentence-transformers` con PyTorch, `faiss-cpu`) y la web local (`fastapi`, `uvicorn`, `onnxruntime`).

> **Qué no viene en Git.** El repositorio no contiene `data/` (corpus, consultas, qrels, índice FAISS, informe) ni `artifacts/` (los artefactos de la web, unos 205 MB). Por eso, tras clonar hay que generarlos con los pasos de abajo. Solo `artifacts/manifest.json` (tamaños y SHA-256) está en Git.

## Preparar FiQA y generar los artefactos

Los comandos se ejecutan desde la raíz del proyecto, con el entorno activado. En Windows, `$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"` evita avisos de caché de Hugging Face.

```powershell
# 1. Descargar y validar FiQA completo -> data/fiqa/{corpus,queries,qrels}.jsonl
python main.py prepare

# 2. Construir el índice FAISS de los 57 638 documentos (~35 min en CPU; se reutiliza después)
python main.py index

# 3. Reproducir la evaluación con k = 10 -> data/fiqa/report_k10.json (~100 s)
python main.py evaluate --k 10 --output data/fiqa/report_k10.json

# 4. Empaquetar los artefactos de la web -> artifacts/ (~2 min; descarga el modelo ONNX)
python build_artifacts.py
```

Notas:

- `--data` es una opción global y va **antes** del subcomando si se cambia (`python main.py --data data/fiqa_muestra prepare --sample-queries 20`). Por defecto usa `data/fiqa`.
- El paso 4 necesita el `report_k10.json` del paso 3 y el índice FAISS del paso 2.
- En Windows, `build_artifacts.py --help` falla con la consola en cp1252 por un carácter del texto de ayuda; para ver la ayuda usa `$env:PYTHONUTF8="1"` antes.

## Buscar desde la línea de comandos

```powershell
python main.py search "can I deduct a home office if I work remotely as a freelancer" --method semantic --top 3
python main.py search "consulta libre" --method bm25 --top 10
```

`--method` acepta `jaccard`, `tfidf`, `bm25` o `semantic`. Devuelve JSON con `rank`, `id`, `score`, `title` y `text`.

## Ejecutar la web en local

```powershell
python -m uvicorn app:app --port 8000
```

Abre <http://127.0.0.1:8000>. La misma aplicación sirve la interfaz (`public/`) y la API:

| Ruta | Función |
|---|---|
| `/api/health` | Estado, tiempos de arranque y memoria. |
| `/api/search?q=&method=&n=` | Top-n de un método; con 0 resultados léxicos añade `suggestion`. |
| `/api/compare?q=&n=` | Los cuatro métodos para una consulta (posiciones). |
| `/api/evaluation` | Valores de `report_k10.json`. |

Sin `artifacts/` la API responde 503 con el motivo, y `/api/health` lo muestra.

## Pruebas

```powershell
python -m unittest test_lente -v          # API, reutilización de artefactos, sugerencia ortográfica
python verify_engine.py                    # MAP/P/R con el camino de producción vs report_k10.json
python verify_encoder.py                   # tokenizador ONNX/WordPiece y vectores vs PyTorch y FAISS (lento)
```

`test_lente` necesita `artifacts/` y `data/fiqa/report_k10.json`. `verify_encoder.py` usa `tokenizers` y PyTorch, que no van al despliegue.

Para comprobar la web publicada frente al motor local (mismas posiciones y puntuaciones, y MAP igual al informe):

```powershell
python scripts\check_deployment.py https://lente-fiqa.vercel.app
```

## Despliegue en Vercel

La web se despliega **con la CLI de Vercel**, no con un build por Git:

- Los artefactos de `artifacts/` (índice FAISS, modelo ONNX, corpus comprimido, índice léxico) no están en Git. La CLI sube la carpeta del proyecto y **`.vercelignore` es una lista blanca**: solo viajan el código de la web, `pyproject.toml`/`uv.lock`, `public/` y `artifacts/`. Quedan fuera `data/`, `.venv/`, `requirements.txt` (con PyTorch) y las herramientas de desarrollo. Todo módulo Python nuevo debe añadirse a esa lista (`!nombre.py`).
- `vercel.json` declara el preset de FastAPI, `app.py` como función (`includeFiles: artifacts/**`), `maxDuration` de 60 s y las cabeceras de seguridad (CSP estricta, `nosniff`, `Referrer-Policy`).
- Las dependencias de producción están en `pyproject.toml` y `uv.lock`, sin PyTorch: el codificador de consultas usa **ONNX Runtime** con el mismo modelo y revisión que la parte 1.
- El paquete de la función supera el límite estándar de Vercel, así que `vercel.json` activa `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` (funciones grandes, en beta). Si un build falla con «exceeds the maximum function size», añade esa variable en el proyecto.

Pasos con la CLI, usando **tu propia cuenta** de Vercel (no se incluyen credenciales en el repositorio):

```powershell
npx vercel@latest login                     # una vez
npx vercel@latest link                      # vincula la carpeta a un proyecto; no conectes Git
npx vercel@latest deploy --prod             # sube código + artifacts/ y construye en Vercel
```

Con una cuenta y un proyecto nuevos hay que crear antes el proyecto (`vercel project create`) y aceptar el aviso de que no hay Git. Los `git push` **no** redespliegan la web, porque los artefactos no están en el repositorio.

## Estructura del proyecto

| Archivo | Función |
|---|---|
| `corpus.py` | Descarga y valida FiQA; exporta JSON Lines en `data/fiqa/`. |
| `retrieval.py` | Tokenización, índice invertido (Jaccard, TF-IDF, BM25) e índice semántico FAISS. |
| `main.py` | CLI: `prepare`, `index`, `search`, `evaluate`. |
| `lente_engine.py` | Motor de la web: abre los artefactos y presenta resultados, comparaciones y evaluación. |
| `onnx_encoder.py` | Codificador de consultas MiniLM con ONNX Runtime y WordPiece propio. |
| `spelling.py` | Sugerencia ortográfica para consultas léxicas sin resultados. |
| `app.py` | API FastAPI y servidor local de `public/`. |
| `build_artifacts.py` | Genera `artifacts/` a partir de `data/fiqa/` sin recalcular embeddings ni métricas. |
| `public/` | Interfaz web (HTML, CSS y JavaScript sin compilación). |
| `test_lente.py`, `verify_engine.py`, `verify_encoder.py` | Pruebas y verificaciones. |
| `scripts/check_deployment.py` | Compara la URL publicada con el motor local. |
| `scripts/prepare_background.py` | Prepara la imagen de fondo de la portada. |
| `CREDITS.md`, `public/img/CREDITS.md` | Licencias y origen de la fotografía, fuentes y modelo. |

## Estado de las comprobaciones (2026-11-21)

Esto es lo que se ejecutó en el entorno de desarrollo; el resto está documentado pero no se repitió al escribir este README.

- **Ejecutado:** `python -m unittest test_lente` (33 pruebas, OK); `python main.py evaluate --k 10` (reproduce los MAP del informe; 648 consultas, diferencia de AP igual a 0 en los cuatro métodos); `python main.py search --method semantic` (devuelve los mismos ids y puntuaciones que el informe); `uvicorn app:app` con `/api/health`, la portada y la sugerencia para `moretgage`; `scripts/check_deployment.py` contra <https://lente-fiqa.vercel.app> (resultado OK).
- **Revisado en el código, no ejecutado de nuevo:** `prepare` (requiere descargar FiQA) e `index` (~35 min); `build_artifacts.py` (regenera artefactos y el modelo ONNX); `verify_encoder.py`, `verify_engine.py`, el despliegue con la CLI y los pasos de Vercel.
