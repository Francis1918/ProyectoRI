# Contexto — Proyecto RI (parte 1: motor y evaluación · parte 2: Lente en Vercel)

Última actualización: 2026-11-21. Enunciado: `proy01.md`. Entrega: 23-nov-2026.

**URL pública (parte 2): <https://lente-fiqa.vercel.app>** — verificada el 2026-11-21 (ver §10).
Revisión de compañeros, README e informe técnico: parte 3 (no hechos).

---

# PARTE 1 — Motor local y evaluación (validada; sin cambios de comportamiento)

## 1. Lo que ya existía (código dejado por Codex, conservado)
- `corpus.py`: carga/validación de FiQA (`BeIR/fiqa`, qrels de test), IDs normalizados a `str`, exporta JSONL a `data/fiqa/`. Opción `--sample-queries` para muestras.
- `retrieval.py`: `tokenize` (NFKC, casefold, regex de palabras, stopwords en inglés), `InvertedIndex` (postings término→{doc: frecuencia}) con Jaccard binario, coseno TF-IDF y BM25 (k1=1.2, b=0.75); `SemanticIndex` (sentence-transformers `all-MiniLM-L6-v2`, revisión fijada, CPU, FAISS `IndexFlatIP` con embeddings normalizados; mismo modelo para documentos y consultas; el índice se guarda y se reutiliza por hash del contenido).
- `main.py`: subcomandos `prepare`, `index`, `search`, `evaluate`.

## 2. Qué se hizo en la sesión de la parte 1
- Entorno: `.venv\Scripts\python.exe` (3.13.9) con faiss/torch/sentence_transformers/datasets. Los bloqueos del agente anterior no se reprodujeron.
- Descargué FiQA completo y construí el índice FAISS completo (~35 min en CPU, una sola vez).

## 3. Qué se verificó
- FiQA: 57 638 documentos, 648 consultas de test, 1 706 qrels; todas las referencias de qrels existen.
- Índice FAISS `data/fiqa/documents-180307632d0f….faiss` (88 532 013 bytes): `ntotal=57638`, `d=384`, producto interno.
- Evaluación completa de los cuatro métodos, k=10 (101 s con el índice ya construido).

## 4. Resultados (FiQA completo, k=10, 648 consultas) — `data/fiqa/report_k10.json`
| Método | MAP@10 | Precision@10 media | Recall@10 medio |
|---|---|---|---|
| Jaccard binario | 0.0298 | 0.0144 | 0.0589 |
| Coseno TF-IDF | 0.1159 | 0.0492 | 0.2060 |
| BM25 | 0.1726 | 0.0656 | 0.2969 |
| Semántico (MiniLM + FAISS) | 0.2914 | 0.1045 | 0.4413 |

Consulta de prueba (semántico): «can I deduct a home office if I work remotely as a freelancer» → top-3 ids 295509 (0.7468), 288537 (0.6529), 377621 (0.6165).

## 5. Significado del corte k
k = número de documentos del ranking que se evalúan (por defecto 10, `--k`).
- Precision@k = relevantes en el top-k / k. Recall@k = relevantes en el top-k / total de relevantes de la consulta.
- AP@k = suma de las precisiones en cada posición con acierto dentro del top-k, dividida entre el **total de relevantes** (no entre min(k, relevantes)). MAP = media de AP@k. Es conservador cuando hay más de k relevantes.
- FiQA tiene ~2.6 relevantes por consulta (1 706 / 648): compara métodos entre sí, no con 1.0.

## 6. Comandos de la parte 1 (desde la raíz, PowerShell)
```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
.\.venv\Scripts\python.exe main.py --data data/fiqa prepare                  # descarga y valida FiQA completo
.\.venv\Scripts\python.exe main.py index                                      # construye el índice FAISS (~35 min CPU, una vez)
.\.venv\Scripts\python.exe main.py search "consulta libre" --method bm25 --top 10   # jaccard | tfidf | bm25 | semantic
.\.venv\Scripts\python.exe main.py evaluate --k 10 --output data/fiqa/report_k10.json
```
Muestra pequeña: `main.py --data data/fiqa_muestra prepare --sample-queries 20`. `hf_xet` no está instalado (solo avisa).
**Comprobado de nuevo en la parte 2:** `search` (BM25 y Semántico con PyTorch) devuelve las mismas posiciones y puntuaciones que antes.

---

# PARTE 2 — Lente: buscador web de FiQA en Vercel

## 7. Qué se construyó
| Archivo | Papel |
|---|---|
| `app.py` | API FastAPI: `/api/health`, `/api/search`, `/api/compare`, `/api/evaluation`. Carga el motor una vez por instancia (lifespan + carga perezosa con candado). Registra tiempos de arranque. Sirve `public/` solo en local. |
| `lente_engine.py` | Abre los artefactos y responde: reutiliza `InvertedIndex.search` y `SemanticIndex.search_many` de `retrieval.py`; añade presentación, posiciones por método, solapamiento y evaluación. |
| `retrieval.py` (ampliado, sin tocar el ranking) | `InvertedIndex.export_arrays/from_arrays` + `CompactPostings` (índice léxico en arrays), `SemanticIndex.open` (abre el FAISS ya construido con un codificador intercambiable), `faiss_path` (extrae la regla del nombre del `.faiss`). |
| `onnx_encoder.py` | Codificador de consultas MiniLM con ONNX Runtime + WordPiece propio (sin torch ni `tokenizers`). |
| `build_artifacts.py` | Empaqueta `artifacts/` desde lo ya calculado (no recalcula embeddings ni métricas). |
| `public/` | Interfaz: `index.html`, `css/lente.css`, `js/app.js` (sin build), fuentes, fondos, `favicon.svg`. |
| `pyproject.toml`, `uv.lock`, `.python-version` | Dependencias de PRODUCCIÓN (17 paquetes, sin torch). `requirements.txt` sigue siendo el de la parte 1 (+`onnxruntime` para probar la web en local). |
| `vercel.json`, `.vercelignore` | Preset FastAPI, `maxDuration`, cabeceras (CSP…), lista blanca de lo que se sube. |
| `verify_encoder.py`, `verify_engine.py`, `test_lente.py`, `scripts/check_deployment.py`, `scripts/idle_probe.sh` (+`health_summary.py`) | Verificación (ver §9). |
| `CREDITS.md`, `public/img/CREDITS.md`, `scripts/prepare_background.py` | Licencias y origen de la foto, fuentes y modelo. |

## 8. Decisiones (y el porqué)

### 8.1 ¿Cabe el motor tal cual en Vercel? No: bloqueo concreto
Medido, no supuesto:
- La rueda Linux `torch==2.15.0` pesa **514 MB comprimida** (resolución manylinux x86_64/cp313); `sentence-transformers + faiss-cpu + fastapi` suman **633 MB comprimidos** (42 paquetes). Instalado en el venv: torch 552 MB + scipy 115 + transformers 115 + sklearn 45 + faiss 75 + numpy 34 ≈ **936 MB**. El build de Vercel rechaza funciones de más de **225 MB** (límite efectivo observado; la documentación habla de 500 MB para Python). El modo beta *large functions* (hasta 5 GB) permitiría subirlo, pero cargar torch solo para codificar una consulta de pocas palabras supone ~1 GB más de función y un arranque en frío bastante peor; no se probó porque ONNX resuelve el problema con el mismo modelo.
- **Decisión:** en el despliegue web, el codificador de consultas pasa de PyTorch a **ONNX Runtime** con el **mismo modelo y la misma revisión** (`sentence-transformers/all-MiniLM-L6-v2` @ `1110a243…`, grafo `onnx/model.onnx` de esa revisión), mismo preprocesado (tokenización BERT uncased, truncado a 256, mean pooling con máscara, normalización L2). El CLI sigue usando PyTorch.
- **Equivalencia demostrada** (§9): vectores con coseno ≥ 0.9999999 y |Δ| ≤ 2·10⁻⁷ frente a los vectores guardados en FAISS y frente a PyTorch; MAP/Precision/Recall **idénticos** en las 648 consultas × 4 métodos.
- Se escribió un WordPiece propio (~100 líneas) para no arrastrar `tokenizers` + `huggingface_hub` (~35 MB). Verificado contra el tokenizador oficial en 1 112 064 code points, los 57 638 documentos y las 648 consultas. Única diferencia conocida: 568 code points (0,05 %) añadidos o reclasificados en Unicode 11–15 que el tokenizador oficial (tablas más antiguas) trata como «sin asignar»; irrelevante para consultas reales.

### 8.2 Cómo llegan los archivos al despliegue
`data/` está en `.gitignore` y el `.faiss` pesa 88,5 MB; la solución es **no pasar por Git**:
1. `python build_artifacts.py` genera `artifacts/` (**204,7 MiB**) desde `data/fiqa/` copiando tal cual el `.faiss` de 57 638 vectores y `report_k10.json`, y derivando: `lexical.npz` (índice invertido en arrays), `docs.bin` (cada documento comprimido por separado, 46,9 MB → 25,2 MB), `model/` (ONNX + vocab) y `judgments.json` (qrels).
2. `artifacts/` está en `.gitignore` salvo `manifest.json` (tamaños y SHA-256 de cada archivo); se comprueban existencia y tamaño al arrancar.
3. La CLI de Vercel sube la carpeta junto al código (`.vercelignore` es una lista blanca: no viajan `data/`, `.venv/`, `requirements.txt` con torch ni la maqueta `img.png`). Subida de 28 archivos (el mayor, 90,4 MB) sin problema de tamaño por archivo.
- Descartado: commit del `.faiss` (hincha el repo, GitHub avisa por >50 MB y rechaza >100 MB), Git LFS (cuota de ancho de banda de GitHub y build dependiente de LFS), descargar al arrancar (latencia + dependencia externa en el arranque en frío), reconstruir en el build (~35 min de CPU).
- Consecuencia: los builds por `git push` **no funcionan** (el repo no contiene los artefactos); el despliegue es por CLI. No se conectó el repositorio de GitHub al proyecto. Si se quisiera CI, habría que publicar `artifacts/` como asset de release y descargarlo en el build.

### 8.3 Límites reales de Vercel (medidos en el despliegue)
| Qué | Observado |
|---|---|
| Tamaño máximo de función en el build | **225 MB** («Total bundle size (242.34 MB) exceeds the maximum function size (225 MB)») |
| Tamaño real de la función | **λ fastapi 284,1 MB** (artefactos 214,7 MB + dependencias ~66,5 MiB) |
| Modo | «exceeds the standard size limit; enabling large functions (beta)» — activado con `VERCEL_SUPPORT_LARGE_FUNCTIONS=1`. Está en `vercel.json` → `build.env` **y** como variable del proyecto (la creé a mano). No pude probar que `build.env` baste por sí solo: la CLI exige aprobación humana para borrar la variable (`vercel env rm … --confirm-name`) y no la esquivé. En un proyecto nuevo, si el build falla con «exceeds the maximum function size (225 MB)», añadirla con `vercel env add` |
| Memoria | 2 GB / 1 vCPU (Hobby, no configurable). Uso real: **RSS 351–387 MB** tras cargar todo y atender búsquedas |
| Duración máxima | 300 s (Hobby); se fijó `maxDuration: 60` |
| Build | 6 s en `iad1` (2 vCPU, 8 GB); dependencias desde `uv.lock` |
| Región | `iad1` (Washington D.C.) |
- Se reduce lo posible antes de pedir el modo beta: sin torch/tokenizers/huggingface (la ruta con tokenizers+hub habría sumado ~35 MB), corpus comprimido (46,9 MB → 25,2 MB), índice léxico en arrays (28 MB de pickle y 3,5 s de carga → 8,7 MiB y 0,35 s). Aun así no baja de 225 MB: FAISS (88,5) + ONNX (90,4) ya son 179 MB y no se pueden cambiar sin alterar el modelo.
- **Riesgo asumido:** *large functions* es beta. Si se retirara, el plan B (backend compatible fuera de Vercel) está en §12.

### 8.4 Qué se carga al arrancar y qué por consulta
- Arranque (una vez por instancia): `lexical.npz` → `InvertedIndex.from_arrays`; `FAISS read_index` (57 638 vectores, 384 d); sesión ONNX; `docs.bin` por `mmap`; `report_k10.json` y qrels. **No** se recalculan los 57 638 vectores y **no** se reconstruye el índice léxico (en local: 9,8 s para construirlo vs 0,35 s para cargarlo).
- Por consulta nueva: **un solo embedding** (el de la consulta; la comparación de los cuatro métodos reutiliza una sola codificación) + búsqueda. Comprobado con pruebas que prohíben reconstruir índices y cuentan las llamadas al codificador (§9).
- `SemanticIndex.search_many` conserva el criterio de la parte 1 (ranking completo + desempate por id), por eso las métricas coinciden al 100 %.

### 8.5 Interfaz
- Portada: foto panorámica de montañas nevadas al crepúsculo azul/violeta, «Lente», barra central y menú compacto (Jaccard · TF-IDF · BM25 · Semántico, con la familia de modelo al lado; **Semántico seleccionado al inicio**). Enlace «Evaluación FiQA» arriba a la derecha.
- Al buscar, la barra sube a una cabecera y los resultados van sobre fondo limpio (claro u oscuro según el sistema). Hay top-n (5–50), modo «Comparar los cuatro» y estados de **carga** (con aviso de arranque en frío a los 3,5 s), **sin resultados** (con alternativas), **error** (con «Reintentar») y consulta vacía.
- Comparación **por posiciones, no por puntuaciones**: cada documento muestra su posición en los cuatro métodos (Jac/TF/BM/Sem) y hay una matriz de solapamiento; las puntuaciones crudas solo se muestran dentro de cada método con la advertencia de que no son comparables (Jaccard ∈ [0,1], dos cosenos en espacios distintos, BM25 sin tope).
- Vista **Evaluación FiQA**: resumen (MAP@10, Precision@10 y Recall@10 medios, consultas con algún acierto), «dónde gana cada método» (AP@10 por consulta; p. ej. Semántico mejor que BM25 en 313 de 648, peor en 108, empate 227), tabla de las 648 consultas (métrica elegible, filtro, orden, paginación, botón «Comparar») y definiciones. Valores de `report_k10.json`, sin recalcular.
- Si la consulta coincide con una de las 648 de prueba, se marcan los documentos relevantes según los qrels.
- Accesibilidad: menú como *listbox* con teclado completo (↑↓ Inicio Fin Intro Esc y letra inicial), foco visible, enlace «Saltar al contenido», región `aria-live`, diálogo nativo para el texto completo, `prefers-reduced-motion`, `forced-colors`, contraste AA, móvil sin desbordamiento horizontal.
- Foto: «Crazy Mountains, Montana», Janusz Sobolewski, **CC BY 2.0**, Wikimedia Commons (independiente de la maqueta `img.png`, que **no** se usa); recortada y con color ajustado (documentado en `CREDITS.md`, atribución visible en el pie). Fuentes Fraunces y Hanken Grotesk (OFL), alojadas en el propio sitio.

### 8.6 Seguridad
- Sin claves ni secretos en el código ni en el despliegue. `vercel link` generó un `.env.local` con un token OIDC efímero: se borró y `.env*` está en `.gitignore`. La única variable del proyecto es `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` (no secreta).
- Cabeceras: CSP estricta (`default-src 'self'`, sin scripts ni estilos externos), `nosniff`, `Referrer-Policy`, `frame-ancestors 'none'`. Toda entrada se escapa en el cliente; la API valida `q` (1–300), `n` (1–50) y `method` y devuelve errores JSON en español.

## 9. Pruebas realizadas (todas pasan)
| Comando | Qué prueba | Resultado |
|---|---|---|
| `python verify_encoder.py` | Tokenización ONNX/WordPiece = tokenizador oficial; vectores = FAISS y PyTorch | **TODO OK**: vocabulario idéntico (30 522); 1 112 064 code points (+568 explicados por versión de Unicode); 57 638 documentos y 648 consultas idénticos; coseno mín **0,9999999**, |Δ| máx **1,9·10⁻⁷** |
| `python verify_engine.py` | MAP/P/R por consulta con el camino de producción vs `report_k10.json` | **648/648 consultas idénticas en los 4 métodos**, |ΔAP| = 0; MAP 0.029843 / 0.115946 / 0.172619 / 0.291413 |
| `python -m unittest test_lente -v` | API (4 métodos, comparar, validación 422, Unicode, vacío, qrels, evaluación), reutilización (no reconstruye índices; 0 embeddings en léxico, 1 en semántico, 1 en comparar) | **13 pruebas OK** |
| `python build_artifacts.py` | Índice léxico de arrays idéntico al original (ids y puntuaciones) en 648 consultas × 3 métodos; FAISS + ONNX abren y buscan | OK |
| `python scripts/check_deployment.py https://lente-fiqa.vercel.app` | La URL pública devuelve las mismas posiciones y puntuaciones que el motor local, y la evaluación coincide con el informe | **OK** (consulta nueva: 10/10 resultados en los 4 métodos, |Δscore| ≤ 6·10⁻⁸ — ruido de coma flotante Linux/Windows) |
| Navegador (Edge + playwright-core) | Portada, menú por teclado, 4 métodos en la UI, comparar, evaluación, móvil 390 px, modo oscuro, vacío, error simulado, aviso de arranque lento | Sin errores de consola/red; sin desbordamiento horizontal |

Latencias en la instancia de Vercel (servidor, consulta nueva): Jaccard 12,5 ms · TF-IDF 16,6 · BM25 22,9 · Semántico 53 · comparar los cuatro 94 ms. Desde Colombia, extremo a extremo: 330–510 ms con la instancia caliente.

## 10. Despliegue y arranque en frío (medido)
- Proyecto `francis-1918/lente-fiqa` (cuenta Hobby autenticada en la CLI); alias público **https://lente-fiqa.vercel.app** (el alias de producción es público; las URLs únicas de cada despliegue están tras Vercel Authentication).
- **Primera búsqueda justo tras desplegar** (sin instancia caliente): **3,18 s** extremo a extremo. Desglose (marcas de tiempo de la propia instancia frente al instante de envío local; sujeto al desfase entre relojes): ~1,7 s hasta que arranca el proceso Python (aprovisionar y montar la función de 284 MB) + 0,6 s importar dependencias + 1,0 s cargar artefactos (léxico 0,25 · ONNX 0,46 · FAISS 0,23 · calentamiento 0,06) + 0,064 s la búsqueda. RSS 354 MB.
- Una medición anterior dio 0,59 s porque Vercel ya había **precalentado** una instancia de producción; no se presenta como arranque en frío.
- Vercel escala a varias instancias (se observaron ids distintos); cada una carga los artefactos una vez.
- **Arranque en frío por inactividad:** tras 600 s sin tráfico la primera búsqueda tardó **3,53 s** (instancia nueva `59d6825b`: ~1,7 s de plataforma + 1,6 s hasta tener el motor listo + 0,067 s de búsqueda; RSS 381 MB). La sonda (`scripts/idle_probe.sh`) paró ahí; no se midieron esperas más largas.
- **Muestras de arranque en frío (primera búsqueda de una instancia nueva):** 3,18 s (tras desplegar) · 3,53 s (tras 10 min de inactividad) · 2,23 s (tras otro despliegue). Orden de magnitud: **2–3,5 s**; las siguientes búsquedas, 12–53 ms en servidor y ~0,35–0,5 s extremo a extremo.
- **Memoria real en Vercel:** RSS 351 MB al terminar la carga; 381–387 MB tras 15 peticiones (búsquedas, comparaciones y evaluación). Límite de la instancia: 2 GB.
- En la interfaz, la portada llama a `/api/health` al cargar (despierta la API sin forzar una búsqueda) y, si una búsqueda tarda más de 3,5 s, explica que el servidor se está iniciando.

## 11. Comandos exactos
```powershell
# --- una vez: generar artefactos (usa data/fiqa ya preparado e indexado en la parte 1) ---
$env:HF_HUB_DISABLE_SYMLINKS_WARNING="1"
.\.venv\Scripts\python.exe -m pip install onnxruntime==1.32.0          # ya en requirements.txt
.\.venv\Scripts\python.exe build_artifacts.py                          # ~2 min -> artifacts/ (204,7 MiB) + manifest.json

# --- probar en local ---
.\.venv\Scripts\python.exe -m uvicorn app:app --port 8000              # web + API en http://127.0.0.1:8000
.\.venv\Scripts\python.exe -m unittest test_lente -v
.\.venv\Scripts\python.exe verify_encoder.py ; .\.venv\Scripts\python.exe verify_engine.py

# --- desplegar (cuenta de la CLI: francis-1918) ---
npx vercel@latest login                                                # solo si no hay sesión
npx vercel@latest link --yes --scope francis-1918 --project lente-fiqa # 1.ª vez (crear antes: vercel project create lente-fiqa)
npx vercel@latest deploy --prod --yes --scope francis-1918             # sube código + artifacts/ y construye en Vercel
.\.venv\Scripts\python.exe scripts\check_deployment.py https://lente-fiqa.vercel.app
```
Notas: no conectar el repositorio de GitHub al proyecto (los artefactos no están en Git). Si `vercel link` pregunta por Git, decir que no. `uv.lock` se regenera con `python -m uv lock --python 3.13` tras cambiar `pyproject.toml`.

## 12. Plan B si *large functions* dejara de estar disponible
Mantener la web (`public/`) en Vercel y alojar solo la API en un contenedor: `app.py` ya admite CORS con `LENTE_CORS_ORIGINS=https://lente-fiqa.vercel.app` y el cliente acepta `<meta name="lente-api" content="https://…">` para apuntar a otro host. Sería un `Dockerfile` con `pyproject.toml` + `artifacts/` + `uvicorn app:app` (sirve en Render/HF Spaces/Fly). **No** se usó ningún otro proveedor ni se preparó cuenta; antes de hacerlo hay que decidirlo.

## 13. Incidencias y notas del entorno
- Orden de problemas al desplegar (todos resueltos): (1) «pattern "app.py" … doesn't match any Serverless Functions inside the `api` directory» → faltaba `"framework": "fastapi"` en `vercel.json`; (2) «Total bundle size (242.34 MB) exceeds the maximum function size (225 MB)» → *large functions*; (3) `/api/docs` de Swagger se rompe con la CSP → desactivado (queda `/api/openapi.json`).
- `/api/health` es público y expone id de instancia, región, memoria y tiempos de arranque (sin datos sensibles); se puede quitar antes de la entrega final si se prefiere.
- El conector MCP de Vercel de esta sesión está autenticado en **otra cuenta** (`vladimirjon`); **no se usó**. Todo se hizo con la CLI de `francis1918`. El proyecto previo `scifact-api` no se tocó.
- Un auto-add del IDE dejó los artefactos (~205 MB) en el índice de Git; se desindexaron (`git restore --staged -- artifacts`) y se purgaron los 9 blobs huérfanos (`git prune`), `.git` volvió a 3 MB. Ahora `artifacts/*` está ignorado salvo `manifest.json`. Nada se confirmó ni se subió a GitHub desde esta sesión.
- El commit `191b4c2 Pjt parte 2 preview` (añade `img.png`) es del usuario; no se modificó.
- El MCP de Playwright busca Chrome (no instalado): las pruebas de navegador se hicieron con `playwright-core` + Microsoft Edge.
- Wikimedia limita descargas de originales grandes (429); se usó la miniatura de 3840 px del propio CDN.

## 14. Pendiente (parte 3)
- Revisión de compañeros, README con instrucciones de ejecución, informe técnico (≤ 5 págs.).
- Análisis f del enunciado (en qué consultas gana/pierde el semántico frente a BM25/TF-IDF): la vista Evaluación ya da los conteos y una tabla por consulta con acceso a «Comparar»; falta redactar casos concretos.
- Confirmar y subir a GitHub el código nuevo (sin `artifacts/`).
