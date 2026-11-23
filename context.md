# Contexto — Proyecto RI (parte 1: motor y evaluación · parte 2: Lente en Vercel)

Última actualización: 2026-11-21. Enunciado: `proy01.md`. Entrega: 23-nov-2026.

**URL pública (parte 2): <https://lente-fiqa.vercel.app>** — **incluye la sugerencia ortográfica** (§8.7) desde el 2026-11-21 (producción `dpl_APYp3KJjXuTDY7wceVx8E8aZcaRY`); verificada **sin iniciar sesión** el mismo día: 31/31 comprobaciones de interfaz, API y `check_deployment.py` OK (§16).
Origen de esa versión: el Preview <https://lente-fiqa-pysd7yjh1-francis-1918.vercel.app> (protegido por Vercel Authentication, no es público), validado antes en local y en el propio Preview (§9, §15).
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
| `spelling.py` | Sugerencia ortográfica para el «Sin resultados» léxico (§8.7). Módulo aislado; `lente_engine.py` solo lo llama con 0 resultados léxicos. **Está en la lista blanca de `.vercelignore`** (sin eso no viaja a Vercel y la función falla al importarlo). |
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

### 8.7 Sugerencia ortográfica en «Sin resultados» (Jaccard, TF-IDF, BM25)
- **Qué hace:** si un método léxico devuelve 0 resultados y hay una corrección clara, el estado vacío muestra primero «¿Quisiste decir **mortgage**?». La consulta original se queda en la caja, el título y la URL hasta que se pulsa; al pulsar se ejecuta la corregida **con el mismo método y top-n** (`#/buscar?q=mortgage&m=<método>&n=<n>`). «Buscar con Semántico» y «Comparar los cuatro métodos» siguen igual. Sin candidata clara, el estado vacío es el de siempre.
- **Dónde se engancha:** `Engine.search` llama a `Speller.suggest` solo si `not hits and method in LEXICAL`; `/api/search` añade `suggestion` (`{query, changes:[{from,to,distance,documents}]}` o `null`). Semántico, `/api/compare` y los rankings no cambian (hay un test que prohíbe consultar el corrector en esos caminos). Como todos los términos corregidos están en el vocabulario, la consulta corregida siempre devuelve resultados.
- **Reutiliza el índice léxico:** vocabulario = `term_index`; frecuencia = nº de documentos por término (`np.diff(offsets)`), sin copias ni cálculo en el arranque. El diccionario de sugerencias (términos ASCII alfabéticos con df ≥ 10: **14 537 de 78 180**) se agrupa por longitud la primera vez que hace falta (~16 ms).
- **Distancia:** Damerau-Levenshtein restringida (OSA): inserción, omisión, sustitución y transposición de dos letras contiguas = 1 cada una. Implementada dos veces (`edit_distance` de referencia y `_distances` vectorizada con numpy sobre el diccionario); 3 422 pares reales sin ninguna discrepancia y un test las compara.
- **Regla de decisión** (constantes al inicio de `spelling.py`): palabras de 4 a 24 letras; ≤ 1 edición hasta 6 letras y ≤ 2 desde 7; puntuación `df / 25^distancia`; solo se sugiere si la mejor supera 6× a la segunda; palabras tan cerca de una palabra vacía como de la candidata (p. ej. «wiht») se descartan; en consultas de varias palabras deben corregirse **todas** o no hay sugerencia. Se usa el df en el diccionario porque FiQA tiene erratas propias («mortage» en 19 documentos, «teh» en 8, «recieve» en 14).
- **Calibración (datos reales de FiQA, no a ojo):** erratas sintéticas de palabras frecuentes ponderadas por df (1 200 con una errata): sugiere en ~83 % y acierta 98,4 % (los «fallos» son ambigüedades reales, p. ej. «uext» a 1 edición de «text» y de «next»); cadenas aleatorias de 4–12 letras: 1,4 % reciben sugerencia (casi todas de 4–6 letras); teclazos («asdf», «qwer», «hjkl», «zzzz») ninguna. Con 2 ediciones ya desde 6 letras, «jwriki»→«wiki» y «mikelf»→«likely» (8 % de las cadenas de 6 letras): por eso el límite está en 7. **Ojo: esas cifras valen para palabras frecuentes.** Con palabras que nunca intervinieron en la calibración (más raras) el acierto entre las sugeridas baja a **94,4 %** y sugiere en el **76 %** (validación adicional en §9); no se ha prometido ni se debe asumir cobertura universal.
- **Coste:** 1–18 ms por consulta vacía en local (solo se paga en el estado vacío); sin cambios en el arranque.
- **Hallazgo importante — «morgage» no llega al estado vacío:** aparece en **5 documentos de FiQA**, así que Jaccard/TF-IDF/BM25 devuelven 5 resultados y la UI no muestra sugerencia. El corrector sí propone «mortgage» para ella (test `test_misspelling_that_exists_in_the_corpus_is_not_an_empty_state`). «moretgage» (0 documentos) sí activa la sugerencia. Mostrarla para «morgage» exigiría ampliar el disparador a palabras raras de la consulta (cambia el flujo normal); **no se hizo** porque el encargo acota la función al estado vacío. Lo mismo vale para cualquier errata que exista en FiQA, aunque sea en un solo documento (p. ej. «liablity», «guarenteed», «qualifed» en 1 documento; «brokrage» en 4; «deductable» en 15). **Comprobado en producción** (§16): «morgage» devuelve 5 resultados con Jaccard, TF-IDF y BM25 y no muestra sugerencia. Decisión pendiente del usuario.
- **Límites conocidos:** solo inglés/ASCII; no corrige palabras con dígitos ni de menos de 4 letras; consultas mixtas donde una palabra no tiene candidata clara («moretgage xqzvwkjh», «teh mortgge») no reciben sugerencia.
- **Teclado:** la sugerencia es un enlace (`<a class="btn">`) como los otros dos botones: se llega con Tab y se activa con **Intro**; Espacio no activa enlaces (comportamiento nativo, igual que «Buscar con Semántico»). Tras pulsar, el foco va al título de resultados y el anuncio `aria-live` del estado vacío incluye «¿Quisiste decir …?». El texto sugerido se escapa (probado con una sugerencia hostil simulada).

## 9. Pruebas realizadas (todas pasan)
| Comando | Qué prueba | Resultado |
|---|---|---|
| `python verify_encoder.py` | Tokenización ONNX/WordPiece = tokenizador oficial; vectores = FAISS y PyTorch | **TODO OK**: vocabulario idéntico (30 522); 1 112 064 code points (+568 explicados por versión de Unicode); 57 638 documentos y 648 consultas idénticos; coseno mín **0,9999999**, |Δ| máx **1,9·10⁻⁷** |
| `python verify_engine.py` | MAP/P/R por consulta con el camino de producción vs `report_k10.json` | **648/648 consultas idénticas en los 4 métodos**, |ΔAP| = 0; MAP 0.029843 / 0.115946 / 0.172619 / 0.291413 |
| `python -m unittest test_lente -v` | API (4 métodos, comparar, validación 422, Unicode, vacío, qrels, evaluación), reutilización (no reconstruye índices; 0 embeddings en léxico, 1 en semántico, 1 en comparar) **+ sugerencia ortográfica** (§8.7) | **33 pruebas OK** (13 previas + 7 de API de la sugerencia + 13 del corrector con vocabulario de juguete) |
| `python verify_engine.py` (repetido tras añadir la sugerencia) | Rankings/métricas intactos | **648/648 idénticas en los 4 métodos**, |ΔAP| = 0 (MAP 0.029843 / 0.115946 / 0.172619 / 0.291413) |
| `python build_artifacts.py` | Índice léxico de arrays idéntico al original (ids y puntuaciones) en 648 consultas × 3 métodos; FAISS + ONNX abren y buscan | OK |
| `python scripts/check_deployment.py https://lente-fiqa.vercel.app` | La URL pública devuelve las mismas posiciones y puntuaciones que el motor local, y la evaluación coincide con el informe | **OK** (consulta nueva: 10/10 resultados en los 4 métodos, |Δscore| ≤ 6·10⁻⁸ — ruido de coma flotante Linux/Windows) |
| Navegador (Edge + playwright-core) | Portada, menú por teclado, 4 métodos en la UI, comparar, evaluación, móvil 390 px, modo oscuro, vacío, error simulado, aviso de arranque lento | Sin errores de consola/red; sin desbordamiento horizontal |

**Sugerencia ortográfica — qué se probó (2026-11-21).** Datos reales de FiQA; «moretgage» y «mortgage» se comprobaron en los tres métodos léxicos:
| Caso | Resultado observado |
|---|---|
| «moretgage» (Jaccard, TF-IDF, BM25) | 0 resultados + sugerencia «mortgage» (2 285 documentos, distancia 1); la caja y la URL conservan «moretgage»; solo se pidió la original al servidor |
| «morgage» (los tres) | **5 resultados normales, sin sugerencia** (existe en 5 documentos; ver §8.7). `Speller.suggest("morgage")` sí devuelve «mortgage» |
| «mortgage» (los tres y Semántico) | resultados normales, sin sugerencia |
| «xqzvwkjh», «asdfghjkl» (los tres) | estado vacío de siempre, sin sugerencia, con sus dos botones |
| «moretgage» con Semántico | 10 resultados normales, sin sugerencia |
| Clic (BM25, top-n 20) | `#/buscar?q=mortgage&m=bm25&n=20`, 20 resultados, 1 petición `method=bm25&n=20`; «Atrás» vuelve al vacío con «moretgage» |
| Teclado (Jaccard) | 4 Tab desde el título llegan al botón; contorno de foco 3 px; **Intro** ejecuta la corregida (`m=jaccard&n=10`, 10 resultados) y el foco pasa al título |
| «Buscar con Semántico» / «Comparar los cuatro métodos» | siguen funcionando con la consulta original |
| Otros | sugerencia hostil simulada escapada (sin inyección); 390 px sin desbordamiento; modo oscuro; sin errores de consola ni de red |
Batería del navegador (Edge + `playwright-core`, script temporal fuera del repositorio): **43/43 en local y 43/43 en el Preview**. En el Preview, `check_deployment.py` (con la cabecera de Trusted Sources) dio **OK**: mismas posiciones que el motor local (|Δscore| ≤ 6·10⁻⁸) y MAP idéntico a `report_k10.json`.

**Validación adicional de la sugerencia (solo lectura, 2026-11-21; sin cambiar umbrales, rankings ni despliegues).** Palabras de FiQA distintas de «mortgage».
- **Candidatos pedidos:** «credti→credit», «investmnt→investment», «divvidend→dividend», «insurence→insurance». Las cuatro erratas tienen **0 documentos** en FiQA (cero resultados léxicos) y los objetivos 4 424 / 4 295 / 916 / 2 170: sin ambigüedad. **Aviso:** el par exacto «investmnt→investment» salió en mis muestras de calibración (se reprodujeron con las mismas semillas), así que se añadieron casos con palabras que nunca intervinieron en ella: «currnecy→currency», «paychek→paycheck», «penallty→penalty», «discoumt→discount».
- **Premisa falsa registrada:** «liablity», «guarenteed» (1 documento cada una), «brokrage» (4), «deductable» (15) y «qualifed» (1) **existen en FiQA**: hay resultados, no estado vacío y no hay sugerencia (5 de ~36 erratas inventadas a mano, 14 %). Es el caso de «morgage» (§8.7); el corrector solo sí las corrige.
- **Batería: 23 consultas × 3 métodos** (Jaccard, TF-IDF y BM25 coincidieron siempre; el Preview dio exactamente lo mismo que el local):
| Consulta | Esperada | Resultados iniciales | Sugerencia observada | Resultado |
|---|---|---|---|---|
| credti · investmnt · divvidend · insurence | credit · investment · dividend · insurance | 0 | la esperada (las cuatro) | pasó |
| currnecy · paychek · penallty · discoumt (nuevas) | currency · paycheck · penalty · discount | 0 | la esperada (las cuatro) | pasó |
| divvidend insurence · penallty currnecy (varias palabras, ambas sin resultados) | dividend insurance · penalty currency | 0 | la esperada | pasó |
| dividend (bien escrita) | ninguna | 10 | ninguna | pasó |
| morgage | ninguna en la interfaz (`spelling.py` solo: mortgage) | 5 | ninguna | pasó |
| qxjvbzkw (sin corrección clara) | ninguna | 0 | ninguna | pasó |
| proect · chaged (dos alternativas: project/protect, changed/charged; razón ~1,1) | ninguna | 0 | ninguna | pasó |
| credti xqzvwkjh (una palabra sin arreglo) | ninguna (no se corrige a medias) | 0 | ninguna | pasó |
| pensionn · pensoin | pension | 0 | ninguna | falló |
| cerdti (2 ediciones, 6 letras) · cred1t (dígito) · txa (3 letras) | credit · credit · tax | 0 | ninguna | falló (límites de diseño) |
| liablity · guarenteed | liability · guaranteed | 1 | ninguna | no aplica (existen en FiQA) |
  Conteo sobre las 23 consultas: **10 sugerencias correctas, 0 incorrectas, 13 sin sugerencia** (6 correctas porque no debía haberla; 5 correcciones obvias que no se ofrecen: pensionn, pensoin, cerdti, cred1t, txa; 2 sin estado vacío). Estas 10/10 son una batería elegida a mano, **no una tasa de cobertura**.
- **Por qué «pensionn» y «pensoin» se quedan sin sugerencia:** *pension* (444 documentos, puntuación 17,8) compite con su plural *pensions* (166 → 6,6; razón 2,7) y, en la transposición, con *person* a 2 ediciones (2 842 → 4,6; razón 3,9); ambas < 6, el corrector prefiere callar.
- **Evaluación independiente (1 600 erratas, 400 por tipo; palabras que nunca se usaron al calibrar; errata inexistente en FiQA; par no visto):**
| Tipo | Correctas | Incorrectas | Sin sugerencia | Acierto si sugiere |
|---|---|---|---|---|
| Inserción | 345 | 1 | 54 | 99,7 % |
| Omisión | 245 | 27 | 128 | 90,1 % |
| Sustitución | 282 | 19 | 99 | 93,7 % |
| Transposición | 279 | 21 | 100 | 93,0 % |
| **Total** | **1 151 (71,9 %)** | **68 (4,3 %)** | **381 (23,8 %)** | **94,4 %** (sugiere en el 76,2 %) |
  Por longitud: en omisión, sustitución y transposición el acierto es 81–84 % con 4–6 letras y 92–99 % con 7 o más; la inserción acierta 99–100 % en ambos tramos. Causas de «sin sugerencia» (submuestra de 500): 95 % ambigüedad con una competidora de puntuación cercana (casi siempre plural/singular o una palabra común), 5 % objetivo con menos de 10 documentos. Las incorrectas suelen ser erratas que también son truncaciones válidas de otra palabra más frecuente («demande»→*demand* en vez de *demanded*, «marginaly»→*marginal*). Las palabras nuevas son más raras que las de la calibración (df mediano 67 frente a 95), de ahí que las cifras de §8.7 (98,4 % / 83 %) no se generalicen.
- **Interfaz con casos nuevos (clic y teclado):** 19/19 en local y 19/19 en el Preview. «paychek» con TF-IDF y n=30 → clic → `#/buscar?q=paycheck&m=tfidf&n=30`, 30 resultados, una sola petición `method=tfidf&n=30`; «discoumt» con Jaccard y n=20 → 4 Tab + **Intro** → `q=discount&m=jaccard&n=20`; «divvidend insurence» con BM25 y n=5 → «¿Quisiste decir dividend insurance?» → 5 resultados; los controles (proect, chaged, qxjvbzkw, pensionn) conservan el vacío de siempre con sus dos botones y «Buscar con Semántico» sigue funcionando.
- **Qué funciona y qué no:** funciona mejor la inserción y las palabras de 7+ letras, y las consultas de varias palabras cuando cada palabra tiene candidata clara. Quedan fuera: erratas que ya existen en FiQA; 2 ediciones en palabras de menos de 7 letras; palabras de menos de 4 letras; dígitos y no ASCII; candidatas ambiguas (flexiones, p. ej. singular/plural); consultas con una palabra irreparable; y, por diseño, Semántico y la vista de comparación.

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
# o: publicar un Preview ya probado (Vercel lo reconstruye como producción; así se publicó la sugerencia, §16):
npx vercel@latest promote https://<preview>.vercel.app --scope francis-1918 --yes
npx vercel@latest inspect <url-del-despliegue> --wait --scope francis-1918   # espera a que termine la construcción
.\.venv\Scripts\python.exe scripts\check_deployment.py https://lente-fiqa.vercel.app
# volver a la versión anterior si algo falla: npx vercel@latest promote <url-del-despliegue-anterior> --scope francis-1918 --yes
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
- `.vercelignore` es una lista blanca: todo módulo Python nuevo necesita su `!archivo.py` (se añadió `!spelling.py`; sin eso la función falla al importarlo).
- En Git Bash, `npx vercel api /v6/...` falla («Invalid arguments») porque MSYS convierte `/v6/...` en una ruta de Windows: usar `MSYS_NO_PATHCONV=1`. `GET /v6/deployments/<id>/files` da el SHA-1 de cada archivo desplegado y sirve para comprobar byte a byte que un despliegue es el árbol de trabajo.
- En las baterías de navegador, `net::ERR_ABORTED` del fondo (`/img/…`) o del `favicon.svg` es la descarga cancelada al navegar a `about:blank` entre escenarios; en sesión única (carga → búsqueda → sugerencia → clic, esperando a la red) hay 0 fallos y 0 respuestas ≥ 400. Cualquier otro fallo de red o de consola se trata como error.

## 14. Pendiente (parte 3)
- Revisión de compañeros, README con instrucciones de ejecución, informe técnico (≤ 5 págs.).
- Análisis f del enunciado (en qué consultas gana/pierde el semántico frente a BM25/TF-IDF): la vista Evaluación ya da los conteos y una tabla por consulta con acceso a «Comparar»; falta redactar casos concretos.
- Confirmar y subir a GitHub el código nuevo (sin `artifacts/`). **La sugerencia ya está en producción pero sigue sin commit** (§16).
- Decidir si «morgage» (existe en 5 documentos) y, en general, las erratas que existen en FiQA deben también recibir sugerencia, lo que exigiría ampliar el disparador más allá del estado vacío y cambiaría el flujo normal (§8.7). El encargo actual lo excluye.
- Si se quisiera cerrar los límites de §9 (flexiones singular/plural, palabras de 4–6 letras), habría que recalibrar con la evaluación independiente de §9, no con las muestras de palabras frecuentes de §8.7.

## 15. Preview de la sugerencia ortográfica (2026-11-21)
- Despliegue: `npx vercel@latest deploy --yes --scope francis-1918` (sin `--prod`) → **<https://lente-fiqa-pysd7yjh1-francis-1918.vercel.app>**, id `dpl_Hsyg6nnCHiG5kHrWWP8xNjmyspEC`, entorno `preview`, READY. El build con *large functions* funcionó también en Preview (`build.env` de `vercel.json` basta, o la variable del proyecto aplica a Preview).
- Está protegido por Vercel Authentication. Para probarlo sin desactivar la protección: `npx vercel@latest curl <url>/api/... --scope francis-1918` para HTTP, y para el navegador/scripts la cabecera `x-vercel-trusted-oidc-idp-token` con el token que da `npx vercel@latest env run --scope francis-1918 -- <comando>` (variable `VERCEL_OIDC_TOKEN`; no se imprime ni se guarda).
- Cuando se creó el Preview, producción (`lente-fiqa.vercel.app`) no se había tocado (`/api/search?q=moretgage&method=bm25` devolvía 0 resultados y ninguna clave `suggestion`). **Más tarde ese mismo Preview se promovió a producción (§16).**
- Los cambios están en el árbol de trabajo **sin commit** (`spelling.py` nuevo; modificados `lente_engine.py`, `app.py`, `public/js/app.js`, `test_lente.py`, `.vercelignore`); nada se subió a GitHub.

## 16. Publicación en producción y verificación (2026-11-21)
- **Comprobaciones previas.** Git: solo cambios de la mejora (`spelling.py` nuevo; `lente_engine.py`, `app.py`, `public/js/app.js`, `.vercelignore`, `test_lente.py` y este archivo), sin commit y sin cambios ajenos. Proyecto: `.vercel/project.json` = `prj_S2dGmdKRFmCyN7hnZKYNC6AoU9mQ` / `team_IwWVdBENf3IiRo8RsjUSbkmU` = `francis-1918/lente-fiqa`; la CLI está autenticada como `francis1918`. El Preview `dpl_Hsyg6nnCHiG5kHrWWP8xNjmyspEC` pertenece a ese proyecto, se creó a las 22:31:38 UTC (todos los archivos que viajan son anteriores) y sus **29 archivos de código/datos tienen el mismo SHA-1 que el árbol de trabajo** (incluidos `spelling.py`, `lente_engine.py`, `app.py`, `public/js/app.js` y `vercel.json`; `.vercelignore` no se sube; `out/fastapi` es la función construida). Conclusión: la versión probada estaba completa, no hizo falta añadir nada.
- **Publicación:** `npx vercel@latest promote https://lente-fiqa-pysd7yjh1-francis-1918.vercel.app --scope francis-1918 --yes`. Vercel creó un despliegue de producción nuevo (reconstruido; 19 s): **`dpl_APYp3KJjXuTDY7wceVx8E8aZcaRY`** (<https://lente-fiqa-griunldw2-francis-1918.vercel.app>), *Ready*, `target: production`, con los alias `lente-fiqa.vercel.app` y `lente-fiqa-francis-1918.vercel.app`. Sus 30 archivos tienen los mismos hashes que los del Preview.
- **Versión anterior (por si hay que volver):** `dpl_Dumc8s8MSk5p5u1EXn3mZEVairsF` (<https://lente-fiqa-9x9xtbj3v-francis-1918.vercel.app>, producción de las 16:04 COT, sin sugerencia). No hizo falta recuperarla.
- **Pruebas en <https://lente-fiqa.vercel.app>, sin sesión, sin cookies y sin cabeceras especiales:**
  - API pública: «moretgage» (0 resultados → «mortgage») y «credti» (0 → «credit») con Jaccard, TF-IDF y BM25; «mortgage» 5 resultados (n=5) sin sugerencia; «xqzvwkjh» 0 resultados sin sugerencia; Semántico con «moretgage» 5 resultados sin sugerencia.
  - Interfaz (Edge, contexto limpio) **31/31**: la página carga sin pasar por un login de Vercel; «moretgage» y «credti» muestran «¿Quisiste decir mortgage/credit?» en los tres métodos léxicos, con la consulta original intacta y una sola petición; clic con BM25 y n=20 (`#/buscar?q=mortgage&m=bm25&n=20`, 20 resultados) y con TF-IDF y n=30 (`q=credit&m=tfidf&n=30`, 30 resultados), ambos con una sola petición con el método y el top-n originales; teclado (4 Tab + **Intro**, Jaccard, n=20 → `q=credit`, 20 resultados); «mortgage» 10 resultados normales sin sugerencia en los tres métodos; «xqzvwkjh» estado vacío de siempre sin sugerencia en los tres; «morgage» 5 resultados sin sugerencia; Semántico (por defecto) con «mortgage» y «moretgage» 10 resultados; «Buscar con Semántico» y «Comparar los cuatro métodos» desde el vacío de «moretgage» funcionan; Comparar «mortgage» muestra 4 columnas, posiciones por método y matriz de solapamiento; sin errores de consola ni de red (ver nota de `ERR_ABORTED` en §13).
  - `python scripts/check_deployment.py https://lente-fiqa.vercel.app` → **OK**: mismas posiciones que el motor local en los 4 métodos y en comparar (|Δscore| ≤ 6·10⁻⁸) y MAP idéntico a `report_k10.json` (0.0298 / 0.1159 / 0.1726 / 0.2914).
- **URL verificada:** <https://lente-fiqa.vercel.app> (producción, pública).
- **No comprobado:** el arranque en frío tras esta publicación no se volvió a medir (las cifras de §10 son de despliegues anteriores); modo oscuro y 390 px se probaron en local y en el Preview, no en producción; el corrector por separado solo existe en local (no hay endpoint). Ningún bloqueo de acceso ni de permisos.

### Corrección de apilamiento (`z-index`) en la vista principal (`home`)

- **Archivo modificado:** `public/css/lente.css`
- **Problema:** En la vista inicial (`body[data-view="home"]`), el menú desplegable de selección de modelo (`.method-list`) quedaba visualmente por debajo del texto de créditos de la imagen de fondo (`.foot-credit` dentro de `.foot`). Esto ocurría porque el contenedor padre `body[data-view="home"] .top` tenía `z-index: 5`, mientras que `body[data-view="home"] .foot` tenía `z-index: 6`, creando un contexto de apilamiento inferior al pie de página.
- **Solución:** Se elevó el contexto de apilamiento de la cabecera en la vista de inicio cambiando `z-index: 5` a `z-index: 30` en `body[data-view="home"] .top` (manteniendo `.method-list` en `z-index: 50` y `.foot` en una capa inferior), garantizando que el desplegable se superponga correctamente sobre el pie de página.