"""API de Lente (FastAPI). Local: `uvicorn app:app --port 8000`; en Vercel es la función Python.

    GET /api/health                       estado, tiempos de arranque, memoria, instancia
    GET /api/search?q=&method=&n=         top-n de un método (jaccard | tfidf | bm25 | semantic);
                                          con 0 resultados léxicos añade `suggestion` (o null)
    GET /api/compare?q=&n=                los cuatro métodos para la misma consulta (posiciones)
    GET /api/evaluation                   valores reales de report_k10.json (k=10, 648 consultas)

Los artefactos se cargan una vez por instancia (al arrancar o en la primera petición) y se
reutilizan entre consultas; por consulta solo se calcula el embedding de esa consulta.
"""

import time

BOOT = time.time()          # lo primero: así se mide también el coste de importar las dependencias

from contextlib import asynccontextmanager
import os
import threading
from typing import Annotated, Literal
import uuid

from fastapi import FastAPI, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from lente_engine import Engine, memory_mb

INSTANCE = {"id": uuid.uuid4().hex[:8], "started": BOOT, "requests": 0, "searches": 0}
_state = {"engine": None, "error": None, "ready_at": None, "first": None}
_lock = threading.Lock()

Method = Literal["jaccard", "tfidf", "bm25", "semantic"]
Text = Annotated[str, Query(min_length=1, max_length=300, description="Consulta de texto libre")]
Top = Annotated[int, Query(ge=1, le=50, description="Cuántos resultados por método")]


def get_engine():
    if _state["engine"] is not None:
        return _state["engine"]
    with _lock:
        if _state["engine"] is None:
            if _state["error"] is not None:
                raise HTTPException(503, {"code": "engine_unavailable", "message": _state["error"]})
            try:
                _state["engine"] = Engine()
                _state["ready_at"] = time.time()
            except Exception as error:  # el detalle va al log y al /api/health
                _state["error"] = f"{type(error).__name__}: {error}"
                print("ERROR cargando artefactos:", _state["error"], flush=True)
                raise HTTPException(503, {"code": "engine_unavailable", "message": _state["error"]})
    return _state["engine"]


@asynccontextmanager
async def lifespan(_):
    try:
        get_engine()
    except HTTPException:
        pass  # la API arranca igualmente y responde 503 con el motivo
    yield


# Sin /docs interactivo: Swagger UI se carga desde un CDN externo y la CSP (script-src 'self') lo bloquearía.
app = FastAPI(title="Lente · FiQA", version="2.0", lifespan=lifespan,
              docs_url=None, redoc_url=None, openapi_url="/api/openapi.json")


if os.environ.get("LENTE_CORS_ORIGINS"):
    # Solo para el backend alternativo (web en Vercel, API en otro host); en Vercel todo es mismo origen.
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(CORSMiddleware, allow_methods=["GET"], allow_headers=["*"],
                       allow_origins=[origin.strip() for origin in os.environ["LENTE_CORS_ORIGINS"].split(",")])


@app.middleware("http")
async def time_first_request(request: Request, call_next):
    """Registra la primera petición real (no /api/health) de cada instancia: sirve para medir arranques en frío."""
    if _state["first"] is not None or request.url.path == "/api/health":
        return await call_next(request)
    arrived = time.time()
    _state["first"] = {"path": request.url.path}
    response = await call_next(request)
    finished = time.time()
    _state["first"] = {
        "path": request.url.path, "status": response.status_code,
        "arrived_after_boot_s": round(arrived - BOOT, 3),
        "engine_ready_before_arrival": bool(_state["ready_at"] and _state["ready_at"] <= arrived),
        "handler_s": round(finished - arrived, 3),
    }
    return response


@app.exception_handler(HTTPException)
async def http_error(_: Request, error: HTTPException):
    detail = error.detail if isinstance(error.detail, dict) else {"code": "error", "message": str(error.detail)}
    return JSONResponse({"error": detail}, status_code=error.status_code,
                        headers={"Cache-Control": "no-store"})


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, error: RequestValidationError):
    first = error.errors()[0]
    field = ".".join(str(part) for part in first["loc"] if part != "query")
    messages = {"q": "Escribe una consulta de 1 a 300 caracteres.",
                "n": "n debe estar entre 1 y 50.",
                "method": "Método desconocido: usa jaccard, tfidf, bm25 o semantic."}
    return JSONResponse({"error": {"code": "invalid_request",
                                   "message": messages.get(field, f"Petición inválida ({field})")}},
                        status_code=422, headers={"Cache-Control": "no-store"})


def clean(query):
    query = " ".join(query.split())
    if not query:
        raise HTTPException(422, {"code": "invalid_request", "message": "Escribe una consulta."})
    return query

#devule el estado del motor en lo que tarde cargar
@app.get("/api/health") # comprueba el estado del motor cuantos documentos cargo y cuanto tardara en arracar
def health(response: Response):#responde y tarde en cargar el motor y si esta listo o no
    response.headers["Cache-Control"] = "no-store"
    engine = _state["engine"]
    body = {"ready": engine is not None, "error": _state["error"],
            "instance": {**INSTANCE, "uptime_s": round(time.time() - INSTANCE["started"], 1),
                         "vercel": bool(os.environ.get("VERCEL")),
                         "region": os.environ.get("VERCEL_REGION")},
            "memory_mb": memory_mb(),
            "cold_start": {"boot_to_engine_ready_s": round(_state["ready_at"] - BOOT, 3) if _state["ready_at"] else None,
                           "first_request": _state["first"]}}
    if engine is not None:
        body.update(engine.info())
    return body


@app.get("/api/search")
def search(response: Response, q: Text, method: Method = "semantic", n: Top = 10): #recibe el texot que escribir
    response.headers["Cache-Control"] = "no-store"#SI NO Bada us semanticoUSCO NA
    INSTANCE["requests"] += 1 #el metodo que eligi
    INSTANCE["searches"] += 1 # y cuantos resultados que quoier ver
    return get_engine().search(clean(q), method, n)
# app.py pasa esos datos al motor que hace la busqueda y devuelve los resultados
# esta ruta pasa la consulta al motor y devuelve los resultados en json

@app.get("/api/compare")
def compare(response: Response, q: Text, n: Top = 10):
    response.headers["Cache-Control"] = "no-store"
    INSTANCE["requests"] += 1
    INSTANCE["searches"] += 1
    return get_engine().compare(clean(q), n)


@app.get("/api/evaluation") # ahi es donde la web recibe las metricas de precisioin recall  y map@10
def evaluation(response: Response):
    response.headers["Cache-Control"] = "public, max-age=300, s-maxage=86400"
    INSTANCE["requests"] += 1
    return get_engine().evaluation() #entrega las metricas del reporte de evaluacion  del def compare y metricas precalculadas


if not os.environ.get("VERCEL"):
    # En Vercel, public/ lo sirve la CDN; en local lo servimos nosotros.
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory="public", html=True), name="public")
