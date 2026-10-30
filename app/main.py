"""Aplicación principal FastAPI para el servicio Academic RAG.

Configura middlewares (CORS), manejadores globales de excepciones estandarizados,
inclusión de enrutadores para documentos y preguntas, y endpoints de salud.
"""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator, Dict

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.documentos.router import router as documentos_router
from app.preguntas.router import router as preguntas_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gestiona el ciclo de vida de la aplicación: preparación de directorios y recursos."""
    settings_fn = app.dependency_overrides.get(get_settings, get_settings)
    active_settings = settings_fn()
    try:
        Path(active_settings.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(active_settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    yield


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Servicio RAG universitario para el sistema Academic. "
        "Permite indexar apuntes académicos en PDF y responder preguntas "
        "con citas verificables de documento y número de página."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# -------------------------------------------------------------------------
# Configuración de CORS
# -------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------------
# Manejadores globales de errores con formato estándar del proyecto:
# {"error": {"codigo": "...", "mensaje": "..."}}
# -------------------------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Captura excepciones HTTP y devuelve el formato normalizado del proyecto."""
    detail = exc.detail
    if isinstance(detail, dict) and "codigo" in detail and "mensaje" in detail:
        error_body = detail
    else:
        # Mapear código por defecto según código de estado HTTP
        code_map = {
            status.HTTP_400_BAD_REQUEST: "SOLICITUD_INVALIDA",
            status.HTTP_401_UNAUTHORIZED: "NO_AUTORIZADO",
            status.HTTP_403_FORBIDDEN: "ACCESO_PROHIBIDO",
            status.HTTP_404_NOT_FOUND: "NO_ENCONTRADO",
            status.HTTP_409_CONFLICT: "CONFLICTO",
            status.HTTP_422_UNPROCESSABLE_ENTITY: "ENTIDAD_NO_PROCESABLE",
            status.HTTP_500_INTERNAL_SERVER_ERROR: "ERROR_INTERNO",
        }
        codigo = code_map.get(exc.status_code, "ERROR_HTTP")
        mensaje = str(detail) if detail else "Error en la solicitud HTTP."
        error_body = {"codigo": codigo, "mensaje": mensaje}

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": error_body},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Captura errores de validación de Pydantic y formatea en la estructura requerida."""
    mensajes_error = []
    for err in exc.errors():
        campo = " -> ".join([str(loc) for loc in err.get("loc", []) if str(loc) != "body"])
        msg = err.get("msg", "Dato inválido")
        mensajes_error.append(f"{campo}: {msg}" if campo else msg)

    mensaje_unificado = "; ".join(mensajes_error) if mensajes_error else "Error de validación en la solicitud."

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "codigo": "ERROR_VALIDACION",
                "mensaje": mensaje_unificado,
            }
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Captura cualquier excepción no controlada."""
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "codigo": "ERROR_INTERNO_SERVIDOR",
                "mensaje": f"Ocurrió un error inesperado en el servidor: {str(exc)}",
            }
        },
    )


# -------------------------------------------------------------------------
# Inclusión de Routers versionados
# -------------------------------------------------------------------------
app.include_router(documentos_router)
app.include_router(preguntas_router)


# -------------------------------------------------------------------------
# Endpoints raíz y de salud
# -------------------------------------------------------------------------
@app.get(
    "/",
    tags=["General"],
    summary="Información general de la API",
)
async def root() -> Dict[str, Any]:
    """Endpoint raíz con estado básico e hipervínculos de documentación."""
    return {
        "sistema": settings.app_name,
        "version": settings.app_version,
        "estado": "operativo",
        "documentacion": "/docs",
        "endpoints": {
            "subir_documento": "POST /api/v1/documentos",
            "listar_documentos": "GET /api/v1/documentos",
            "consultar_apuntes": "POST /api/v1/preguntar",
            "salud": "GET /api/v1/health",
        },
    }


@app.get(
    "/api/v1/health",
    tags=["General"],
    summary="Chequeo de salud del servicio",
)
async def health_check(
    active_settings: Settings = Depends(get_settings),
) -> Dict[str, Any]:
    """Verifica el estado operativo del servicio y del almacenamiento vectorial."""
    chroma_ok = False
    try:
        from app.documentos.service import DocumentosService

        svc = DocumentosService(settings=active_settings)
        svc.collection.count()
        chroma_ok = True
    except Exception:
        chroma_ok = False

    return {
        "status": "healthy" if chroma_ok else "degraded",
        "chromadb": "connected" if chroma_ok else "unavailable",
        "llm_provider": active_settings.llm_provider,
        "embedding_model": active_settings.embedding_model,
        "version": active_settings.app_version,
    }
