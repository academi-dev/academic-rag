"""Router FastAPI para los endpoints de gestión de documentos.

Define las rutas REST para:
- POST /api/v1/documentos: Subida y procesamiento multipart de archivos PDF.
- GET /api/v1/documentos: Listado paginado de documentos indexados.
- GET /api/v1/documentos/{id}: Consulta de metadatos de un documento por su ID.
- DELETE /api/v1/documentos/{id}: Eliminación de un documento y sus vectores.
"""

from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from app.config import Settings, get_settings
from app.documentos.schemas import (
    DocumentoResponse,
    DocumentoUploadResponse,
    DocumentosListResponse,
    ErrorResponse,
)
from app.documentos.service import DocumentosService

router = APIRouter(
    prefix="/api/v1/documentos",
    tags=["Documentos"],
    responses={
        400: {"model": ErrorResponse, "description": "Solicitud inválida"},
        404: {"model": ErrorResponse, "description": "Documento no encontrado"},
        422: {"model": ErrorResponse, "description": "Error de validación"},
        500: {"model": ErrorResponse, "description": "Error interno del servidor"},
    },
)


def get_documentos_service(
    settings: Settings = Depends(get_settings),
) -> DocumentosService:
    """Inyección de dependencias para el servicio de documentos.

    Args:
        settings: Configuración inyectada.

    Returns:
        DocumentosService: Instancia del servicio inicializado.
    """
    return DocumentosService(settings=settings)


@router.post(
    "",
    response_model=DocumentoUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Subir y procesar un archivo PDF",
    description="Recibe un archivo PDF multipart, extrae su texto por páginas, lo fragmenta y lo almacena vectorizado en ChromaDB.",
)
async def upload_documento(
    file: UploadFile = File(..., description="Archivo PDF a procesar"),
    service: DocumentosService = Depends(get_documentos_service),
    settings: Settings = Depends(get_settings),
) -> DocumentoUploadResponse:
    """Endpoint para cargar y procesar un archivo PDF.

    Args:
        file: Archivo subido en formato multipart.
        service: Servicio de documentos inyectado.
        settings: Configuración general inyectada.

    Returns:
        DocumentoUploadResponse: Metadatos del documento indexado.

    Raises:
        HTTPException 400: Si el archivo no es un PDF válido o excede el tamaño máximo.
        HTTPException 422: Si el archivo está vacío o sin texto legible.
        HTTPException 500: Si ocurre un error inesperado durante el procesamiento.
    """
    # 1. Validar nombre de archivo y extensión
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "codigo": "ARCHIVO_NO_VALIDO",
                "mensaje": "El archivo proporcionado debe tener extensión .pdf.",
            },
        )

    # 2. Leer contenido y verificar tamaño
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "codigo": "ARCHIVO_VACIO",
                "mensaje": "El archivo PDF subido no contiene datos.",
            },
        )

    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "codigo": "TAMANO_EXCEDIDO",
                "mensaje": f"El archivo excede el tamaño máximo permitido de {settings.max_upload_size_bytes} bytes.",
            },
        )

    # 3. Procesar, vectorizar e indexar
    try:
        resultado = service.process_and_store_pdf(filename=filename, content=content)
        return resultado
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "codigo": "PDF_ILEGIBLE",
                "mensaje": str(val_err),
            },
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "codigo": "ERROR_PROCESAMIENTO",
                "mensaje": f"Ocurrió un error inesperado al procesar el PDF: {str(exc)}",
            },
        )


@router.get(
    "",
    response_model=DocumentosListResponse,
    status_code=status.HTTP_200_OK,
    summary="Listar documentos indexados",
    description="Retorna la lista de documentos registrados con paginación configurable.",
)
async def list_documentos(
    limit: int = Query(10, ge=1, le=100, description="Cantidad máxima de documentos a retornar"),
    offset: int = Query(0, ge=0, description="Número de documentos a omitir (desplazamiento)"),
    service: DocumentosService = Depends(get_documentos_service),
) -> DocumentosListResponse:
    """Endpoint para listar documentos con paginación.

    Args:
        limit: Límite de elementos por página (entre 1 y 100).
        offset: Desplazamiento inicial.
        service: Servicio de documentos inyectado.

    Returns:
        DocumentosListResponse: Lista de documentos y total disponible.
    """
    return service.list_documentos(limit=limit, offset=offset)


@router.get(
    "/{documento_id}",
    response_model=DocumentoResponse,
    status_code=status.HTTP_200_OK,
    summary="Consultar un documento por ID",
    description="Obtiene los metadatos y conteo de páginas de un documento mediante su identificador UUID.",
)
async def get_documento(
    documento_id: str,
    service: DocumentosService = Depends(get_documentos_service),
) -> DocumentoResponse:
    """Endpoint para obtener los detalles de un documento indexado.

    Args:
        documento_id: Identificador único UUID del documento.
        service: Servicio de documentos inyectado.

    Returns:
        DocumentoResponse: Metadatos del documento.

    Raises:
        HTTPException 404: Si el documento no existe en el sistema.
    """
    documento = service.get_documento(documento_id=documento_id)
    if not documento:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "codigo": "DOCUMENTO_NO_ENCONTRADO",
                "mensaje": f"No se encontró ningún documento con el ID '{documento_id}'.",
            },
        )
    return documento


@router.delete(
    "/{documento_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Eliminar un documento indexado",
    description="Elimina el archivo, sus fragmentos vectoriales de ChromaDB y sus metadatos.",
)
async def delete_documento(
    documento_id: str,
    service: DocumentosService = Depends(get_documentos_service),
) -> None:
    """Endpoint para eliminar un documento por su ID.

    Args:
        documento_id: Identificador único UUID del documento a eliminar.
        service: Servicio de documentos inyectado.

    Raises:
        HTTPException 404: Si el documento no existe.
    """
    eliminado = service.delete_documento(documento_id=documento_id)
    if not eliminado:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "codigo": "DOCUMENTO_NO_ENCONTRADO",
                "mensaje": f"No se encontró ningún documento con el ID '{documento_id}' para eliminar.",
            },
        )
    return None
