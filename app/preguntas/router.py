"""Router FastAPI para el endpoint de consultas académicas (RAG).

Expone:
- POST /api/v1/preguntar: Procesa una pregunta en lenguaje natural y retorna
  la respuesta sustentada con citas de páginas y documentos.
"""

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import Settings, get_settings
from app.documentos.schemas import ErrorResponse
from app.documentos.service import DocumentosService
from app.preguntas.schemas import PreguntaRequest, RespuestaResponse
from app.preguntas.service import PreguntasService

router = APIRouter(
    prefix="/api/v1",
    tags=["Preguntas"],
    responses={
        400: {"model": ErrorResponse, "description": "Solicitud inválida"},
        404: {"model": ErrorResponse, "description": "Documento filtrado no encontrado"},
        422: {"model": ErrorResponse, "description": "Error de validación"},
        500: {"model": ErrorResponse, "description": "Error interno del servidor"},
    },
)


def get_preguntas_service(
    settings: Settings = Depends(get_settings),
) -> PreguntasService:
    """Inyección de dependencias para el servicio de preguntas y respuestas.

    Args:
        settings: Configuración general de la aplicación.

    Returns:
        PreguntasService: Instancia del servicio inicializado.
    """
    return PreguntasService(settings=settings)


def get_documentos_service(
    settings: Settings = Depends(get_settings),
) -> DocumentosService:
    """Inyección de dependencias para el servicio de documentos.

    Args:
        settings: Configuración general.

    Returns:
        DocumentosService: Instancia de servicio para validaciones.
    """
    return DocumentosService(settings=settings)


@router.post(
    "/preguntar",
    response_model=RespuestaResponse,
    status_code=status.HTTP_200_OK,
    summary="Realizar una pregunta sobre los apuntes indexados",
    description="Recibe una pregunta en lenguaje natural, busca semánticamente en los fragmentos de PDF indexados en ChromaDB y devuelve una respuesta con citas precisas de documento y página.",
)
async def preguntar(
    request: PreguntaRequest,
    preguntas_service: PreguntasService = Depends(get_preguntas_service),
    documentos_service: DocumentosService = Depends(get_documentos_service),
) -> RespuestaResponse:
    """Endpoint para formular preguntas sobre los apuntes.

    Args:
        request: Cuerpo de la petición con la pregunta y filtros opcionales.
        preguntas_service: Servicio de preguntas inyectado.
        documentos_service: Servicio de documentos inyectado para validar existencia del filtro.

    Returns:
        RespuestaResponse: Respuesta generada con lista de citas y páginas de origen.

    Raises:
        HTTPException 400: Si la pregunta está vacía.
        HTTPException 404: Si se especifica un documento_id que no existe.
        HTTPException 500: Si ocurre un error en la búsqueda o generación.
    """
    pregunta_limpia = request.pregunta.strip()
    if not pregunta_limpia:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "codigo": "PREGUNTA_VACIA",
                "mensaje": "La pregunta no puede estar vacía o contener solo espacios en blanco.",
            },
        )

    # Validar que si se indicó documento_id, este exista
    if request.documento_id:
        doc = documentos_service.get_documento(request.documento_id)
        if not doc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "codigo": "DOCUMENTO_NO_ENCONTRADO",
                    "mensaje": f"El documento con ID '{request.documento_id}' no existe en el sistema.",
                },
            )

    try:
        respuesta = preguntas_service.responder(request=request)
        return respuesta
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "codigo": "ERROR_CONSULTA_RAG",
                "mensaje": f"Ocurrió un error al procesar la consulta en la base vectorial: {str(exc)}",
            },
        )
