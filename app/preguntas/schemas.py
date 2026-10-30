"""Esquemas de datos Pydantic para el módulo de preguntas y respuestas.

Define las estructuras para la recepción de preguntas de usuarios y la devolución
de respuestas estructuradas con citas de documentos y páginas.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class Cita(BaseModel):
    """Estructura de una cita bibliográfica recuperada de un apunte en PDF.

    Atributos:
        pagina: Número de página física en el documento de origen (1-indexado).
        texto: Fragmento de texto extraído que fundamenta la respuesta.
        documento: Nombre original del archivo PDF de procedencia.
        documento_id: Identificador único UUID del documento.
        posicion: Índice ordinal del fragmento en el documento.
        relevancia: Puntuación de similitud o distancia vectorial (si está disponible).
    """

    pagina: int = Field(..., description="Número de página del documento donde se ubica el texto")
    texto: str = Field(..., description="Fragmento de texto relevante citado")
    documento: str = Field(..., description="Nombre del archivo PDF de origen")
    documento_id: Optional[str] = Field(None, description="Identificador UUID del documento")
    posicion: Optional[int] = Field(None, description="Posición relativa del fragmento")
    relevancia: Optional[float] = Field(None, description="Puntuación o distancia de similitud")


class PreguntaRequest(BaseModel):
    """Solicitud de consulta o pregunta enviada por el usuario.

    Atributos:
        pregunta: Texto de la pregunta en lenguaje natural.
        documento_id: Identificador UUID opcional para filtrar la búsqueda a un solo apunte.
        top_k: Cantidad de fragmentos relevantes a recuperar (por defecto 3).
    """

    pregunta: str = Field(
        ...,
        min_length=1,
        description="Pregunta en lenguaje natural sobre los apuntes académicos",
        example="¿Cuáles son los principios fundamentales del diseño orientado a objetos?",
    )
    documento_id: Optional[str] = Field(
        None,
        description="ID del documento específico a consultar. Si no se indica, busca en todos.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Cantidad máxima de fragmentos relevantes a recuperar",
    )


class RespuestaResponse(BaseModel):
    """Respuesta generada para la pregunta del usuario con citas justificativas.

    Atributos:
        pregunta: Pregunta original formulada.
        respuesta: Texto explicativo de la respuesta generada.
        citas: Lista de fragmentos y páginas de donde se obtuvo la información.
        total_citas: Conteo total de citas incluidas.
        proveedor_llm: Nombre del proveedor o método utilizado para componer la respuesta.
    """

    pregunta: str = Field(..., description="Pregunta formulada por el usuario")
    respuesta: str = Field(..., description="Respuesta construida con base en los apuntes")
    citas: List[Cita] = Field(..., description="Lista de citas que sustentan la respuesta")
    total_citas: int = Field(..., description="Cantidad total de citas encontradas")
    proveedor_llm: str = Field(..., description="Proveedor generador ('local' o 'openai')")
