"""Esquemas de datos Pydantic para el módulo de documentos.

Define las estructuras de respuesta y modelos de transferencia de datos (DTO)
para la subida, consulta y eliminación de documentos PDF.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class DocumentoResponse(BaseModel):
    """Modelo de respuesta con la información completa de un documento indexado.

    Atributos:
        id: Identificador único UUID del documento.
        nombre_archivo: Nombre original del archivo PDF subido.
        tamano_bytes: Tamaño del archivo en bytes.
        total_paginas: Cantidad total de páginas procesadas del documento.
        total_fragmentos: Cantidad de fragmentos de texto generados e indexados.
        fecha_creacion: Fecha de creación en formato ISO 8601 (UTC).
    """

    id: str = Field(..., description="Identificador único UUID del documento")
    nombre_archivo: str = Field(..., description="Nombre original del archivo PDF")
    tamano_bytes: int = Field(..., description="Tamaño del archivo en bytes")
    total_paginas: int = Field(..., description="Total de páginas procesadas")
    total_fragmentos: int = Field(..., description="Total de fragmentos vectorizados")
    fecha_creacion: str = Field(..., description="Fecha de indexación en UTC (ISO 8601)")


class DocumentoUploadResponse(BaseModel):
    """Modelo de respuesta retornado al subir y procesar exitosamente un PDF.

    Atributos:
        id: Identificador único UUID asignado al documento.
        nombre_archivo: Nombre del archivo procesado.
        mensaje: Mensaje informativo del estado de la operación.
        total_paginas: Total de páginas extraídas.
        total_fragmentos: Total de fragmentos indexados en la base vectorial.
        fecha_creacion: Fecha de creación en formato ISO 8601 (UTC).
    """

    id: str = Field(..., description="Identificador único UUID del documento")
    nombre_archivo: str = Field(..., description="Nombre del archivo procesado")
    mensaje: str = Field(..., description="Mensaje descriptivo del resultado")
    total_paginas: int = Field(..., description="Total de páginas extraídas")
    total_fragmentos: int = Field(..., description="Total de fragmentos indexados")
    fecha_creacion: str = Field(..., description="Fecha de creación en formato ISO 8601")


class DocumentosListResponse(BaseModel):
    """Modelo de respuesta para el listado paginado de documentos indexados.

    Atributos:
        items: Lista de documentos indexados.
        total: Conteo total de documentos registrados.
        limit: Límite de elementos por página solicitado.
        offset: Desplazamiento inicial aplicado en la consulta.
    """

    items: List[DocumentoResponse] = Field(..., description="Lista de documentos")
    total: int = Field(..., description="Total de documentos registrados")
    limit: int = Field(..., description="Límite de documentos por página")
    offset: int = Field(..., description="Desplazamiento inicial de paginación")


class ErrorDetalle(BaseModel):
    """Estructura interna del detalle de un error."""

    codigo: str = Field(..., description="Código normalizado de error")
    mensaje: str = Field(..., description="Descripción legible del error")


class ErrorResponse(BaseModel):
    """Estructura estándar de respuesta para errores de la API.

    Formato: {"error": {"codigo": "...", "mensaje": "..."}}
    """

    error: ErrorDetalle = Field(..., description="Detalle del error ocurrido")
