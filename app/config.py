"""Módulo de configuración para Academic RAG.

Carga y valida las variables de entorno para el servicio RAG, incluyendo rutas de
persistencia de ChromaDB, modelos de embeddings, configuración de fragmentación (chunking)
y proveedores de generación de respuestas.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict
except ImportError:
    # Fallback compatible si pydantic-settings no está instalado
    from pydantic import BaseModel as BaseSettings  # type: ignore

    class SettingsConfigDict:  # type: ignore
        def __init__(self, **kwargs):
            pass


class Settings(BaseSettings):
    """Configuración global de la aplicación obtenida desde variables de entorno.

    Atributos:
        app_name: Nombre legible de la aplicación.
        app_version: Versión semántica del servicio.
        chroma_persist_dir: Directorio para persistir la base de datos vectorial ChromaDB.
        embedding_model: Nombre del modelo HuggingFace/Sentence-Transformers para embeddings.
        chunk_size: Tamaño de cada fragmento de texto (en caracteres).
        chunk_overlap: Solapamiento entre fragmentos continuos (en caracteres).
        llm_provider: Proveedor de generación de respuestas ('local' o 'openai').
        openai_api_key: Clave de API de OpenAI (opcional si se usa 'openai').
        upload_dir: Directorio temporal para almacenar archivos PDF subidos.
        collection_name: Nombre de la colección en ChromaDB para almacenar fragmentos.
        host: Dirección IP o interfaz de enlace del servidor.
        port: Puerto TCP en el que escucha el servidor FastAPI.
        debug: Modo de depuración activo/inactivo.
        max_upload_size_bytes: Tamaño máximo permitido para archivos PDF (50 MB por defecto).
    """

    app_name: str = "Academic RAG API"
    app_version: str = "1.0.0"

    # Almacenamiento vectorial
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    collection_name: str = os.getenv("COLLECTION_NAME", "academic_documents")

    # Fragmentación
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # Proveedor LLM
    llm_provider: str = os.getenv("LLM_PROVIDER", "local")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", None)

    # Subida de archivos y servidor
    upload_dir: str = os.getenv("UPLOAD_DIR", "./uploads")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() in ("true", "1", "yes")
    max_upload_size_bytes: int = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(50 * 1024 * 1024)))

    # Soporte para Pydantic v2 SettingsConfigDict
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    ) if hasattr(BaseSettings, "model_config") else None


@lru_cache()
def get_settings() -> Settings:
    """Retorna una instancia única en caché de la configuración del sistema.

    Returns:
        Settings: Configuración inicializada y validada.
    """
    settings = Settings()
    # Asegurar existencia de directorios requeridos cuando el entorno tiene permisos de escritura
    try:
        Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    return settings
