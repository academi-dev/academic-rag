"""Servicio de negocio para la gestión, almacenamiento y vectorización de documentos.

Administra el ciclo de vida de los documentos PDF: guardado en disco, extracción de texto,
generación de embeddings con sentence-transformers, persistencia en ChromaDB y registro
de metadatos para consultas y eliminaciones.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from app.config import Settings, get_settings
from app.documentos.processor import PDFProcessor
from app.documentos.schemas import (
    DocumentoResponse,
    DocumentoUploadResponse,
    DocumentosListResponse,
)
from app.vector_store import create_vector_collection


class DocumentosService:
    """Servicio que coordina la persistencia física y vectorial de documentos."""

    def __init__(self, settings: Optional[Settings] = None):
        """Inicializa el servicio con la configuración indicada o la global en caché.

        Args:
            settings: Instancia opcional de configuración.
        """
        self.settings = settings or get_settings()
        self.processor = PDFProcessor(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        # Inicialización de almacenamiento vectorial persistente (ChromaDB o respaldo local)
        self._init_chroma()

        # Registro de documentos para metadatos agregados
        self.registry_path = Path(self.settings.chroma_persist_dir) / "registry.json"
        self._ensure_registry_exists()

    def _init_chroma(self) -> None:
        """Inicializa la colección vectorial persistente y el directorio de subidas."""
        try:
            Path(self.settings.upload_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        self.collection = create_vector_collection(self.settings)

    def _ensure_registry_exists(self) -> None:
        """Verifica la existencia del archivo de registro de metadatos o lo inicializa."""
        if not self.registry_path.exists():
            self.registry_path.parent.mkdir(parents=True, exist_ok=True)
            self._save_registry({})

    def _load_registry(self) -> Dict[str, Dict[str, Any]]:
        """Lee el registro de metadatos de documentos desde disco.

        Returns:
            Dict[str, Dict[str, Any]]: Diccionario mapeando id de documento a sus metadatos.
        """
        try:
            if self.registry_path.exists():
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    def _save_registry(self, registry: Dict[str, Dict[str, Any]]) -> None:
        """Guarda el registro de metadatos de documentos en disco.

        Args:
            registry: Estructura de diccionario a persistir.
        """
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, ensure_ascii=False, indent=2)

    def process_and_store_pdf(
        self, filename: str, content: bytes
    ) -> DocumentoUploadResponse:
        """Guarda un archivo PDF, extrae su texto por páginas, lo fragmenta y lo almacena en ChromaDB.

        Args:
            filename: Nombre original del archivo proporcionado por el usuario.
            content: Contenido binario del archivo PDF.

        Returns:
            DocumentoUploadResponse: Resultado del procesamiento con totales y metadatos.

        Raises:
            ValueError: Si el archivo no es un PDF válido o falla la extracción de texto.
        """
        documento_id = str(uuid.uuid4())
        fecha_creacion = datetime.now(timezone.utc).isoformat()
        tamano_bytes = len(content)

        # 1. Guardar archivo en disco
        safe_name = Path(filename).name
        target_path = Path(self.settings.upload_dir) / f"{documento_id}_{safe_name}"
        with open(target_path, "wb") as f:
            f.write(content)

        try:
            # 2. Extraer texto por páginas
            pages_data = self.processor.extract_text_by_page(content)
            total_paginas = len(pages_data)

            # 3. Dividir en fragmentos conservando número de página y posición
            chunks = self.processor.process_pages_to_chunks(pages_data)
            total_fragmentos = len(chunks)

            if total_fragmentos == 0:
                raise ValueError("No se pudieron generar fragmentos con contenido de texto.")

            # 4. Preparar vectores e insertar en ChromaDB
            ids: List[str] = []
            documents: List[str] = []
            metadatas: List[Dict[str, Any]] = []

            for chunk in chunks:
                chunk_id = f"{documento_id}_chunk_{chunk['posicion']}"
                ids.append(chunk_id)
                documents.append(chunk["texto"])
                metadatas.append({
                    "documento_id": documento_id,
                    "nombre_archivo": safe_name,
                    "pagina": int(chunk["pagina"]),
                    "posicion": int(chunk["posicion"]),
                    "fecha_creacion": fecha_creacion,
                })

            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

            # 5. Registrar documento en el almacén de metadatos
            registry = self._load_registry()
            registry[documento_id] = {
                "id": documento_id,
                "nombre_archivo": safe_name,
                "ruta_archivo": str(target_path),
                "tamano_bytes": tamano_bytes,
                "total_paginas": total_paginas,
                "total_fragmentos": total_fragmentos,
                "fecha_creacion": fecha_creacion,
            }
            self._save_registry(registry)

            return DocumentoUploadResponse(
                id=documento_id,
                nombre_archivo=safe_name,
                mensaje="Documento procesado y vectorizado exitosamente.",
                total_paginas=total_paginas,
                total_fragmentos=total_fragmentos,
                fecha_creacion=fecha_creacion,
            )
        except Exception:
            # Si ocurre algún fallo, limpiar el archivo guardado
            if target_path.exists():
                try:
                    os.remove(target_path)
                except OSError:
                    pass
            raise

    def list_documentos(self, limit: int = 10, offset: int = 0) -> DocumentosListResponse:
        """Obtiene la lista paginada de documentos indexados.

        Args:
            limit: Número máximo de registros a retornar.
            offset: Desplazamiento inicial de la consulta.

        Returns:
            DocumentosListResponse: Lista paginada y total de documentos.
        """
        registry = self._load_registry()
        # Ordenar por fecha_creacion descendente (más recientes primero)
        all_docs = sorted(
            registry.values(),
            key=lambda d: d.get("fecha_creacion", ""),
            reverse=True,
        )

        total = len(all_docs)
        paginated = all_docs[offset : offset + limit]

        items = [
            DocumentoResponse(
                id=doc["id"],
                nombre_archivo=doc["nombre_archivo"],
                tamano_bytes=doc["tamano_bytes"],
                total_paginas=doc["total_paginas"],
                total_fragmentos=doc["total_fragmentos"],
                fecha_creacion=doc["fecha_creacion"],
            )
            for doc in paginated
        ]

        return DocumentosListResponse(
            items=items,
            total=total,
            limit=limit,
            offset=offset,
        )

    def get_documento(self, documento_id: str) -> Optional[DocumentoResponse]:
        """Recupera los metadatos de un documento por su identificador UUID.

        Args:
            documento_id: Identificador único del documento.

        Returns:
            Optional[DocumentoResponse]: Datos del documento o None si no existe.
        """
        registry = self._load_registry()
        doc = registry.get(documento_id)
        if not doc:
            return None

        return DocumentoResponse(
            id=doc["id"],
            nombre_archivo=doc["nombre_archivo"],
            tamano_bytes=doc["tamano_bytes"],
            total_paginas=doc["total_paginas"],
            total_fragmentos=doc["total_fragmentos"],
            fecha_creacion=doc["fecha_creacion"],
        )

    def delete_documento(self, documento_id: str) -> bool:
        """Elimina un documento, sus fragmentos vectoriales en ChromaDB y su archivo físico.

        Args:
            documento_id: Identificador único del documento a eliminar.

        Returns:
            bool: True si el documento existía y fue eliminado, False si no se encontró.
        """
        registry = self._load_registry()
        doc = registry.get(documento_id)
        if not doc:
            return False

        # 1. Eliminar fragmentos de ChromaDB
        try:
            self.collection.delete(where={"documento_id": documento_id})
        except Exception:
            pass

        # 2. Eliminar archivo físico
        file_path = doc.get("ruta_archivo")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass

        # 3. Remover del registro
        del registry[documento_id]
        self._save_registry(registry)
        return True
