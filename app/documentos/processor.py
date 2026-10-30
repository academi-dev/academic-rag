"""Procesador de archivos PDF y fragmentación de texto para RAG.

Se encarga de extraer texto de documentos PDF página por página preservando el número
de página y de dividir el texto en fragmentos (chunks) con solapamiento (overlap) para
su posterior indexación vectorial en ChromaDB.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Dict, List, Union
import PyPDF2

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_SPLITTER_AVAILABLE = True
except ImportError:
    LANGCHAIN_SPLITTER_AVAILABLE = False


class PDFProcessor:
    """Clase utilitaria para procesar archivos PDF y generar fragmentos de texto."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """Inicializa el procesador de documentos.

        Args:
            chunk_size: Cantidad máxima de caracteres por fragmento.
            chunk_overlap: Cantidad de caracteres compartidos entre fragmentos consecutivos.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if LANGCHAIN_SPLITTER_AVAILABLE:
            self._splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""],
            )
        else:
            self._splitter = None

    def extract_text_by_page(
        self, file_source: Union[str, Path, BinaryIO, bytes]
    ) -> List[Dict[str, Any]]:
        """Extrae el contenido textual de un archivo PDF manteniendo el rastreo de páginas.

        Args:
            file_source: Ruta al archivo PDF, stream binario o arreglo de bytes.

        Returns:
            List[Dict[str, Any]]: Lista de diccionarios con la estructura:
                [{"pagina": int, "texto": str}, ...] (páginas 1-indexadas).

        Raises:
            ValueError: Si el archivo no es un PDF válido o no contiene texto legible.
        """
        stream: BinaryIO
        should_close = False

        if isinstance(file_source, (str, Path)):
            stream = open(file_source, "rb")
            should_close = True
        elif isinstance(file_source, bytes):
            stream = BytesIO(file_source)
        else:
            stream = file_source

        try:
            reader = PyPDF2.PdfReader(stream)
            total_pages = len(reader.pages)
            pages_data: List[Dict[str, Any]] = []

            for page_index in range(total_pages):
                page = reader.pages[page_index]
                text = page.extract_text() or ""
                clean_text = text.strip()

                if clean_text:
                    pages_data.append({
                        "pagina": page_index + 1,  # 1-indexado para citación humana
                        "texto": clean_text,
                    })

            if not pages_data:
                raise ValueError("El archivo PDF no contiene texto legible o está vacío.")

            return pages_data
        except Exception as e:
            if isinstance(e, ValueError):
                raise e
            raise ValueError(f"Error al analizar el archivo PDF: {str(e)}") from e
        finally:
            if should_close:
                stream.close()

    def _split_text_fallback(self, text: str) -> List[str]:
        """Algoritmo de respaldo para división de texto cuando LangChain no está disponible.

        Args:
            text: Cadena de texto a fragmentar.

        Returns:
            List[str]: Lista de fragmentos respetando chunk_size y chunk_overlap.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks: List[str] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]

            # Intentar cortar por el último espacio o punto si no estamos al final
            if end < text_len:
                last_space = chunk.rfind(" ")
                last_period = chunk.rfind(". ")
                break_point = max(last_space, last_period)
                if break_point > int(self.chunk_size * 0.5):
                    chunk = chunk[: break_point + 1]
                    end = start + len(chunk)

            chunk_clean = chunk.strip()
            if chunk_clean:
                chunks.append(chunk_clean)

            step = len(chunk) - self.chunk_overlap
            start += max(1, step)

        return chunks

    def chunk_text(self, text: str) -> List[str]:
        """Divide un texto en fragmentos utilizando LangChain o el algoritmo de respaldo.

        Args:
            text: Texto completo a fragmentar.

        Returns:
            List[str]: Lista de fragmentos generados.
        """
        if self._splitter is not None:
            return self._splitter.split_text(text)
        return self._split_text_fallback(text)

    def process_pages_to_chunks(
        self, pages_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Transforma una lista de páginas en una lista de fragmentos con metadatos asociados.

        Args:
            pages_data: Lista de páginas obtenida con `extract_text_by_page`.

        Returns:
            List[Dict[str, Any]]: Fragmentos resultantes con la siguiente estructura:
                [
                    {
                        "pagina": int,
                        "posicion": int,
                        "texto": str
                    },
                    ...
                ]
        """
        chunks_result: List[Dict[str, Any]] = []
        global_position = 0

        for page_item in pages_data:
            page_number = page_item["pagina"]
            page_text = page_item["texto"]

            text_chunks = self.chunk_text(page_text)
            for chunk_content in text_chunks:
                clean_chunk = chunk_content.strip()
                if not clean_chunk:
                    continue

                chunks_result.append({
                    "pagina": page_number,
                    "posicion": global_position,
                    "texto": clean_chunk,
                })
                global_position += 1

        return chunks_result
