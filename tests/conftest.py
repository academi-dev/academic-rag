"""Configuración global y fixtures de Pytest para Academic RAG.

Proporciona un entorno aislado de pruebas con directorios temporales, cliente
de pruebas FastAPI TestClient y generador de archivos PDF sintéticos válidos.
"""

from io import BytesIO
from pathlib import Path
from typing import Generator, List
import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app


def build_minimal_pdf(pages_text: List[str]) -> bytes:
    """Genera un archivo PDF válido en bytes con páginas y texto legibles por PyPDF2.

    Crea una estructura PDF 1.4 canónica con fuentes Helvetica estándar
    para que la extracción de texto sea exacta sin requerir dependencias externas.

    Args:
        pages_text: Lista de cadenas de texto, una por cada página deseada.

    Returns:
        bytes: Archivo PDF completo codificado en bytes.
    """
    objects: List[bytes] = []

    def add_object(content: bytes) -> int:
        objects.append(content)
        return len(objects)

    # 1: Catálogo raíz (se actualizará referencia a Pages)
    # 2: Árbol de páginas
    # 3: Fuente Helvetica compartida
    font_content = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    font_id = 3  # Será el objeto 3

    page_ids: List[int] = []
    content_ids: List[int] = []

    # Se reservan los objetos 1, 2, 3
    # Los streams de contenido y páginas comienzan en el objeto 4
    current_obj = 4
    for _ in pages_text:
        content_ids.append(current_obj)
        current_obj += 1
        page_ids.append(current_obj)
        current_obj += 1

    # Objeto 1: Catalog
    catalog_obj = b"<< /Type /Catalog /Pages 2 0 R >>"

    # Objeto 2: Pages
    kids_refs = b" ".join([f"{pid} 0 R".encode("ascii") for pid in page_ids])
    pages_obj = (
        b"<< /Type /Pages /Kids [ "
        + kids_refs
        + b" ] /Count "
        + str(len(pages_text)).encode("ascii")
        + b" >>"
    )

    objects_list: List[bytes] = [catalog_obj, pages_obj, font_content]

    # Construir contenidos y páginas
    for text, c_id, p_id in zip(pages_text, content_ids, page_ids):
        # Escapar paréntesis en texto PDF
        safe_text = text.replace("(", "\\(").replace(")", "\\)")
        stream_data = (
            f"BT\n/F1 12 Tf\n50 750 Td\n({safe_text}) Tj\nET\n".encode("utf-8")
        )
        stream_obj = (
            f"<< /Length {len(stream_data)} >>\nstream\n".encode("ascii")
            + stream_data
            + b"endstream"
        )
        objects_list.append(stream_obj)

        page_dict = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {c_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        ).encode("ascii")
        objects_list.append(page_dict)

    # Ensamblar archivo PDF con tabla xref
    buffer = BytesIO()
    buffer.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")

    offsets: List[int] = []
    for idx, obj in enumerate(objects_list, start=1):
        offsets.append(buffer.tell())
        buffer.write(f"{idx} 0 obj\n".encode("ascii"))
        buffer.write(obj)
        buffer.write(b"\nendobj\n")

    xref_pos = buffer.tell()
    buffer.write(f"xref\n0 {len(objects_list) + 1}\n".encode("ascii"))
    buffer.write(b"0000000000 65535 f \n")
    for off in offsets:
        buffer.write(f"{off:010d} 00000 n \n".encode("ascii"))

    trailer = (
        f"trailer\n<< /Size {len(objects_list) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode("ascii")
    buffer.write(trailer)

    return buffer.getvalue()


@pytest.fixture(scope="function")
def sample_pdf_bytes() -> bytes:
    """Fixture que retorna un PDF válido de prueba con 2 páginas académicas."""
    pages = [
        "Capitulo 1: Introduccion a la Programacion Orientada a Objetos. Los pilares son encapsulamiento, herencia y polimorfismo.",
        "Capitulo 2: Complejidad Algoritmica. La notacion Big O define el limite superior del tiempo de ejecucion en el peor caso.",
    ]
    return build_minimal_pdf(pages)


@pytest.fixture(scope="function")
def isolated_settings(tmp_path: Path) -> Settings:
    """Fixture que crea una configuración aislada en directorios temporales."""
    test_chroma = tmp_path / "chroma_test"
    test_uploads = tmp_path / "uploads_test"
    test_chroma.mkdir(parents=True, exist_ok=True)
    test_uploads.mkdir(parents=True, exist_ok=True)

    settings = Settings(
        app_name="Academic RAG Test",
        chroma_persist_dir=str(test_chroma),
        upload_dir=str(test_uploads),
        collection_name="test_collection",
        chunk_size=300,
        chunk_overlap=30,
        llm_provider="local",
        openai_api_key=None,
    )
    return settings


@pytest.fixture(scope="function")
def client(isolated_settings: Settings) -> Generator[TestClient, None, None]:
    """Fixture que provee un TestClient de FastAPI con dependencias aisladas."""
    app.dependency_overrides[get_settings] = lambda: isolated_settings
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
