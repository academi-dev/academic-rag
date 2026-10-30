"""Pruebas para los endpoints de documentos (/api/v1/documentos).

Valida la subida multipart de PDFs, validaciones de formato y tamaño, listado
paginado, consulta individual por UUID y eliminación completa de registros y vectores.
"""

from io import BytesIO
import uuid
from fastapi.testclient import TestClient


def test_subir_documento_exitoso(client: TestClient, sample_pdf_bytes: bytes) -> None:
    """Verifica que un archivo PDF válido se procese e indexe correctamente con código 201."""
    files = {
        "file": ("apuntes_poo.pdf", BytesIO(sample_pdf_bytes), "application/pdf")
    }

    response = client.post("/api/v1/documentos", files=files)

    assert response.status_code == 201
    data = response.json()
    assert "id" in data
    # Validar que el ID sea un UUID válido
    uuid_obj = uuid.UUID(data["id"])
    assert str(uuid_obj) == data["id"]
    assert data["nombre_archivo"] == "apuntes_poo.pdf"
    assert data["total_paginas"] == 2
    assert data["total_fragmentos"] >= 2
    assert "fecha_creacion" in data


def test_subir_documento_extension_invalida(client: TestClient) -> None:
    """Verifica que rechace archivos con extensión distinta de .pdf con código 400 y formato estándar."""
    files = {
        "file": ("notas.txt", BytesIO(b"Contenido en texto plano"), "text/plain")
    }

    response = client.post("/api/v1/documentos", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["codigo"] == "ARCHIVO_NO_VALIDO"
    assert "extensión .pdf" in data["error"]["mensaje"]


def test_subir_documento_vacio(client: TestClient) -> None:
    """Verifica que rechace archivos vacíos de 0 bytes con código 400."""
    files = {
        "file": ("vacio.pdf", BytesIO(b""), "application/pdf")
    }

    response = client.post("/api/v1/documentos", files=files)

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["codigo"] == "ARCHIVO_VACIO"


def test_listar_documentos_paginado(client: TestClient, sample_pdf_bytes: bytes) -> None:
    """Verifica el listado de documentos con parámetros de paginación."""
    # 1. Al inicio debe estar vacío
    res_inicial = client.get("/api/v1/documentos?limit=5&offset=0")
    assert res_inicial.status_code == 200
    data_inicial = res_inicial.json()
    assert data_inicial["total"] == 0
    assert data_inicial["items"] == []

    # 2. Subir un documento
    files = {
        "file": ("apuntes_algoritmos.pdf", BytesIO(sample_pdf_bytes), "application/pdf")
    }
    client.post("/api/v1/documentos", files=files)

    # 3. Consultar listado nuevamente
    res_post = client.get("/api/v1/documentos?limit=10&offset=0")
    assert res_post.status_code == 200
    data_post = res_post.json()
    assert data_post["total"] == 1
    assert len(data_post["items"]) == 1
    assert data_post["items"][0]["nombre_archivo"] == "apuntes_algoritmos.pdf"
    assert data_post["items"][0]["total_paginas"] == 2


def test_obtener_documento_por_id_exitoso(client: TestClient, sample_pdf_bytes: bytes) -> None:
    """Verifica la consulta de metadatos de un documento específico mediante su UUID."""
    files = {
        "file": ("apuntes_sistemas.pdf", BytesIO(sample_pdf_bytes), "application/pdf")
    }
    upload_res = client.post("/api/v1/documentos", files=files)
    doc_id = upload_res.json()["id"]

    get_res = client.get(f"/api/v1/documentos/{doc_id}")
    assert get_res.status_code == 200
    doc_data = get_res.json()
    assert doc_data["id"] == doc_id
    assert doc_data["nombre_archivo"] == "apuntes_sistemas.pdf"
    assert doc_data["total_paginas"] == 2


def test_obtener_documento_inexistente(client: TestClient) -> None:
    """Verifica respuesta 404 al consultar un UUID que no existe en el registro."""
    doc_id_falso = str(uuid.uuid4())
    response = client.get(f"/api/v1/documentos/{doc_id_falso}")

    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["codigo"] == "DOCUMENTO_NO_ENCONTRADO"


def test_eliminar_documento_exitoso(client: TestClient, sample_pdf_bytes: bytes) -> None:
    """Verifica la eliminación completa de un documento y la verificación de su ausencia."""
    files = {
        "file": ("apuntes_a_borrar.pdf", BytesIO(sample_pdf_bytes), "application/pdf")
    }
    upload_res = client.post("/api/v1/documentos", files=files)
    doc_id = upload_res.json()["id"]

    # Eliminar
    delete_res = client.delete(f"/api/v1/documentos/{doc_id}")
    assert delete_res.status_code == 204

    # Intentar obtenerlo nuevamente debe dar 404
    get_res = client.get(f"/api/v1/documentos/{doc_id}")
    assert get_res.status_code == 404


def test_eliminar_documento_inexistente(client: TestClient) -> None:
    """Verifica que intentar eliminar un documento no registrado devuelva 404."""
    doc_id_falso = str(uuid.uuid4())
    delete_res = client.delete(f"/api/v1/documentos/{doc_id_falso}")

    assert delete_res.status_code == 404
    data = delete_res.json()
    assert "error" in data
    assert data["error"]["codigo"] == "DOCUMENTO_NO_ENCONTRADO"
