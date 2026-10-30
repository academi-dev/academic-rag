"""Pruebas para el endpoint de preguntas académicas (/api/v1/preguntar).

Valida la recuperación semántica de fragmentos en ChromaDB, la generación de
respuestas con citas exactas (página y documento), el filtrado por documento específico
y el manejo de errores con códigos normalizados.
"""

from io import BytesIO
import uuid
from fastapi.testclient import TestClient


def test_preguntar_sin_documentos_indexados(client: TestClient) -> None:
    """Verifica que una consulta cuando la base está vacía retorne 200 con citas vacías y aviso claro."""
    payload = {
        "pregunta": "¿Cuáles son los pilares de la programación orientada a objetos?",
        "top_k": 3,
    }

    response = client.post("/api/v1/preguntar", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["pregunta"] == payload["pregunta"]
    assert data["citas"] == []
    assert data["total_citas"] == 0
    assert "No se encontró información relevante" in data["respuesta"]


def test_preguntar_con_documento_indexado_retorna_citas(
    client: TestClient, sample_pdf_bytes: bytes
) -> None:
    """Verifica que al consultar apuntes indexados se devuelvan citas con número de página y texto."""
    # 1. Indexar PDF con contenido de POO y Big O
    files = {
        "file": ("apuntes_parcial_1.pdf", BytesIO(sample_pdf_bytes), "application/pdf")
    }
    upload_res = client.post("/api/v1/documentos", files=files)
    assert upload_res.status_code == 201

    # 2. Realizar pregunta sobre el contenido de la página 1
    payload = {
        "pregunta": "¿Qué es la programación orientada a objetos y cuáles son sus pilares?",
        "top_k": 2,
    }

    response = client.post("/api/v1/preguntar", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["total_citas"] > 0
    assert len(data["citas"]) > 0

    # Verificar estructura de cada cita
    primera_cita = data["citas"][0]
    assert "pagina" in primera_cita
    assert "texto" in primera_cita
    assert "documento" in primera_cita
    assert primera_cita["documento"] == "apuntes_parcial_1.pdf"
    assert isinstance(primera_cita["pagina"], int)

    # Validar que la respuesta integre las citas
    assert "apuntes_parcial_1.pdf" in data["respuesta"]


def test_preguntar_filtrando_por_documento_id(
    client: TestClient, sample_pdf_bytes: bytes
) -> None:
    """Verifica que al filtrar por un documento_id específico, solo se busquen fragmentos de ese apunte."""
    # Indexar dos documentos distintos
    files1 = {
        "file": ("apuntes_a.pdf", BytesIO(sample_pdf_bytes), "application/pdf")
    }
    res1 = client.post("/api/v1/documentos", files=files1)
    doc1_id = res1.json()["id"]

    files2 = {
        "file": ("apuntes_b.pdf", BytesIO(sample_pdf_bytes), "application/pdf")
    }
    res2 = client.post("/api/v1/documentos", files=files2)
    doc2_id = res2.json()["id"]

    # Consultar filtrando explícitamente por doc1_id
    payload = {
        "pregunta": "Explica la notacion Big O y complejidad",
        "documento_id": doc1_id,
        "top_k": 3,
    }

    response = client.post("/api/v1/preguntar", json=payload)
    assert response.status_code == 200
    data = response.json()

    # Todas las citas retornadas deben pertenecer a doc1_id
    for cita in data["citas"]:
        assert cita["documento_id"] == doc1_id
        assert cita["documento"] == "apuntes_a.pdf"


def test_preguntar_con_documento_id_inexistente(client: TestClient) -> None:
    """Verifica respuesta 404 si se intenta filtrar por un documento que no existe."""
    doc_id_falso = str(uuid.uuid4())
    payload = {
        "pregunta": "¿Qué es la herencia?",
        "documento_id": doc_id_falso,
    }

    response = client.post("/api/v1/preguntar", json=payload)

    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["codigo"] == "DOCUMENTO_NO_ENCONTRADO"


def test_preguntar_pregunta_vacia_retorna_400(client: TestClient) -> None:
    """Verifica que una pregunta vacía o de solo espacios retorne código 400 y formato estándar."""
    payload = {
        "pregunta": "     ",
        "top_k": 3,
    }

    response = client.post("/api/v1/preguntar", json=payload)

    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert data["error"]["codigo"] == "PREGUNTA_VACIA"
