# Academic RAG

> Servicio que responde preguntas sobre tus apuntes en PDF citando el documento y la página de origen.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.24+-FC521F)](https://www.trychroma.com)
[![Sentence--Transformers](https://img.shields.io/badge/Embeddings-all--MiniLM--L6--v2-blue)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)](https://www.docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Descripción

**Academic RAG** es el microservicio de recuperación y generación aumentada (Retrieval-Augmented Generation) del ecosistema **Academic**. Permite a los estudiantes universitarios subir sus apuntes y lecturas en formato PDF, fragmentar e indexar vectorialmente el contenido en **ChromaDB**, y realizar consultas en lenguaje natural recibiendo respuestas fundamentadas con citas verificables (indicando archivo de procedencia y número de página exacto).

---

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.11+
- **Framework Web:** FastAPI (ASGI de alto rendimiento con validación mediante Pydantic v2)
- **Base Vectorial:** ChromaDB (almacenamiento persistente en volumen)
- **Modelo de Embeddings:** `all-MiniLM-L6-v2` vía `sentence-transformers` (ejecución local, sin costo de API)
- **Extracción y Fragmentación:** PyPDF2 + LangChain (`RecursiveCharacterTextSplitter`)
- **Proveedor LLM:** Modo local estructurado (predeterminado y gratuito) con soporte opcional para OpenAI (`gpt-3.5-turbo` / `gpt-4o-mini`)
- **Contenedores:** Docker y Docker Compose
- **Pruebas:** Pytest y HTTPX (TestClient)

---

## 🚀 Cómo Correrlo

### Opción 1: Con Docker Compose (Recomendada)

1. Clonar el repositorio y copiar las variables de entorno:
   ```bash
   cp .env.example .env
   ```
2. Levantar el servicio con sus volúmenes persistentes (`chroma_db` y `uploads`):
   ```bash
   docker-compose up --build
   ```
3. El servicio estará disponible en:
   - **API Base:** [http://localhost:8000](http://localhost:8000)
   - **Documentación Interactiva (Swagger UI):** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **ReDoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Opción 2: Entorno Local (Virtualenv)

1. Crear y activar un entorno virtual:
   ```bash
   python -m venv .venv
   # En Windows PowerShell:
   .venv\Scripts\Activate.ps1
   # En Linux/macOS:
   source .venv/bin/activate
   ```
2. Instalar dependencias:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. Iniciar el servidor de desarrollo:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

---

## 🧪 Ejecución de Pruebas

Para ejecutar la suite completa de pruebas unitarias y de integración:

```bash
pytest -v
```

Las pruebas validan de forma aislada:
- Subida y validación de archivos PDF (tipos MIME, extensiones y archivos vacíos).
- Paginación y consulta individual por identificador UUID.
- Eliminación de documentos y sus vectores asociados.
- Respuestas de RAG y formato estricto de citas (`pagina`, `documento`, `texto`).
- Formato uniforme de errores: `{"error": {"codigo": "...", "mensaje": "..."}}`.

---

## 📖 Endpoints de la API

Todas las rutas principales están versionadas bajo `/api/v1/`.

### 1. Documentos

#### `POST /api/v1/documentos`
Sube y procesa un archivo PDF multipart. Extrae el texto página por página, lo fragmenta y lo almacena vectorizado en ChromaDB.

- **Content-Type:** `multipart/form-data`
- **Body:** `file: [archivo.pdf]`
- **Respuesta (201 Created):**
  ```json
  {
    "id": "e4a2c3d1-9f87-4b21-8b3a-5f6e7d8c9a0b",
    "nombre_archivo": "apuntes_sistemas.pdf",
    "mensaje": "Documento procesado y vectorizado exitosamente.",
    "total_paginas": 12,
    "total_fragmentos": 45,
    "fecha_creacion": "2026-10-28T15:00:00.000000+00:00"
  }
  ```

#### `GET /api/v1/documentos`
Lista los documentos indexados con paginación (`limit`, `offset`).

- **Query Params:** `limit=10&offset=0`
- **Respuesta (200 OK):**
  ```json
  {
    "items": [
      {
        "id": "e4a2c3d1-9f87-4b21-8b3a-5f6e7d8c9a0b",
        "nombre_archivo": "apuntes_sistemas.pdf",
        "tamano_bytes": 1048576,
        "total_paginas": 12,
        "total_fragmentos": 45,
        "fecha_creacion": "2026-10-28T15:00:00.000000+00:00"
      }
    ],
    "total": 1,
    "limit": 10,
    "offset": 0
  }
  ```

#### `GET /api/v1/documentos/{id}`
Obtiene el detalle de un documento indexado por su UUID.

#### `DELETE /api/v1/documentos/{id}`
Elimina el PDF del disco, purga sus vectores en ChromaDB y lo borra del registro. Retorna `204 No Content`.

---

### 2. Consultas y Preguntas (RAG)

#### `POST /api/v1/preguntar`
Realiza una consulta en lenguaje natural sobre los apuntes indexados.

- **Content-Type:** `application/json`
- **Body:**
  ```json
  {
    "pregunta": "¿Cuáles son los pilares fundamentales de la Programación Orientada a Objetos?",
    "documento_id": null,
    "top_k": 3
  }
  ```
- **Respuesta (200 OK):**
  ```json
  {
    "pregunta": "¿Cuáles son los pilares fundamentales de la Programación Orientada a Objetos?",
    "respuesta": "Basado en los apuntes académicos indexados, se encontró la siguiente información relevante...\n\n[1] apuntes_sistemas.pdf (Página 1):\n\"Los cuatro pilares fundamentales de la POO son: Encapsulamiento, Abstracción, Herencia y Polimorfismo...\"",
    "citas": [
      {
        "pagina": 1,
        "texto": "Los cuatro pilares fundamentales de la POO son: Encapsulamiento, Abstracción, Herencia y Polimorfismo...",
        "documento": "apuntes_sistemas.pdf",
        "documento_id": "e4a2c3d1-9f87-4b21-8b3a-5f6e7d8c9a0b",
        "posicion": 0,
        "relevancia": 0.895
      }
    ],
    "total_citas": 1,
    "proveedor_llm": "local"
  }
  ```

---

### 3. Salud del Servicio

#### `GET /api/v1/health`
Retorna el estado de conexión con ChromaDB y la configuración activa:
```json
{
  "status": "healthy",
  "chromadb": "connected",
  "llm_provider": "local",
  "embedding_model": "all-MiniLM-L6-v2",
  "version": "1.0.0"
}
```

---

## 🛡️ Formato Estándar de Errores

Todos los errores retornados por la API siguen estrictamente la especificación del sistema Academic:

```json
{
  "error": {
    "codigo": "DOCUMENTO_NO_ENCONTRADO",
    "mensaje": "No se encontró ningún documento con el ID '...'."
  }
}
```

Códigos HTTP utilizados: `200`, `201`, `204`, `400`, `401`, `403`, `404`, `409`, `422`, `500`.

---

## 👥 Equipo de Desarrollo

- **Jean Mena** — [@hilton45](https://github.com/hilton45)
- **Eyfer Mena** — [@eyfer3174-commits](https://github.com/eyfer3174-commits)

**Organización:** [academic-dev](https://github.com/academic-dev)

---

## 📄 Licencia

Este proyecto está bajo la Licencia **MIT**. Consulta el archivo [LICENSE](LICENSE) para más detalles.
