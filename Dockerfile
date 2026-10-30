# Dockerfile para Academic RAG (FastAPI + ChromaDB + SentenceTransformers)
FROM python:3.11-slim

# Evitar generación de archivos .pyc y forzar salida sin buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Instalar dependencias del sistema mínimas necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Instalar dependencias de Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Crear directorios para datos persistentes
RUN mkdir -p /app/chroma_db /app/uploads

# Copiar código fuente de la aplicación
COPY app/ ./app/
COPY README.md .

# Exponer el puerto configurado
EXPOSE 8000

# Healthcheck para Docker
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Comando de inicio del servidor ASGI
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
