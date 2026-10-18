# servicio-dominio · API REST de negocio. Es el único servicio que lee y escribe la base.
# Contexto de build = RAÍZ del repositorio (igual que servicio-ia):
#   docker build -f servicio-dominio/Dockerfile -t servicio-dominio .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

# curl solo para el HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

COPY servicio-dominio/requirements.txt .
RUN pip install -r requirements.txt && useradd --system --uid 10001 --no-create-home app
COPY servicio-dominio/main.py .

USER app
# Puerto interno: solo el gateway publica 8080 hacia afuera
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
