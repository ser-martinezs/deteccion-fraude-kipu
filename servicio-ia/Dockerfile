# Contexto de build = RAÍZ del repositorio (necesita datos/preparar.py, que comparte con el entrenamiento):
#   docker build -f servicio-ia/Dockerfile -t servicio-ia .
# Antes hay que generar el modelo:  python datos/preparar.py && python servicio-ia/entrenar.py
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl solo para el HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY servicio-ia/requirements.txt .
RUN pip install -r requirements.txt

# Código, tratamiento compartido con el entrenamiento y modelo exportado (sin datos crudos ni secretos)
COPY datos/preparar.py ./preparar.py
COPY servicio-ia/main.py .
COPY servicio-ia/model/ ./model/

# No corre como root
RUN useradd --system --uid 10001 --no-create-home app
USER app

# Puerto interno: solo el gateway publica 8080 hacia afuera
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
