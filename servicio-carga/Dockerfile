# servicio-carga · se ejecuta UNA vez y termina (compose: restart "no").
# Contexto de build = RAÍZ del repositorio (necesita datos/tratado, que genera el pipeline con datos/preparar.py):
#   python datos/preparar.py --entrada datos/crudo --salida datos/tratado
#   docker build -f servicio-carga/Dockerfile -t servicio-carga .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY servicio-carga/requirements.txt .
RUN pip install -r requirements.txt && useradd --system --uid 10001 --no-create-home app

COPY servicio-carga/schema.sql servicio-carga/carga.py ./
# Solo las 4 tablas tratadas (nunca los CSV crudos)
COPY datos/tratado/usuarios.csv datos/tratado/dispositivos.csv datos/tratado/lista_negra.csv datos/tratado/transacciones.csv ./datos/

USER app
CMD ["python", "carga.py"]
