# Stage 1: Ejecuta todos los requerimientos para ejecutar la aplicación
FROM python:3.11-slim AS builder

# Instalar herramientas para compilar extensiones nativas de Python
RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /environment

COPY ./requirements.txt ./configuration/

RUN pip install --prefix=./dependencies -q --no-cache-dir -r ./configuration/requirements.txt
RUN python -m venv ./venv

FROM python:3.11-slim AS runner

WORKDIR /hotel

COPY --from=builder /hotel/dependencies ./dependencies
COPY ./src .

ENTRYPOINT [ "uvicorn" "main:app", "--host", "0.0.0.0", "--port", "8000"]