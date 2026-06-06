# Stage 1: Install dependencies
FROM python:3.11-slim AS builder

WORKDIR /environment

COPY ./requirements.txt ./configuration/

RUN pip install --prefix=./install -q --no-cache-dir -r ./configuration/requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim AS runner

ARG APP_LIB_DIR=/opt/hotel

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH=/app:$APP_LIB_DIR/lib/python3.11/site-packages
ENV PATH="$APP_LIB_DIR/bin:$PATH"

EXPOSE 8000

WORKDIR /app

COPY ./src ./src

COPY --from=builder /environment/install $APP_LIB_DIR

RUN apt-get update && apt-get install -y --no-install-recommends \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["fastapi", "run", "src/api/main.py", "--host", "0.0.0.0", "--port", "8000"]
