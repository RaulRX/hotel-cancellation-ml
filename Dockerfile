# Stage 1: Install dependencies
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
  build-essential libgomp1 \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /environment

COPY ./requirements.txt ./configuration/

RUN pip install --prefix=./install -q --no-cache-dir -r ./configuration/requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim AS runner

RUN apt-get update && apt-get install -y --no-install-recommends \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*

ARG APP_LIB_DIR=/opt/hotel

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PYTHONPATH=/app:$APP_LIB_DIR/lib/python3.11/site-packages
ENV PATH="$APP_LIB_DIR/bin:$PATH"

EXPOSE 8000

WORKDIR /app

COPY --from=builder /environment/install $APP_LIB_DIR

COPY ./src ./src


ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
