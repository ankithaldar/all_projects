# syntax=docker/dockerfile:1
# Multi-stage uv build for the forecast-ml serving image.
# Runtime is python:3.14-slim, non-root UID 10001 only.
# Base digest: python:3.14-slim manifest-list
# sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f
# resolved 2026-11-08 via Docker Hub library/python tag 3.14-slim;
# bump deliberately, never by floating tag. uv is pinned to
# 0.13.7 (PyPI latest 2026-11-08; local `uv --version` 0.13.7).
FROM python:3.14-slim@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f AS builder
WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
RUN pip install --no-cache-dir 'uv==0.13.7'
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY conf ./conf
RUN uv sync --frozen --no-dev

FROM python:3.14-slim@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f AS runtime
WORKDIR /app
RUN groupadd --gid 10001 appgroup \
    && useradd --uid 10001 --gid 10001 --no-create-home \
       --shell /usr/sbin/nologin appuser
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src ./src
COPY --from=builder /app/conf ./conf
COPY pyproject.toml uv.lock README.md ./
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HOME=/tmp \
    FORECAST_REQUIRE_AUTH=1 \
    FORECAST_BIND_HOST=0.0.0.0
EXPOSE 8000
USER 10001
# Fail-closed entrypoint: validate the non-loopback
# bind (requires FORECAST_API_KEY or OIDC) before
# serving; a missing key aborts startup. The serving
# flags below preserve the frozen entrypoint command.
ENTRYPOINT ["sh", "-c", "python -c 'from forecast_ml_framework.serving.app import validate_bind_host; validate_bind_host(\"0.0.0.0\")' && exec uvicorn forecast_ml_framework.serving.app:app --host 0.0.0.0 --port 8000"]
