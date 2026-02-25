FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app
WORKDIR /app
COPY requirements.txt requirements-core.txt ./
RUN pip install -r requirements.txt \
 && useradd --create-home --uid 1000 app

# Test image: dev tools on top, runs the suite and the linters offline.
FROM base AS test
COPY requirements-dev.txt ./
RUN pip install -r requirements-dev.txt
COPY . .
USER app
CMD ["sh", "-c", "python -m pytest -p no:cacheprovider && ruff check --no-cache . && ruff format --no-cache --check ."]

# Runtime image for the API and the dashboard. Code and sample data stay
# owned by root, so the app user can read but not modify them.
FROM base AS runtime
COPY . .
RUN mkdir /data && chown app /data
USER app
EXPOSE 8000 8501
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
