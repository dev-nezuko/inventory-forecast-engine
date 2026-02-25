FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt requirements-core.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
