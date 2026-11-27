# One image for the scorer, consumer and producer (command chosen in docker-compose.yml).
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY fraud/ fraud/
COPY services/ services/
COPY configs/ configs/
COPY feature_repo/ feature_repo/
COPY services/entrypoint.sh /entrypoint.sh
ENTRYPOINT ["/bin/sh", "/entrypoint.sh"]
