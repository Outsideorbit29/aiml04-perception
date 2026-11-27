# F7 analyst dashboard. Separate, lighter image than services/Dockerfile (no xgboost/feast/kafka).
# Build from the repo root:  docker build -f services/dashboard/Dockerfile -t aiml01-dashboard:dev .
FROM python:3.10-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false STREAMLIT_SERVER_HEADLESS=true
WORKDIR /app
COPY requirements-dashboard.txt .
RUN pip install --no-cache-dir -r requirements-dashboard.txt
COPY fraud/ fraud/
COPY configs/ configs/
COPY services/__init__.py services/__init__.py
COPY services/dashboard/ services/dashboard/
EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --retries=10 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "services/dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
