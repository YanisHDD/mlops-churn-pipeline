FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ configs/
COPY src/ src/
COPY service/ service/
COPY artifacts/model.joblib artifacts/model.joblib

ENV MODEL_PATH=/app/artifacts/model.joblib
EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
