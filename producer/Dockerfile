# Build from the repo root context (see docker-compose build.context: .).
# That lets us copy both shared/ and this service's folder into the image.
FROM python:3.12-slim

WORKDIR /app

# Install deps first (this layer caches unless requirements change).
COPY producer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared code and the service code into the image.
COPY shared/ ./shared/
COPY producer/ ./producer/

# So `from shared.schemas import ...` resolves (repo root is /app).
ENV PYTHONPATH=/app

CMD ["python", "producer/producer.py"]
