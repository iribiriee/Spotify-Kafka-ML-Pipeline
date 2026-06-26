FROM python:3.12-slim

WORKDIR /app

COPY consumer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY consumer/ ./consumer/

ENV PYTHONPATH=/app

# Runs inference + drift as threads. models/ and data/ are mounted at runtime
# (see docker-compose volumes), not copied here — they change and are shared.
CMD ["python", "consumer/main.py"]
