FROM python:3.12-slim

WORKDIR /app

COPY dashboard/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY dashboard/ ./dashboard/

ENV PYTHONPATH=/app
EXPOSE 5000

# data/ is mounted at runtime (for the phase-1 reference CSV).
CMD ["python", "dashboard/backend.py"]
