FROM python:3.12-slim

WORKDIR /app

COPY retrainer/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY retrainer/ ./retrainer/

ENV PYTHONPATH=/app

# models/ (write new versions) and data/ (read buffer) are mounted at runtime.
CMD ["python", "retrainer/retrainer.py"]
