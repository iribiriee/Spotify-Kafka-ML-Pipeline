#!/bin/bash
# reset.sh — wipe all state and start a clean run from model v1.
# Use this before a demo so the version starts at v1 and predictions aren't
# layered on top of a previous run.
set -e

# Activate the virtual environment (so python has the project's packages).
source .venv/bin/activate

echo "[reset] stopping stack and wiping Kafka volume..."
docker compose down -v

echo "[reset] clearing model versions, pointer, and buffer..."
rm -rf models data/buffer

echo "[reset] starting MLflow first (so baseline training can log to it)..."
docker compose up -d mlflow
echo "[reset] waiting for MLflow to be ready..."
sleep 10

echo "[reset] training fresh v1 baseline (phase-1 only)..."
python retrainer/train_model.py

echo "[reset] bringing the full stack up..."
docker compose up --build