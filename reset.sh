#!/bin/bash
# reset.sh — wipe all state and start a clean run from model v1.
# Use this before a demo so the version starts at v1 and predictions aren't
# layered on top of a previous run.
set -e

echo "[reset] stopping stack and wiping Kafka volume..."
docker compose down -v

echo "[reset] clearing model versions, pointer, and buffer..."
rm -rf models data/buffer

echo "[reset] training fresh v1 baseline (phase-1 only)..."
python retrainer/train_model.py

echo "[reset] bringing the stack up..."
docker compose up --build