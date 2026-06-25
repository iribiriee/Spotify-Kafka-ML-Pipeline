#!/bin/bash
# =============================================================================
# create_topics.sh — creates the 3 pipeline topics, then lists them.
# Run by the init-topics service once Kafka is healthy. Idempotent.
#
# Design: 1 partition per topic so every track keeps strict global order
# (drift happens at known row positions — multiple partitions would scramble
# that). Replication factor 1 because this is a single-broker dev cluster.
# =============================================================================
set -euo pipefail

BOOTSTRAP="${KAFKA_BOOTSTRAP_INTERNAL:-kafka:29092}"

create() {
  local topic="$1"
  echo "Creating topic: ${topic}"
  kafka-topics --create \
    --if-not-exists \
    --topic "${topic}" \
    --bootstrap-server "${BOOTSTRAP}" \
    --partitions 1 \
    --replication-factor 1
}

create "${TOPIC_RAW_TRACKS:-raw-tracks}"
create "${TOPIC_PREDICTIONS:-predictions}"
create "${TOPIC_DRIFT_ALERTS:-drift-alerts}"

echo ""
echo "All topics now present:"
kafka-topics --list --bootstrap-server "${BOOTSTRAP}"
echo "init-topics done."