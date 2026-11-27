#!/bin/sh
# Register the Feast feature view against Redis before starting (idempotent), unless disabled.
set -e
if [ "${FEAST_APPLY:-1}" = "1" ] && [ "${ONLINE_STORE:-feast}" = "feast" ]; then
  cd /app/feature_repo
  mkdir -p data
  feast -f feature_store.redis.yaml apply
  cd /app
fi
exec "$@"
