#!/usr/bin/env bash
# F8.4: record the material for the captioned demo video (laptop, Docker Desktop, Git Bash).
#   bash scripts/record_demo.sh            # ~4 min of stack time, then `docker compose down`
#   python scripts/make_demo_video.py      # assemble media/aiml01_demo.mp4 (CPU, ~1 min)
# Starts the compose stack on a fresh case store, replays 6,000 test transactions at 50 tx/s with
# amounts x3 after 2,500 (drift demo), and screenshots the live dashboard / scorer from a headless
# Chrome container on the compose network (laptop browsers can't reach Docker ports; see
# scripts/browser_relay.py). Output: media/demo_capture/ (gitignored).
set -euo pipefail
export MSYS_NO_PATHCONV=1
OUT=media/demo_capture
NET=project_ai_aiml01_default
CHROME=zenika/alpine-chrome:with-puppeteer
REPO_WIN="$(pwd -W 2>/dev/null || pwd)"

rm -rf "$OUT" && mkdir -p "$OUT" serving_bundle/old
stamp=$(date +%Y%m%d_%H%M%S)
for f in serving_bundle/data/cases.db* serving_bundle/data/consumer_stats.json \
         serving_bundle/data/monitor_status.json; do
  [ -e "$f" ] && mv "$f" "serving_bundle/old/$(basename "$f").$stamp"
done

docker compose up -d --build kafka redis scorer consumer monitor dashboard
echo "waiting for the consumer warm-up..."
for _ in $(seq 120); do
  docker compose logs consumer 2>/dev/null | grep -q "consumer ready" && break
  sleep 2
done
docker compose ps --format 'table {{.Service}}\t{{.Status}}\t{{.Ports}}' > "$OUT/ps.txt"

docker run --rm --name aiml01-demo-capture --network "$NET" --user root \
  -v "$REPO_WIN:/repo:ro" -v "$REPO_WIN/$OUT:/out" \
  -v "$REPO_WIN/scripts/demo_capture.js:/usr/src/app/demo_capture.js:ro" \
  "$CHROME" node /usr/src/app/demo_capture.js > "$OUT/capture.log" 2>&1 &
CAP=$!
sleep 8
docker compose run --rm producer --rate 50 --limit 6000 \
  --drift amount_scale=3,shift_after=2500 2>&1 | tee "$OUT/producer.log"
wait $CAP || { echo "capture failed:"; tail -20 "$OUT/capture.log"; }

# One /score call on the most recently scored transaction (latest features for its client).
docker compose exec -T scorer python - > "$OUT/score.json" <<'EOF' || true
import json, sqlite3, urllib.error, urllib.request
import pandas as pd
from services.common import json_safe
tid = sqlite3.connect("/app/data/cases.db").execute(
    "SELECT transaction_id FROM scored ORDER BY rowid DESC LIMIT 1").fetchone()[0]
df = pd.read_parquet("/app/data/replay.parquet")
row = json_safe(df[df["TransactionID"] == tid].to_dict("records")[0])
req = urllib.request.Request("http://localhost:8000/score", data=json.dumps(row).encode(),
                             headers={"Content-Type": "application/json"})
try:
    print(json.dumps(json.loads(urllib.request.urlopen(req).read()), indent=2))
except urllib.error.HTTPError as e:
    print(json.dumps({"http_status": e.code, "detail": e.read().decode()}, indent=2))
EOF
docker compose exec -T scorer python -c \
  "import urllib.request; print(urllib.request.urlopen('http://localhost:8000/model').read().decode())" \
  > "$OUT/model.json" || true
docker compose logs --no-color consumer > "$OUT/consumer.log"
docker compose logs --no-color monitor > "$OUT/monitor.log"
cp serving_bundle/data/consumer_stats.json "$OUT/" 2>/dev/null || true
docker compose down
echo "frames: $(ls "$OUT"/f*.jpg | wc -l)  ->  python scripts/make_demo_video.py"
