#!/usr/bin/env bash
# Mide el arranque en frío por INACTIVIDAD: espera, lanza una búsqueda real y mira si hay una
# instancia nueva (cambia instance.id en /api/health). Se detiene en la primera instancia nueva.
#   PYTHON=.venv/Scripts/python.exe LAST_ID=<id actual> scripts/idle_probe.sh https://lente-fiqa.vercel.app 600 900 1200
# Los argumentos tras la URL son las esperas (s) entre sondas. Nota: Vercel puede escalar a varias
# instancias, así que un id distinto prueba una instancia nueva, y el tiempo total prueba si fue en frío.
BASE="${1:?url base}"; shift
Q='can%20I%20deduct%20a%20home%20office%20if%20I%20work%20remotely%20as%20a%20freelancer'
PY="${PYTHON:-python}"
last_id="${LAST_ID:-}"
for wait in "$@"; do
  echo "[$(date -u +%H:%M:%S)] esperando ${wait}s sin tráfico..."
  sleep "$wait"
  start=$(date -u +%H:%M:%S.%3N)
  line=$(curl -sS -o /dev/null -w "http=%{http_code} total=%{time_total}s ttfb=%{time_starttransfer}s" "$BASE/api/search?q=$Q&method=semantic&n=3")
  info=$(curl -sS "$BASE/api/health" | "$PY" "$(dirname "$0")/health_summary.py")
  echo "[$start] búsqueda: $line | $info"
  id=$(echo "$info" | sed -E 's/instancia=([a-z0-9]+).*/\1/')
  if [ -n "$last_id" ] && [ -n "$id" ] && [ "$id" != "$last_id" ]; then
    echo "=> INSTANCIA NUEVA tras ${wait}s de inactividad (arranque en frío por inactividad)"; exit 0
  fi
  last_id="$id"
done
echo "=> la instancia siguió caliente en todas las sondas"
