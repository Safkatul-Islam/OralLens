#!/bin/sh
set -eu

port="${PORT:-10000}"
case "$port" in
    ""|*[!0-9]*)
        echo "PORT must be an integer between 1 and 65535." >&2
        exit 1
        ;;
esac
if [ "$port" -lt 1 ] || [ "$port" -gt 65535 ]; then
    echo "PORT must be an integer between 1 and 65535." >&2
    exit 1
fi

python /opt/orallens/infra/render/fetch_model.py

exec python -m uvicorn app.main:app \
    --app-dir /opt/orallens/backend \
    --host 0.0.0.0 \
    --port "$port" \
    --workers 1
