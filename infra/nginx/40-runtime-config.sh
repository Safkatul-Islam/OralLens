#!/bin/sh
set -eu

api_base_url="${ORALLENS_API_BASE_URL:-}"

if ! printf '%s' "$api_base_url" \
    | grep -Eq '^https?://[A-Za-z0-9.-]+(:[0-9]{1,5})?$'; then
    echo "ORALLENS_API_BASE_URL must be an explicit HTTP(S) origin." >&2
    exit 1
fi

umask 022
printf 'window.__ORALLENS_RUNTIME_CONFIG__ = Object.freeze({ API_BASE_URL: "%s" });\n' \
    "$api_base_url" \
    > /usr/share/nginx/html/runtime-config.js
