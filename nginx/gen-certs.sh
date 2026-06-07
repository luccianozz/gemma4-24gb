#!/usr/bin/env bash
# Generate self-signed TLS cert for dev / single-node.
# For production, replace with Let's Encrypt (certbot sidecar) or ACM-issued certs.

set -euo pipefail
cd "$(dirname "$0")/certs"

if [[ -f server.crt && -f server.key ]]; then
    echo "certs already exist — skipping. delete certs/* to regenerate."
    exit 0
fi

CN="${CERT_CN:-gemma4-24gb.local}"
DAYS="${CERT_DAYS:-365}"

openssl req -x509 -newkey rsa:4096 -sha256 -days "$DAYS" -nodes \
    -keyout server.key -out server.crt \
    -subj "/CN=$CN" \
    -addext "subjectAltName=DNS:$CN,DNS:localhost,IP:127.0.0.1"

chmod 600 server.key
echo "generated certs/server.crt + certs/server.key for CN=$CN (valid $DAYS days)"
