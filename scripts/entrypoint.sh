#!/bin/bash
set -euo pipefail

echo "=== Crypto Data Streaming Container Starting ==="
echo "Python: $(python --version)"
echo "Java: $(java -version 2>&1 | head -1)"

exec "$@"
