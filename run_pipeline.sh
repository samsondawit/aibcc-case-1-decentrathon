set -euo pipefail; set -a && [ -f .env ] && . ./.env && set +a; \
for s in build_features.py build_benefits.py generate_pushes.py export_final.py; do \
  echo ">>> running $s"; python3 "$s"; \
done; \
echo "All stages done."
