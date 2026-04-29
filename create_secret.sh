#!/bin/bash

# ==============================================================================
# Skrypt do tworzenia Secretu w Kubernetes dla poświadczeń S3 (Minio)
# ==============================================================================

# Definicja zmiennych
SECRET_NAME="s3-credentials"
NAMESPACE="kubeflow"
S3_ACCESS_KEY="example-access=key"
S3_SECRET_KEY="example-secret-key"
S3_ENDPOINT="http://example-svc:9000"

echo "Rozpoczynam tworzenie secretu: $SECRET_NAME w przestrzeni nazw: $NAMESPACE..."

# Wykonanie polecenia kubectl 
kubectl create secret generic "$SECRET_NAME" \
  --from-literal=S3_ACCESS_KEY="$S3_ACCESS_KEY" \
  --from-literal=S3_SECRET_KEY="$S3_SECRET_KEY" \
  --from-literal=S3_ENDPOINT_URL="$S3_ENDPOINT" \
  -n "$NAMESPACE" \
  --dry-run=client -o yaml | kubectl apply -f -

# Sprawdzenie, czy polecenie zakończyło się sukcesem
if [ $? -eq 0 ]; then
    echo "Sukces: Secret '$SECRET_NAME' został utworzony/zaktualizowany."
else
    echo "Błąd: Nie udało się utworzyć secretu."
    exit 1
fi
