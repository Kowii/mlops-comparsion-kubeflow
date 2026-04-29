#!/bin/bash

# ==============================================================================
# Kompleksowy skrypt konfiguracji dostępu S3 dla KServe
# ==============================================================================

# 1. Konfiguracja zmiennych
SECRET_NAME="example-credentials"
NAMESPACE="kubeflow"
S3_USER="user"
S3_PASS="pass"
S3_ENDPOINT="http://service.kubeflow.svc.cluster.local:9000"

echo "--- Konfiguracja S3 dla namespace: $NAMESPACE ---"

# 2. Usunięcie starego sekretu (jeśli istnieje)
microk8s kubectl delete secret $SECRET_NAME -n $NAMESPACE --ignore-not-found

# 3. Tworzenie sekretu z nazwami pól akceptowanymi przez KServe/AWS SDK
# Dodajemy zarówno S3_ jak i AWS_ dla maksymalnej kompatybilności
echo "Tworzenie sekretu: $SECRET_NAME..."
microk8s kubectl create secret generic $SECRET_NAME \
  --from-literal=AWS_ACCESS_KEY_ID=$S3_USER \
  --from-literal=AWS_SECRET_ACCESS_KEY=$S3_PASS \
  --from-literal=S3_ACCESS_KEY=$S3_USER \
  --from-literal=S3_SECRET_KEY=$S3_PASS \
  --from-literal=S3_ENDPOINT_URL=$S3_ENDPOINT \
  --from-literal=AWS_ENDPOINT_URL=$S3_ENDPOINT \
  -n $NAMESPACE

# 4. Adnotacja sekretu 
microk8s kubectl annotate secret $SECRET_NAME -n $NAMESPACE \
  "serving.kserve.io/s3-endpoint=$S3_ENDPOINT" \
  "serving.kserve.io/s3-usehttps=0" \
  --overwrite

# 5. Połączenie sekretu z kontem serwisowym
# To polecenie sprawia, że pody w tym namespace będą widzieć te poświadczenia
echo "Przypisuję sekret do ServiceAccount 'default'..."
microk8s kubectl patch serviceaccount default -n $NAMESPACE \
  -p "{\"secrets\": [{\"name\": \"$SECRET_NAME\"}]}"

# 6. Weryfikacja
echo "---------------------------------------------------"
if [ $? -eq 0 ]; then
    echo "SUKCES: Konfiguracja zakończona."
else
    echo "BŁĄD: Coś poszło nie tak podczas konfiguracji."
    exit 1
fi
echo "---------------------------------------------------"
