FROM python:3.13-slim
RUN pip install --no-cache-dir pandas pyarrow scikit-learn xgboost joblib s3fs pyyaml
WORKDIR /app
COPY train_script.py /app/train.py
ENTRYPOINT ["python", "train.py"]
