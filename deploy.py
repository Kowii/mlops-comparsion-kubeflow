import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import create_model


# załadowanie konfiguracji
def _load_config() -> dict:
    import yaml
    with open(f"config.yaml", 'r') as f:
        return yaml.safe_load(f)

config = _load_config()
app = FastAPI()

# załadowanie modelu
model = joblib.load(config["model"]["path"])

# załadowanie transformera
transformer = joblib.load(config["data"]["transformer_path"])

# załadowanie walidacji zmiennych z konfiguracji
type_map = {
    "str": str,
    "int": int,
    "float": float
}
raw_schema = config["model"]["input_schema"]
input_schema = {
    k: (type_map[v], ...) for k, v in raw_schema.items()
}

ModelRequest = create_model("ModelRequest", **input_schema)


@app.get("/")
def read_root():
    return {"message": f"Welcome to {config['model']['name']} API"}



@app.post("/predict")
def predict(request: ModelRequest):
    # wczytanie zapytania
    request_df = pd.DataFrame(request.dict(), index=[0])
    # transformacja zapytania
    transformed_request = transformer.transform(request_df)
    # predykcja
    prediction = model.predict(transformed_request)
    probability = model.predict_proba(transformed_request)
    return {
        "prediction": int(prediction[0]),
        "probability": float(probability[0][1]),
    }

