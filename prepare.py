# przygotowanie danych będzie składało się z 3 etapów:
# załadowanie danych, oczyszczenie danych, transformacja oraz eksport


# importy zależności
import pandas as pd
import yaml
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# załadowanie konfiguracji
def _load_config(congig_path: str) -> dict:

    with open(f"config.yaml", 'r') as f:
        return yaml.safe_load(f)


# import danych
def ingest_data(config: dict) -> pd.DataFrame:
    return pd.read_csv(config["data"]["raw_path"])

# walidacja danych
# sprawdza, czy zgadzają się nazwy kolumn i rodzaj danych
def validate_data(config: dict, data: pd.DataFrame) -> pd.DataFrame:
    required_columns = config["data"]["required_columns"]
    required_columns_names = required_columns.keys()
    missing_columns = data.columns.difference(required_columns_names)
    if missing_columns.size > 0:
        raise ValueError(f"Missing required columns: {missing_columns}")
    try:
        data = data.astype(required_columns)
    except ValueError as e:
        raise ValueError(f"Incorrect column data types: {e}")
    return data

# czyszczenie danych
def clean_data(config: dict, data: pd.DataFrame) -> pd.DataFrame:
    # usuwanie duplikatów
    print("Starting data cleaning")
    print("Dropping duplicated rows")
    data.drop_duplicates(inplace=True)

    if config["data"]["remove_empty"]:
        # decyzja z konfiguracji o usunięciu wierszy z brakującymi danymi
        print("Rows with empty values will be removed")
        data.dropna(inplace=True)
    else:
        # decyzja z konfiguracji o zastąpieniu braków danymi z transformera
        print("Rows with empty values handled in data transformation")

    print("Data cleaning ended")
    return data
# transformacja danych
# kodowanie, skalowanie, imputacja
def transform_data(config: dict, data: pd.DataFrame) -> pd.DataFrame:
    print("Starting data transformation")


    num_imputer_strategy = config["data"]["imputer_num"]
    cat_imputer_strategy = config["data"]["imputer_cat"]

    # kolumny wymagające skalowania
    scaled_columns = config["data"]["scaled_columns"]

    # kolumny wymagające kodowania
    encoded_columns = config["data"]["onehot_columns"]

    # kolumny wymagające jedynie imputacji
    binary_columns = config["data"]["binary_columns"]

    # pipeline dla danych liczbowych
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=num_imputer_strategy)),
        ("scaler", StandardScaler()),
    ])

    # pipeline dla danych kategorycznych
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=cat_imputer_strategy)),
        ("onehot", OneHotEncoder(sparse_output=False))
    ])

    # transformer dla cech
    transformer = ColumnTransformer(
        transformers=[
            ("num_pipeline", num_pipeline, scaled_columns),
            ("cat_pipeline", cat_pipeline, encoded_columns),
            ("bin_pipeline", SimpleImputer(strategy="most_frequent"), binary_columns)
        ],
        remainder="drop"
    ).set_output(transform="pandas")

    # oddzielenie cech od celu
    X = data.drop(columns=config["data"]["target_column"])
    y = data[config["data"]["target_column"]]

    X = transformer.fit_transform(X)

    data = pd.concat([X, y.reset_index(drop=True)], axis=1)
    data.dropna(inplace=True)

    # eksport transformera
    joblib.dump(transformer, config["data"]["transformer_path"])
    print("Data transformation ended")
    return data

# eksport danych
# eksport pandas dataframe do parquet
def export_data(config: dict, data: pd.DataFrame):
    print("Exporting data started")
    output_path = config["data"]["data_processed_path"]
    data.to_parquet(output_path)
    print(f"Data saved to: {output_path}")

def prepare():
    print("Starting data preparation")
    print("Loading config")
    config = _load_config()
    data = ingest_data(config)
    export_data(config, transform_data(config, clean_data(config, validate_data(config, data))))
    print("Data preparation finished")

if __name__ == "__main__":
    prepare()