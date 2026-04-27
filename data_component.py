# data_component.py

from kfp import dsl
from kfp.dsl import component

@component(
    base_image="python:3.13",
    packages_to_install=["pandas", "pyarrow", "scikit-learn", "joblib", "s3fs", "pyYAML"]
)
def prepare(
    config_path: str,
    output_dataset: dsl.Output[dsl.Dataset],
    output_transformer: dsl.Output[dsl.Artifact]
):
    """
    Conducts data preparation for ML Pipeline
    :param output_transformer: Path to export transformer in joblib format
    :param output_dataset: Path to export dataset in parquet format
    :param config_path: Path to config file in Minio

    :return:
    """
    # importy zależności
    import pandas as pd
    import yaml
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    import joblib
    import os
    import s3fs

    storage_options = {
        "key": os.environ["S3_ACCESS_KEY"],
        "secret": os.environ["S3_SECRET_KEY"],
        "client_kwargs": {"endpoint_url": os.environ["S3_ENDPOINT_URL"]},
    }

    # załadowanie konfiguracji
    def _load_config(path: str) -> dict:
        fs = s3fs.S3FileSystem(**storage_options)
        with fs.open(path, 'r') as f:
            return yaml.safe_load(f)

    # import danych
    def ingest_data(config: dict) -> pd.DataFrame:
        raw_path = config["data"]["raw_path"]
        return pd.read_csv(raw_path, storage_options=storage_options)

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
    def transform_data(config: dict, data: pd.DataFrame):
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

        print("Data transformation ended")
        return data, transformer

    # eksport danych
    # eksport pandas dataframe do parquet
    def export_data(data: pd.DataFrame):
        print("Exporting data started")
        output_path = output_dataset.path
        data.to_parquet(output_path)
        print(f"Data saved to: {output_path}")

    # eksport transformatora
    # eksport do formatu joblib
    def export_transformer(transformer):
        print("Exporting transformer started")
        output_path = output_transformer.path
        joblib.dump(transformer, output_path)
        print(f"Data saved to: {output_path}")


    print("Starting data preparation")
    print("Loading config")
    config = _load_config(config_path)
    data = ingest_data(config)
    data = validate_data(config, data)
    data = clean_data(config, data)
    data, transformer = transform_data(config, data)
    export_data(data)
    export_transformer(transformer)
    print("Data preparation finished")


