from kfp import dsl
from kfp import compiler


# ==========================================
# KOMPONENT 1: PREPARE DATA
# ==========================================
@dsl.component(
    base_image="python:3.13",
    packages_to_install=["pandas", "pyarrow", "scikit-learn", "joblib"]
)
def prepare_data(
        raw_data_path: str,
        target_column: str,
        remove_empty: bool,
        imputer_num: str,
        imputer_cat: str,
        scaled_columns: list,
        onehot_columns: list,
        binary_columns: list,
        output_dataset: dsl.Output[dsl.Dataset],
        output_transformer: dsl.Output[dsl.Artifact]
):
    import pandas as pd
    from sklearn.impute import SimpleImputer
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    import joblib

    print(f"Pobieranie danych z MinIO: {raw_data_path}")

    # Konfiguracja połączenia z wewnętrznym MinIO w Kubernetesie
    # do zmiany secret by pobierało z env?
    storage_options = {
        "key": "minio",
        "secret": "minio123",
        "client_kwargs": {
            # To jest wewnętrzny adres URL MinIO w klastrze (nie musisz robić port-forward by to działało!)
            "endpoint_url": "http://minio-service.kubeflow.svc.cluster.local:9000"
        }
    }

    # Pandas sam ściągnie CSV z MinIO używając S3
    data = pd.read_csv(raw_data_path, storage_options=storage_options)

    print("Starting data cleaning")
    data.drop_duplicates(inplace=True)

    if remove_empty:
        print("Rows with empty values will be removed")
        data.dropna(inplace=True)
    else:
        print("Rows with empty values handled in data transformation")

    print("Starting data transformation")
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_num)),
        ("scaler", StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=imputer_cat)),
        ("onehot", OneHotEncoder(sparse_output=False))
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("num_pipeline", num_pipeline, scaled_columns),
            ("cat_pipeline", cat_pipeline, onehot_columns),
            ("bin_pipeline", SimpleImputer(strategy="most_frequent"), binary_columns)
        ],
        remainder="drop"
    ).set_output(transform="pandas")

    X = data.drop(columns=target_column)
    y = data[target_column]

    X_transformed = transformer.fit_transform(X)
    data_final = pd.concat([X_transformed, y.reset_index(drop=True)], axis=1)
    data_final.dropna(inplace=True)

    print(f"Exporting processed data to {output_dataset.path}")
    data_final.to_parquet(output_dataset.path)

    print(f"Exporting transformer to {output_transformer.path}")
    joblib.dump(transformer, output_transformer.path)


# ==========================================
# KOMPONENT 2: TRAIN MODEL
# ==========================================
@dsl.component(
    base_image="python:3.13",
    packages_to_install=["pandas", "scikit-learn", "pyarrow", "xgboost", "joblib"]
)
def train_model(
        input_dataset: dsl.Input[dsl.Dataset],
        target_column: str,
        test_size: float,
        random_state: int,
        algorithm: str,
        n_estimators: int,
        max_depth: int,
        scale_pos_weight: float,
        min_recall: float,
        min_accuracy: float,
        output_model: dsl.Output[dsl.Model]
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
    import joblib

    print(f"Loading data from {input_dataset.path}")
    data = pd.read_parquet(input_dataset.path)

    print("Splitting data")
    X = data.drop([target_column], axis=1)
    y = data[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Starting training using algorithm: {algorithm}")
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight="balanced")
    elif algorithm == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(scale_pos_weight=scale_pos_weight)
    else:
        from sklearn.linear_model import LogisticRegression
        # Przykładowe wagi dla regresji logistycznej
        model = LogisticRegression(class_weight={0: 1, 1: 11})

    model.fit(X_train, y_train)
    print("Model trained")

    y_pred = model.predict(X_test)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print(f"Recall: {recall}")
    print(f"Accuracy: {accuracy}")

    if recall > min_recall and accuracy > min_accuracy:
        print(f"Promoting model. Saving to {output_model.path}")
        joblib.dump(model, output_model.path)
    else:
        raise Exception(f"Model failed quality gate! Recall: {recall}, Accuracy: {accuracy}")


# ==========================================
# DEFINICJA POTOKU (PIPELINE)
# ==========================================
@dsl.pipeline(
    name="diabetes-prediction-pipeline",
    description="E2E Pipeline for Diabetes Prediction"
)
def diabetes_pipeline(
        raw_data_path: str = "/data/raw/diabetes.csv",  # UWAGA: Ścieżka musi być dostępna w kontenerze!
        target_column: str = "diabetes",
        remove_empty: bool = True,
        imputer_num: str = "median",
        imputer_cat: str = "most_frequent",
        test_size: float = 0.2,
        random_state: int = 42,
        algorithm: str = "random_forest",
        n_estimators: int = 100,
        max_depth: int = 5,
        scale_pos_weight: float = 9.0,
        min_recall: float = 0.7,
        min_accuracy: float = 0.3
):
    # Ważne: Typy złożone (jak listy) nie są domyślnie wspierane jako argumenty potoku na poziomie UI,
    # dlatego przekazujemy je statycznie wewnątrz definicji potoku.
    scaled_cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    onehot_cols = ["gender", "smoking_history"]
    binary_cols = ["hypertension", "heart_disease"]

    prepare_task = prepare_data(
        raw_data_path=raw_data_path,
        target_column=target_column,
        remove_empty=remove_empty,
        imputer_num=imputer_num,
        imputer_cat=imputer_cat,
        scaled_columns=scaled_cols,
        onehot_columns=onehot_cols,
        binary_columns=binary_cols
    )

    train_task = train_model(
        input_dataset=prepare_task.outputs["output_dataset"],
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
        algorithm=algorithm,
        n_estimators=n_estimators,
        max_depth=max_depth,
        scale_pos_weight=scale_pos_weight,
        min_recall=min_recall,
        min_accuracy=min_accuracy
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=diabetes_pipeline,
        package_path="diabetes_pipeline.yaml"
    )