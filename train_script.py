# train_script.py
import pandas as pd
import s3fs
import os
import yaml
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score, accuracy_score

storage_options = {
    "key": os.environ.get("S3_ACCESS_KEY"),
    "secret": os.environ.get("S3_SECRET_KEY"),
    "client_kwargs": {"endpoint_url": os.environ.get("S3_ENDPOINT_URL")},
}


def _load_config(path: str, overrides: dict) -> dict:
    fs = s3fs.S3FileSystem(**storage_options)
    with fs.open(path, 'r') as f:
        config = yaml.safe_load(f)

    for key, value in overrides.items():
        if value is not None and value not in [-1, "-1", ""]:
            print(f"Override -> key: {key}, value: {value}")
            config['train'][key] = value

    if overrides.get('second_class_weight') not in [None, -1, "-1", ""]:
        config['train']['class_weight'] = {0: 1, 1: int(overrides['second_class_weight'])}

    return config


def load_data(dataset_path: str) -> pd.DataFrame:
    print(f"Loading data from {dataset_path}")
    return pd.read_parquet(dataset_path, storage_options=storage_options)


def split_data(config: dict, data: pd.DataFrame):
    print("Splitting data")
    X = data.drop([config["shared"]["target_column"]], axis=1)
    y = data[config["shared"]["target_column"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["train"]["test_size"],
        random_state=config["train"]["random_state"]
    )
    print("Data split finished")
    return X_train, X_test, y_train, y_test


def train_model(config: dict, X_train, y_train):
    print("Starting training")
    algorithm = config["train"]["algorithm"]
    print(f"using algorithm: {algorithm}")

    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=config["train"]["n_estimators"],
            max_depth=config["train"]["max_depth"],
            class_weight=config["train"].get("class_weight", "balanced")
        )
    elif algorithm == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            scale_pos_weight=config["train"].get("scale_pos_weight", 1),
        )
    else:
        print(f"Algorithm {algorithm} not configured, using Logistic Regression fallback")
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            class_weight=config["train"].get("class_weight", "balanced"),
        )

    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        print("Training failed - probably values weren't transformed - please check prepare step")
        raise e
    print("Model trained")
    return model


def evaluate_model(config: dict, X_test, y_test, model):
    y_pred = model.predict(X_test)

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"recall={recall}")
    print(f"accuracy={accuracy}")


def train(config_path: str, dataset_path: str, overrides: dict):
    config = _load_config(config_path, overrides)
    data = load_data(dataset_path)
    X_train, X_test, y_train, y_test = split_data(config, data)
    model = train_model(config, X_train, y_train)
    evaluate_model(config, X_test, y_test, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--dataset_path", type=str, required=True)

    parser.add_argument("--n_estimators", type=int, default=-1)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--algorithm", type=str, default="")
    parser.add_argument("--class_weight", type=str, default="")
    parser.add_argument("--second_class_weight", type=int, default=-1)

    args = parser.parse_args()

    katib_overrides = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'algorithm': args.algorithm,
        'class_weight': args.class_weight,
        'second_class_weight': args.second_class_weight
    }

    train(args.config_path, args.dataset_path, katib_overrides)