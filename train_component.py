from kfp import dsl
from kfp.dsl import component


@component(
    base_image="python:3.13",
    packages_to_install=["pandas", "pyarrow", "scikit-learn", "xgboost", "joblib", "s3fs", "pyyaml"]
)
def train(
        config_path: str,
        input_dataset: dsl.Input[dsl.Dataset],
        output_model: dsl.Output[dsl.Model],
        kfp_metrics: dsl.Output[dsl.Metrics],
        override_n_estimators: int = 0,
        override_max_depth: int = 0,
        override_algorithm: str = ""
):
    """
    Conducts model training and exports trained model
    :param config_path: Path to config file in Minio
    :param input_dataset: Prepared dataset from previous pipeline step
    :param output_model: Path to export model in joblib format
    :param kfp_metrics: Output metrics for Kubeflow UI
    """
    import pandas as pd
    import yaml
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
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
            config = yaml.safe_load(f)
        if override_n_estimators > 0:
            config["train"]["n_estimators"] = override_n_estimators
            print(f"Overridden n_estimators: {override_n_estimators}")
        if override_max_depth > 0:
            config["train"]["max_depth"] = override_max_depth
            print(f"Overridden max_depth: {override_max_depth}")
        if override_algorithm != "":
            config["train"]["algorithm"] = override_algorithm
            print(f"Overridden algorithm: {override_algorithm}")
        return config

    # podział danych
    def split_data(config: dict, data: pd.DataFrame):
        print("Splitting data")
        X = data.drop([config["data"]["target_column"]], axis=1)
        y = data[config["data"]["target_column"]]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=config["train"]["test_size"],
            random_state=config["train"]["random_state"]
        )
        return X_train, X_test, y_train, y_test

    # trening
    def train_model(config: dict, X_train, y_train):
        print("Starting training")
        algorithm = config["train"]["algorithm"]
        print(f"using algorithm: {algorithm}")

        if algorithm == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(
                n_estimators=config["train"]["n_estimators"],
                max_depth=config["train"]["max_depth"],
                class_weight="balanced"
            )
        elif algorithm == "xgboost":
            from xgboost import XGBClassifier
            model = XGBClassifier(
                scale_pos_weight=config["train"]["scale_pos_weight"],
            )
        else:
            print(f"Algorithm {algorithm} not configured, using Logistic Regression fallback")
            from sklearn.linear_model import LogisticRegression
            # Używamy get() by uniknąć KeyError jeśli class_weight nie będzie słownikiem
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

    # ewaluacja
    def evaluate_model(config: dict, X_test, y_test, model):
        y_pred = model.predict(X_test)

        print("Classification report:")
        print(classification_report(y_test, y_pred))
        print("Confusion matrix:")
        print(confusion_matrix(y_test, y_pred))

        recall = recall_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Recall: {recall}")
        print(f"Accuracy: {accuracy}")

        # logowanie metryk do kubeflow ui
        kfp_metrics.log_metric("recall", float(recall))
        kfp_metrics.log_metric("accuracy", float(accuracy))

        min_recall = config["train"]["min_recall"]
        min_accuracy = config["train"]["min_accuracy"]

        if recall > min_recall and accuracy > min_accuracy:
            print(f"Accuracy over {min_accuracy} and Recall over {min_recall} - promoting model")
            joblib.dump(model, output_model.path)
            print(f"Model saved to Kubeflow Artifact Store: {output_model.path}")
        else:
            error_msg = f"Quality gate failed! Recall: {recall} < {min_recall} or Accuracy: {accuracy} < {min_accuracy}"
            print(error_msg)
            raise RuntimeError(error_msg)


    print("Training process started")
    print("Loading config")
    config = _load_config(config_path)

    print(f"Loading prepared data from Kubeflow Dataset: {input_dataset.path}")
    data = pd.read_parquet(input_dataset.path)

    X_train, X_test, y_train, y_test = split_data(config, data)
    model = train_model(config, X_train, y_train)
    evaluate_model(config, X_test, y_test, model)
    print("Training process finished successfully")