# skrypt wytrenuje model na podstawie przygotowanych danych i wyeksportuje go
# do formatu .pkl
import pandas as pd


# załadowanie konfiguracji
def _load_config() -> dict:
    import yaml
    with open(f"config.yaml", 'r') as f:
        return yaml.safe_load(f)

# załadowanie danych
def load_data(config: dict) -> pd.DataFrame:
    print("Loading data")
    return pd.read_parquet(config["shared"]["data_processed_path"])

# podział danych
def split_data(config: dict, data: pd.DataFrame):
    from sklearn.model_selection import train_test_split
    print("Data loaded")
    print("Splitting data")
    # podział na X i y
    X = data.drop([config["data"]["target_column"]], axis=1)
    y = data[config["data"]["target_column"]]
    # podział na dane testowe i treningowe
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config["train"]["test_size"],
        random_state=config["train"]["random_state"]
    )
    print("Data split finished")
    return X_train, X_test, y_train, y_test

def train_model(config: dict, X_train, y_train):
    print("Starting training")
    # wybór algorytmu z konfiguracji
    algorithm = config["train"]["algorithm"]
    print(f"using algorithm: {algorithm}")
    if config["train"]["algorithm"] == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(
            n_estimators=config["train"]["n_estimators"],
            max_depth=config["train"]["max_depth"],
            class_weight="balanced"
        )
    elif config["train"]["algorithm"] == "xgboost":
        from xgboost import XGBClassifier
        model = XGBClassifier(
            scale_pos_weight=config["train"]["scale_pos_weight"],
        )
    else:
        # wybór domyślnego algorytmu: logisticregression
        print(f"using algorithm: Logistic Regression from sklearn"
              f"model {config["train"]["algorithm"]} not configured")
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(
            class_weight=config["train"]["class_weight"],
        )
    try:
        model.fit(X_train, y_train)
    except ValueError as e:
        print("Training failed - probably values weren't transformed - please prepare data with encoding")
        raise e
    print("Model trained")
    return model

def evaluate_model(config: dict, X_test, y_test, model):
    from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
    y_pred = model.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    recall = recall_score(y_test, y_pred)
    print(f"Recall: {recall}")
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")
    min_recall = config["train"]["min_recall"]
    min_accuracy = config["train"]["min_accuracy"]
    if recall > min_recall and accuracy > min_accuracy:
        import joblib
        print(f"Accuracy over {min_accuracy}")
        print(f"Recall over {min_recall} - promoting model")
        path = config["shared"]["model_path"]
        joblib.dump(model, path)
        print(f"Model saved to {path}")
    else:
        print(f"Recall under {min_recall} or accuracy under {min_accuracy}"
              f" - model not promoted")


def train():
    print("Training process started")
    print("Loading data")
    config = _load_config()
    data = load_data(config)
    X_train, X_test, y_train, y_test = split_data(config, data)
    model = train_model(config, X_train, y_train)
    evaluate_model(config, X_test, y_test, model)

if __name__ == "__main__":
    train()
