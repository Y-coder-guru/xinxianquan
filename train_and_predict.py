from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

IMPUTE_STRATEGY = "median"
MODEL_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.08,
    "max_iter": 300,
    "random_state": 42,
}


def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing required file: {path.name}") from exc
    except Exception as exc:  # pragma: no cover - defensive for malformed input
        raise SystemExit(f"Failed to read {path.name}: {exc}") from exc


def validate_required_columns(
    df: pd.DataFrame, required: set[str], filename: str
) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(
            f"{filename} is missing required columns: {', '.join(missing)}"
        )


def validate_numeric_features(
    df: pd.DataFrame, feature_cols: list[str], filename: str
) -> None:
    non_numeric = [
        col
        for col in feature_cols
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise SystemExit(
            f"{filename} has non-numeric feature columns: "
            + ", ".join(sorted(non_numeric))
        )


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    train_path = repo_root / "train_data.csv"
    test_path = repo_root / "test_data.csv"
    output_path = repo_root / "submission.csv"

    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    validate_required_columns(train_df, {"id", "label"}, "train_data.csv")
    validate_required_columns(test_df, {"id"}, "test_data.csv")
    if train_df.empty:
        raise SystemExit("train_data.csv contains no rows.")
    if test_df.empty:
        raise SystemExit("test_data.csv contains no rows.")

    feature_cols = [col for col in train_df.columns if col not in {"id", "label"}]
    feature_col_set = set(feature_cols)
    x_train = train_df[feature_cols]
    y_train = train_df["label"]
    if y_train.isna().any():
        raise SystemExit("train_data.csv contains missing label values.")
    missing_in_test = sorted(feature_col_set - set(test_df.columns))
    if missing_in_test:
        raise SystemExit(
            "test_data.csv is missing feature columns: "
            + ", ".join(missing_in_test)
        )

    x_test = test_df[feature_cols]
    validate_numeric_features(train_df, feature_cols, "train_data.csv")
    validate_numeric_features(test_df, feature_cols, "test_data.csv")

    label_encoder = LabelEncoder()
    try:
        y_train_encoded = label_encoder.fit_transform(y_train)
    except Exception as exc:  # pragma: no cover - defensive for unexpected labels
        raise SystemExit(
            "Failed to encode labels in train_data.csv: "
            + str(exc)
        ) from exc

    model = make_pipeline(
        SimpleImputer(strategy=IMPUTE_STRATEGY),
        HistGradientBoostingClassifier(**MODEL_PARAMS),
    )
    model.fit(x_train, y_train_encoded)

    try:
        predictions = model.predict(x_test)
    except Exception as exc:  # pragma: no cover - defensive for model issues
        raise SystemExit(
            "Prediction failed for test_data.csv: "
            + str(exc)
        ) from exc
    predicted_labels = label_encoder.inverse_transform(predictions)

    submission = pd.DataFrame({"id": test_df["id"], "label": predicted_labels})
    submission.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Submission written to {output_path}")


if __name__ == "__main__":
    main()
