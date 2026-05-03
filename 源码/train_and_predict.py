from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler

IMPUTE_STRATEGY = "median"
RANDOM_STATE = 42
BASELINE_MODEL_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.08,
    "max_iter": 300,
    "random_state": RANDOM_STATE,
}
CV_FOLDS = 5
SEARCH_FOLDS = 3
SEARCH_ITER = 30
EXTRATREES_PARAM_SPACE = {
    "extratreesclassifier__n_estimators": [300, 600],
    "extratreesclassifier__max_depth": [None, 20, 40],
    "extratreesclassifier__min_samples_split": [2, 5],
    "extratreesclassifier__min_samples_leaf": [1, 2],
    "extratreesclassifier__max_features": [0.5, 0.7, 1.0],
    "extratreesclassifier__class_weight": [None, "balanced"],
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


def describe_label_distribution(labels: pd.Series) -> None:
    counts = labels.value_counts().sort_index()
    ratios = (counts / counts.sum()).round(4)
    summary = pd.DataFrame({"count": counts, "ratio": ratios})
    print("Label distribution:")
    print(summary.to_string())


def evaluate_cv(
    name: str,
    pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
) -> float:
    scores = cross_val_score(
        pipeline, x_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1
    )
    print(f"{name} CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean()


def main() -> None:
    # 源码/ is one level below the repo root
    repo_root = Path(__file__).resolve().parent.parent
    train_path = repo_root / "train_data.csv"
    test_path = repo_root / "test_data.csv"

    model_dir = repo_root / "模型"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "model.joblib"

    output_dir = repo_root / "提交结果"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "submission.csv"

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
    describe_label_distribution(y_train)
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

    baseline_pipeline = make_pipeline(
        SimpleImputer(strategy=IMPUTE_STRATEGY),
        RobustScaler(),
        HistGradientBoostingClassifier(**BASELINE_MODEL_PARAMS),
    )
    baseline_cv = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    evaluate_cv(
        "HistGradientBoosting (baseline)",
        baseline_pipeline,
        x_train,
        y_train_encoded,
        baseline_cv,
    )

    search_pipeline = make_pipeline(
        SimpleImputer(strategy=IMPUTE_STRATEGY),
        RobustScaler(),
        ExtraTreesClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    )
    search_cv = StratifiedKFold(
        n_splits=SEARCH_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    search = RandomizedSearchCV(
        search_pipeline,
        param_distributions=EXTRATREES_PARAM_SPACE,
        n_iter=SEARCH_ITER,
        scoring="accuracy",
        cv=search_cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    search.fit(x_train, y_train_encoded)
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    model = search.best_estimator_

    joblib.dump({"pipeline": model, "label_encoder": label_encoder}, model_path)
    print(f"Model saved to {model_path}")

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
