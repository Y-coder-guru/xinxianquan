from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

IMPUTE_STRATEGY_HIGH_MISSING = "median"
IMPUTE_STRATEGY_LOW_MISSING = "mean"
MISSING_RATIO_MAX_THRESHOLD = 0.2
MISSING_RATIO_AVG_THRESHOLD = 0.05
IMBALANCE_RATIO_THRESHOLD = 2.0
LOGISTIC_MAX_ITER = 2000
RANDOM_STATE = 42
BASELINE_MODEL_PARAMS = {
    "max_depth": 6,
    "learning_rate": 0.08,
    "max_iter": 300,
    "random_state": RANDOM_STATE,
}
CV_FOLDS = 5
CV_REPEATS = 2
SEARCH_FOLDS = 5
SEARCH_ITER = 60
EXTRATREES_PARAM_SPACE = {
    "extratreesclassifier__n_estimators": [300, 600, 800],
    "extratreesclassifier__max_depth": [None, 20, 40],
    "extratreesclassifier__min_samples_split": [2, 5, 10],
    "extratreesclassifier__min_samples_leaf": [1, 2, 4],
    "extratreesclassifier__max_features": ["sqrt", "log2", 0.5, 0.7, 0.9],
}
RANDOMFOREST_PARAM_SPACE = {
    "randomforestclassifier__n_estimators": [300, 600, 800],
    "randomforestclassifier__max_depth": [None, 20, 40],
    "randomforestclassifier__min_samples_split": [2, 5, 10],
    "randomforestclassifier__min_samples_leaf": [1, 2, 4],
    "randomforestclassifier__max_features": ["sqrt", "log2", 0.5, 0.7],
}
HGB_PARAM_SPACE = {
    "histgradientboostingclassifier__max_depth": [None, 6, 10],
    "histgradientboostingclassifier__learning_rate": [0.03, 0.05, 0.08, 0.1],
    "histgradientboostingclassifier__max_iter": [200, 400, 600],
    "histgradientboostingclassifier__max_leaf_nodes": [31, 63, 127],
    "histgradientboostingclassifier__min_samples_leaf": [20, 50, 100],
    "histgradientboostingclassifier__l2_regularization": [0.0, 0.1, 1.0],
}
LOGISTIC_PARAM_SPACE = {
    "logisticregression__C": [0.1, 0.3, 1.0, 3.0, 10.0],
    "logisticregression__penalty": ["l1", "l2"],
}


@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_distributions: dict[str, Any]
    n_iter: int
    needs_sample_weight: bool


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


def summarize_missingness(
    df: pd.DataFrame, feature_cols: list[str], filename: str
) -> pd.Series:
    missing_ratios = df[feature_cols].isna().mean().sort_values(ascending=False)
    print(f"{filename} missing ratio (top 10):")
    print(missing_ratios.head(10).round(4).to_string())
    print(f"{filename} missing ratio (avg): {missing_ratios.mean():.4f}")
    return missing_ratios


def summarize_feature_distribution(
    df: pd.DataFrame, feature_cols: list[str], filename: str
) -> None:
    summary = df[feature_cols].describe().T
    display_cols = ["mean", "std", "min", "max"]
    print(f"{filename} feature summary (first 10):")
    print(summary[display_cols].head(10).round(4).to_string())


def find_constant_features(
    df: pd.DataFrame, feature_cols: list[str]
) -> list[str]:
    return [
        col for col in feature_cols if df[col].nunique(dropna=False) <= 1
    ]


def describe_label_distribution(labels: pd.Series) -> None:
    counts = labels.value_counts().sort_index()
    ratios = (counts / counts.sum()).round(4)
    summary = pd.DataFrame({"count": counts, "ratio": ratios})
    print("Label distribution:")
    print(summary.to_string())


def calculate_imbalance_ratio(labels: pd.Series) -> float:
    counts = labels.value_counts()
    if len(counts) <= 1:
        return 1.0
    return counts.max() / counts.min()


def choose_scoring(labels: pd.Series, imbalance_ratio: float) -> tuple[str, str]:
    if imbalance_ratio >= IMBALANCE_RATIO_THRESHOLD:
        if labels.nunique() == 2:
            return "roc_auc", "ROC-AUC"
        return "f1_macro", "F1-macro"
    return "accuracy", "accuracy"


def choose_impute_strategy(missing_ratio: pd.Series) -> str:
    avg_missing = missing_ratio.mean()
    max_missing = missing_ratio.max()
    if max_missing >= MISSING_RATIO_MAX_THRESHOLD or avg_missing >= MISSING_RATIO_AVG_THRESHOLD:
        strategy = IMPUTE_STRATEGY_HIGH_MISSING
    else:
        strategy = IMPUTE_STRATEGY_LOW_MISSING
    print(
        "Imputation strategy selected: "
        f"{strategy} (avg missing {avg_missing:.4f}, max missing {max_missing:.4f})"
    )
    return strategy


def setup_class_weighting(model, use_class_weight: bool) -> bool:
    """Apply class_weight when supported; return True if sample_weight is needed."""
    if not use_class_weight:
        return False
    if "class_weight" in model.get_params():
        model.set_params(class_weight="balanced")
        return False
    return True


def build_pipeline(
    impute_strategy: str, model, scale: bool
) -> Pipeline:
    steps = [SimpleImputer(strategy=impute_strategy)]
    if scale:
        steps.append(RobustScaler())
    steps.append(model)
    return make_pipeline(*steps)


def build_fit_params(
    pipeline: Pipeline, sample_weight: np.ndarray | pd.Series | None
) -> dict:
    """Build fit params dict with the estimator step name for `sample_weight`."""
    if sample_weight is None:
        return {}
    estimator_step = pipeline.steps[-1][0]
    return {f"{estimator_step}__sample_weight": sample_weight}


def evaluate_cv(
    name: str,
    pipeline,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    cv: StratifiedKFold,
    scoring: str,
    fit_params: dict | None = None,
) -> float:
    fit_params = fit_params or {}
    scores = cross_val_score(
        pipeline,
        x_train,
        y_train,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        fit_params=fit_params,
    )
    print(
        f"{name} CV {scoring}: {scores.mean():.4f} ± {scores.std():.4f}"
    )
    return scores.mean()


def build_model_specs(impute_strategy: str, use_class_weight: bool) -> list[ModelSpec]:
    specs: list[ModelSpec] = []

    extratrees = ExtraTreesClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    extratrees_needs_weight = setup_class_weighting(
        extratrees, use_class_weight
    )
    specs.append(
        ModelSpec(
            name="ExtraTrees",
            pipeline=build_pipeline(impute_strategy, extratrees, scale=False),
            param_distributions=EXTRATREES_PARAM_SPACE,
            n_iter=SEARCH_ITER,
            needs_sample_weight=extratrees_needs_weight,
        )
    )

    random_forest = RandomForestClassifier(
        random_state=RANDOM_STATE, n_jobs=-1
    )
    random_forest_needs_weight = setup_class_weighting(
        random_forest, use_class_weight
    )
    specs.append(
        ModelSpec(
            name="RandomForest",
            pipeline=build_pipeline(impute_strategy, random_forest, scale=False),
            param_distributions=RANDOMFOREST_PARAM_SPACE,
            n_iter=SEARCH_ITER,
            needs_sample_weight=random_forest_needs_weight,
        )
    )

    hgb = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    hgb_needs_weight = setup_class_weighting(hgb, use_class_weight)
    specs.append(
        ModelSpec(
            name="HistGradientBoosting",
            pipeline=build_pipeline(impute_strategy, hgb, scale=False),
            param_distributions=HGB_PARAM_SPACE,
            n_iter=SEARCH_ITER,
            needs_sample_weight=hgb_needs_weight,
        )
    )

    logistic = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=LOGISTIC_MAX_ITER,
        # Saga supports the L1 penalty used in the search space.
        solver="saga",
    )
    logistic_needs_weight = setup_class_weighting(logistic, use_class_weight)
    specs.append(
        ModelSpec(
            name="LogisticRegression",
            pipeline=build_pipeline(impute_strategy, logistic, scale=True),
            param_distributions=LOGISTIC_PARAM_SPACE,
            n_iter=SEARCH_ITER,
            needs_sample_weight=logistic_needs_weight,
        )
    )

    return specs


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

    feature_cols = [
        col for col in train_df.columns if col not in {"id", "label"}
    ]
    feature_col_set = set(feature_cols)
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

    validate_numeric_features(train_df, feature_cols, "train_data.csv")
    validate_numeric_features(test_df, feature_cols, "test_data.csv")

    constant_features = find_constant_features(train_df, feature_cols)
    if constant_features:
        print(
            f"Dropping {len(constant_features)} constant feature(s): "
            + ", ".join(constant_features)
        )
        feature_cols = [
            col for col in feature_cols if col not in constant_features
        ]
    if not feature_cols:
        raise SystemExit("No usable feature columns remain after filtering.")

    x_train = train_df[feature_cols]
    x_test = test_df[feature_cols]
    summarize_feature_distribution(x_train, feature_cols, "train_data.csv")
    missing_ratio_train = summarize_missingness(
        train_df, feature_cols, "train_data.csv"
    )
    summarize_missingness(test_df, feature_cols, "test_data.csv")
    impute_strategy = choose_impute_strategy(missing_ratio_train)

    imbalance_ratio = calculate_imbalance_ratio(y_train)
    print(f"Imbalance ratio (max/min): {imbalance_ratio:.2f}")
    use_class_weight = imbalance_ratio >= IMBALANCE_RATIO_THRESHOLD
    if use_class_weight:
        print(
            "Class imbalance detected; enabling class_weight or sample_weight."
        )
    else:
        print("Class balance within threshold; class_weight disabled.")
    scoring, scoring_label = choose_scoring(y_train, imbalance_ratio)
    print(f"Optimization metric: {scoring_label}")

    label_encoder = LabelEncoder()
    try:
        y_train_encoded = label_encoder.fit_transform(y_train)
    except Exception as exc:  # pragma: no cover - defensive for unexpected labels
        raise SystemExit(
            "Failed to encode labels in train_data.csv: "
            + str(exc)
        ) from exc

    baseline_model = HistGradientBoostingClassifier(**BASELINE_MODEL_PARAMS)
    baseline_needs_weight = setup_class_weighting(
        baseline_model, use_class_weight
    )
    baseline_pipeline = build_pipeline(
        impute_strategy, baseline_model, scale=False
    )

    search_cv = StratifiedKFold(
        n_splits=SEARCH_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    evaluation_cv = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )
    model_specs = build_model_specs(impute_strategy, use_class_weight)
    needs_sample_weight = use_class_weight and (
        baseline_needs_weight
        or any(spec.needs_sample_weight for spec in model_specs)
    )
    sample_weight = None
    if needs_sample_weight:
        # Compute once and reuse for all models that need sample_weight.
        sample_weight = compute_sample_weight(
            class_weight="balanced", y=y_train_encoded
        )
        print("Sample weights computed for models without class_weight.")

    baseline_cv = RepeatedStratifiedKFold(
        n_splits=CV_FOLDS, n_repeats=CV_REPEATS, random_state=RANDOM_STATE
    )
    baseline_fit_params = {}
    if baseline_needs_weight:
        if sample_weight is None:
            raise SystemExit(
                "Sample weights required for baseline model due to class "
                "imbalance and lack of class_weight support."
            )
        baseline_fit_params = build_fit_params(
            baseline_pipeline, sample_weight
        )
    evaluate_cv(
        "HistGradientBoosting (baseline)",
        baseline_pipeline,
        x_train,
        y_train_encoded,
        baseline_cv,
        scoring,
        baseline_fit_params,
    )
    results = []

    for spec in model_specs:
        print(f"Searching {spec.name}...")
        search = RandomizedSearchCV(
            spec.pipeline,
            param_distributions=spec.param_distributions,
            n_iter=spec.n_iter,
            scoring=scoring,
            cv=search_cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1,
        )
        fit_params = {}
        if spec.needs_sample_weight:
            if sample_weight is None:
                raise SystemExit(
                    "Sample weights required for "
                    f"{spec.name} due to class imbalance and lack of "
                    "class_weight support."
                )
            fit_params = build_fit_params(spec.pipeline, sample_weight)
        search.fit(x_train, y_train_encoded, **fit_params)
        print(
            f"{spec.name} best CV {scoring}: {search.best_score_:.4f}"
        )
        print(f"{spec.name} best params: {search.best_params_}")

        stable_score = evaluate_cv(
            f"{spec.name} (tuned)",
            search.best_estimator_,
            x_train,
            y_train_encoded,
            evaluation_cv,
            scoring,
            fit_params,
        )
        results.append(
            {
                "name": spec.name,
                "stable_score": stable_score,
                "search_score": search.best_score_,
                "best_params": search.best_params_,
                "estimator": search.best_estimator_,
                "fit_params": fit_params,
            }
        )

    if not results:
        raise SystemExit("No model results were generated.")
    invalid_results = [
        result
        for result in results
        if pd.isna(result["stable_score"])
    ]
    if invalid_results:
        print(
            "Warning: filtered models with invalid CV scores: "
            + ", ".join(result["name"] for result in invalid_results)
        )
    valid_results = [
        result
        for result in results
        if pd.notna(result["stable_score"])
    ]
    if not valid_results:
        raise SystemExit("All model evaluations failed to return valid scores.")
    best_result = max(valid_results, key=lambda item: item["stable_score"])
    print("Model comparison summary:")
    for result in results:
        print(
            f"{result['name']}: search {result['search_score']:.4f}, "
            f"stable {result['stable_score']:.4f}"
        )
    print(
        "Selected model: "
        f"{best_result['name']} (stable {best_result['stable_score']:.4f})"
    )
    print(f"Selected params: {best_result['best_params']}")

    model = best_result["estimator"]
    model.fit(x_train, y_train_encoded, **best_result["fit_params"])

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
