from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


def load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError as exc:
        raise SystemExit(f"Missing required file: {path.name}") from exc
    except Exception as exc:  # pragma: no cover - defensive for malformed input
        raise SystemExit(f"Failed to read {path.name}: {exc}") from exc


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    train_path = repo_root / "train_data.csv"
    test_path = repo_root / "test_data.csv"
    output_path = repo_root / "submission.csv"

    train_df = load_csv(train_path)
    test_df = load_csv(test_path)

    feature_cols = [col for col in train_df.columns if col not in {"id", "label"}]
    x_train = train_df[feature_cols]
    y_train = train_df["label"]
    missing_in_test = sorted(set(feature_cols) - set(test_df.columns))
    if missing_in_test:
        raise SystemExit(
            "test_data.csv is missing feature columns: "
            + ", ".join(missing_in_test)
        )

    x_test = test_df[feature_cols]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.08,
            max_iter=300,
            random_state=42,
        ),
    )
    model.fit(x_train, y_encoded)

    predictions = model.predict(x_test)
    predicted_labels = label_encoder.inverse_transform(predictions)

    submission = pd.DataFrame({"id": test_df["id"], "label": predicted_labels})
    submission.to_csv(output_path, index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
