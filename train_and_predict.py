from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    train_path = repo_root / "train_data.csv"
    test_path = repo_root / "test_data.csv"
    output_path = repo_root / "submission.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_cols = [col for col in train_df.columns if col not in {"id", "label"}]
    x_train = train_df[feature_cols]
    y_train = train_df["label"]
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
