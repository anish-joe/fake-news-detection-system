import os
import re
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from textclean import clean

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "news.csv")
EXTERNAL_PATH = os.path.join(BASE_DIR, "external.csv")
RANDOM_STATE = 20

LABEL_MAP = {
    "TRUE": "TRUE",
    "REAL": "TRUE",
    "1": "TRUE",
    "1.0": "TRUE",
    "FAKE": "FAKE",
    "FALSE": "FAKE",
    "0": "FAKE",
    "0.0": "FAKE",
}


def normalize_text(value):
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def load(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = pd.read_csv(path, low_memory=False)

    required = {"title", "text", "label"}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(
            f"{os.path.basename(path)} is missing columns: {', '.join(sorted(missing))}"
        )

    data["title"] = data["title"].fillna("").astype(str).str.strip()
    data["text"] = data["text"].fillna("").astype(str).str.strip()

    raw_labels = data["label"].astype("string").str.upper().str.strip()
    data["label"] = raw_labels.map(LABEL_MAP)

    invalid_count = int(data["label"].isna().sum())

    if invalid_count:
        print(f"Removed {invalid_count} rows with invalid labels.")

    data = data.dropna(subset=["label"]).copy()
    data = data[data["text"].str.len() > 50].copy()
    data["body"] = (data["title"] + " " + data["text"]).str.strip()
    data["_text_key"] = data["text"].map(normalize_text)

    conflicting_counts = data.groupby("_text_key")["label"].nunique()
    conflicting_keys = set(conflicting_counts[conflicting_counts > 1].index)

    if conflicting_keys:
        before = len(data)
        data = data[~data["_text_key"].isin(conflicting_keys)].copy()
        print(
            f"Removed {before - len(data)} rows with identical text and conflicting labels."
        )

    before = len(data)
    data = data.drop_duplicates(subset="_text_key", keep="first").copy()

    if before != len(data):
        print(f"Removed {before - len(data)} duplicate articles.")

    data = data.drop(columns=["_text_key"]).reset_index(drop=True)

    if data.empty:
        raise ValueError(f"No usable records found in {path}.")

    return data


def make_vector():
    return TfidfVectorizer(
        preprocessor=clean,
        stop_words="english",
        sublinear_tf=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
        max_features=300000,
        dtype="float32",
    )


def make_candidates():
    return {
        "LogisticRegression": LogisticRegression(
            C=10,
            max_iter=2000,
            solver="liblinear",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "LinearSVC": LinearSVC(
            C=1,
            class_weight="balanced",
            max_iter=5000,
            random_state=RANDOM_STATE,
        ),
    }


def make_pipeline(classifier):
    return Pipeline(
        [
            ("vectorizer", make_vector()),
            ("classifier", classifier),
        ]
    )


def main():
    data = load(DATA_PATH)

    print("\nDataset distribution")
    print(data["label"].value_counts())
    print(f"\nTotal usable records: {len(data)}")

    counts = data["label"].value_counts()

    if set(counts.index) != {"FAKE", "TRUE"} or counts.min() < 2:
        raise ValueError(
            "The dataset must contain at least two usable examples of each class."
        )

    x_train, x_test, y_train, y_test = train_test_split(
        data["body"],
        data["label"],
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["label"],
    )

    train_counts = y_train.value_counts()
    cv_splits = min(5, int(train_counts.min()))

    if cv_splits < 2:
        raise ValueError(
            "Insufficient training examples for stratified cross-validation."
        )

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
    }

    candidates = make_candidates()
    scores = {}

    print(f"\n{cv_splits}-fold cross-validation")

    for name, classifier in candidates.items():
        pipeline = make_pipeline(classifier)

        results = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        mean_f1 = results["test_f1_macro"].mean()
        mean_accuracy = results["test_accuracy"].mean()

        scores[name] = {
            "f1_macro": float(mean_f1),
            "accuracy": float(mean_accuracy),
        }

        print(
            f"{name}: "
            f"Macro F1 = {mean_f1 * 100:.2f}% | "
            f"Accuracy = {mean_accuracy * 100:.2f}%"
        )

    best = max(
        scores,
        key=lambda name: (
            scores[name]["f1_macro"],
            scores[name]["accuracy"],
        ),
    )

    print(f"\nSelected model: {best}")

    evaluation_pipeline = make_pipeline(candidates[best])
    evaluation_pipeline.fit(x_train, y_train)

    predictions = evaluation_pipeline.predict(x_test)

    print("\nHoldout evaluation")
    print(f"Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print(f"Macro F1: {f1_score(y_test, predictions, average='macro') * 100:.2f}%")
    print(
        classification_report(
            y_test,
            predictions,
            labels=["FAKE", "TRUE"],
            target_names=["FAKE", "TRUE"],
            zero_division=0,
        )
    )
    print("Confusion matrix (FAKE, TRUE):")
    print(
        confusion_matrix(
            y_test,
            predictions,
            labels=["FAKE", "TRUE"],
        )
    )

    if os.path.isfile(EXTERNAL_PATH):
        try:
            external = load(EXTERNAL_PATH)
            training_texts = set(data["text"].map(normalize_text))
            external = external[
                ~external["text"].map(normalize_text).isin(training_texts)
            ].copy()

            if external.empty:
                print("\nExternal evaluation skipped: no independent records remain.")
            else:
                external_predictions = evaluation_pipeline.predict(external["body"])

                print("\nExternal evaluation")
                print(f"Independent records: {len(external)}")
                print(
                    f"Accuracy: "
                    f"{accuracy_score(external['label'], external_predictions) * 100:.2f}%"
                )
                print(
                    f"Macro F1: "
                    f"{f1_score(external['label'], external_predictions, average='macro') * 100:.2f}%"
                )
                print(
                    classification_report(
                        external["label"],
                        external_predictions,
                        labels=["FAKE", "TRUE"],
                        target_names=["FAKE", "TRUE"],
                        zero_division=0,
                    )
                )

        except (ValueError, KeyError, pd.errors.ParserError) as error:
            print(f"\nExternal evaluation skipped: {error}")

    print("\nTraining final deployment model on all usable training records...")

    final_pipeline = make_pipeline(candidates[best])
    final_pipeline.fit(data["body"], data["label"])

    vectorizer = final_pipeline.named_steps["vectorizer"]
    model = final_pipeline.named_steps["classifier"]

    vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")
    model_path = os.path.join(BASE_DIR, "finalized_model.pkl")

    with open(vectorizer_path, "wb") as file:
        pickle.dump(vectorizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    with open(model_path, "wb") as file:
        pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved model: {model_path}")
    print(f"Saved vectorizer: {vectorizer_path}")
    print("Training completed successfully.")


if __name__ == "__main__":
    main()
