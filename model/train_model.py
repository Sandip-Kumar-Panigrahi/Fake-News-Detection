import os
import pickle
import sys
import tempfile

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
PROJECT_DIR = os.path.abspath(os.path.dirname(BASE_DIR))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from model.hybrid_model import WeightedHybridClassifier
from model.text_preprocessing import preprocess_text

MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, "artifacts"))


def _atomic_pickle_dump(obj, final_path: str) -> None:
    """
    Write pickle atomically: temp file in same dir, fsync, then replace.
    Avoids corrupt .pkl if the process is interrupted mid-write.
    """
    final_path = os.path.abspath(final_path)
    directory = os.path.dirname(final_path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".pkl", dir=directory)
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
    except BaseException:
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except OSError:
            pass
        raise


FAKE_CSV = os.path.abspath(os.path.join(PROJECT_DIR, "Fake.csv"))
TRUE_CSV = os.path.abspath(os.path.join(PROJECT_DIR, "True.csv"))
DATA_CSV = os.path.abspath(os.path.join(PROJECT_DIR, "data.csv"))

MODEL_CHOICE = "hybrid"  # "hybrid", "auto", "logreg", "nb"


def _merge_title_and_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    If dataset has title + text, combine both so short headline learning improves.
    """
    if "title" in df.columns and "text" in df.columns:
        title_part = df["title"].fillna("").astype(str).str.strip()
        text_part = df["text"].fillna("").astype(str).str.strip()
        df["text"] = (title_part + " " + text_part).str.strip()
    elif "text" in df.columns:
        df["text"] = df["text"].fillna("").astype(str).str.strip()
    return df


def _augment_short_samples(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add extra copies of short samples so model doesn't ignore headline-type inputs.
    """
    token_counts = df["clean_text"].str.split().str.len()
    short_df = df[token_counts <= 8]

    if len(short_df) == 0:
        return df

    augmented = pd.concat([df, short_df], ignore_index=True)
    print("Added short-text augmented rows:", len(short_df))
    return augmented


def _load_from_data_csv(bad_lines_option: str) -> pd.DataFrame:
    """Load combined dataset from data.csv (Headline + Body, Label 0=fake 1=real)."""
    raw = pd.read_csv(DATA_CSV, engine="python", on_bad_lines=bad_lines_option)
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    if "label" not in raw.columns:
        raise ValueError("data.csv must include a Label column (0=fake, 1=real).")

    headline = raw["headline"].fillna("").astype(str).str.strip() if "headline" in raw.columns else ""
    body = raw["body"].fillna("").astype(str).str.strip() if "body" in raw.columns else ""
    if isinstance(headline, str):
        raw["text"] = body
    else:
        raw["text"] = (headline + " " + body).str.strip()

    raw["label"] = pd.to_numeric(raw["label"], errors="coerce")
    combined = raw[["text", "label"]].dropna()
    combined["label"] = combined["label"].astype(int)
    combined = combined[combined["label"].isin([0, 1])]
    print("Loaded dataset from data.csv")
    return combined


def load_dataset():
    bad_lines_option = os.environ.get("BAD_LINES_OPTION", "skip")

    if os.path.isfile(FAKE_CSV) and os.path.isfile(TRUE_CSV):
        fake_data = pd.read_csv(FAKE_CSV, engine="python", on_bad_lines=bad_lines_option)
        true_data = pd.read_csv(TRUE_CSV, engine="python", on_bad_lines=bad_lines_option)

        fake_data = _merge_title_and_text(fake_data)
        true_data = _merge_title_and_text(true_data)

        fake_data["label"] = 0
        true_data["label"] = 1

        combined = pd.concat([fake_data, true_data], ignore_index=True)
        combined = combined[["text", "label"]].dropna()
        print("Loaded dataset from Fake.csv + True.csv")
    elif os.path.isfile(DATA_CSV):
        combined = _load_from_data_csv(bad_lines_option)
    else:
        raise FileNotFoundError(
            "No training data found. Add Fake.csv + True.csv, or data.csv in the project folder."
        )

    combined["text"] = combined["text"].astype(str).str.strip()
    combined = combined[combined["text"].str.len() > 0]

    combined["clean_text"] = combined["text"].apply(preprocess_text)
    combined = combined[combined["clean_text"].str.len() > 0]
    combined = _augment_short_samples(combined)

    print("Loaded dataset rows:", len(combined))
    print("Label counts:\n", combined["label"].value_counts())
    return combined


def _build_hybrid():
    nb_model = MultinomialNB(alpha=0.45)
    lr_model = LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", C=2.0)
    return WeightedHybridClassifier(nb_model=nb_model, lr_model=lr_model, nb_weight=0.42, lr_weight=0.58)


def _get_models():
    return {
        "logreg": LogisticRegression(max_iter=3000, class_weight="balanced", solver="liblinear", C=1.5),
        "nb": MultinomialNB(alpha=0.5),
        "hybrid": _build_hybrid(),
    }


def train_and_save():
    data = load_dataset()

    y = data["label"]
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        data["clean_text"], y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer_candidates = [
        {"ngram_range": (1, 2), "min_df": 1, "max_features": 50000},
        {"ngram_range": (1, 3), "min_df": 1, "max_features": 80000},
        {"ngram_range": (1, 3), "min_df": 2, "max_features": 100000},
    ]

    best_bundle = None
    best_score = -1.0
    model_names = ["logreg", "nb", "hybrid"] if MODEL_CHOICE in {"auto", "hybrid"} else [MODEL_CHOICE]

    for vec_params in vectorizer_candidates:
        vectorizer = TfidfVectorizer(
            ngram_range=vec_params["ngram_range"],
            min_df=vec_params["min_df"],
            max_df=0.95,
            sublinear_tf=True,
            max_features=vec_params["max_features"],
        )
        X_train = vectorizer.fit_transform(X_train_text)
        X_test = vectorizer.transform(X_test_text)

        for model_name in model_names:
            candidate = _get_models()[model_name]
            candidate.fit(X_train, y_train)
            val_preds = candidate.predict(X_test)
            score = f1_score(y_test, val_preds, zero_division=0)
            print(f"F1 ({model_name}, ngrams={vec_params['ngram_range']}, min_df={vec_params['min_df']}, max_features={vec_params['max_features']}): {score:.4f}")

            if score > best_score:
                best_score = score
                best_bundle = {
                    "model": candidate,
                    "vectorizer": vectorizer,
                    "model_name": model_name,
                    "vec_params": vec_params,
                    "preds": val_preds,
                }

    if not best_bundle:
        raise RuntimeError("Could not train any candidate model.")

    model = best_bundle["model"]
    vectorizer = best_bundle["vectorizer"]
    predictions = best_bundle["preds"]
    print("Selected model:", best_bundle["model_name"])
    print("Selected vectorizer params:", best_bundle["vec_params"])

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)

    model_path = os.path.join(MODEL_DIR, "fake_news_model.pkl")
    vectorizer_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

    print("[train_model] Saving model and vectorizer (atomic write)...", flush=True)
    _atomic_pickle_dump(model, model_path)
    print(f"[train_model] Model file written OK: {model_path}", flush=True)
    _atomic_pickle_dump(vectorizer, vectorizer_path)
    print(f"[train_model] Vectorizer file written OK: {vectorizer_path}", flush=True)

    print("Training completed")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Model saved at: {os.path.abspath(model_path)}")
    print(f"Vectorizer saved at: {os.path.abspath(vectorizer_path)}")
    print("[train_model] All artifacts saved successfully.", flush=True)


if __name__ == "__main__":
    try:
        train_and_save()
    except KeyboardInterrupt:
        print("[train_model] Training stopped by user; previous .pkl files (if any) were left unchanged.", flush=True)
        sys.exit(130)
