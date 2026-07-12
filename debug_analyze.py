import os
import re

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FAKE_CSV = os.path.join(BASE_DIR, "Fake.csv")
TRUE_CSV = os.path.join(BASE_DIR, "True.csv")


def preprocess_text(raw_text: str) -> str:
    stemmer = PorterStemmer()
    try:
        stop_words = set(stopwords.words("english"))
    except Exception:
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "to",
            "of",
            "in",
            "on",
            "for",
            "with",
            "is",
            "are",
        }

    text = str(raw_text).lower()
    text = re.sub(r"[^a-zA-Z\\s]", " ", text)

    words = text.split()
    clean_words = []
    for word in words:
        if word not in stop_words and len(word) > 1:
            clean_words.append(stemmer.stem(word))

    return " ".join(clean_words)


def main():
    # Only read a subset so it runs fast on your computer
    fake_df = pd.read_csv(FAKE_CSV, engine="python", on_bad_lines="skip", nrows=8000)
    true_df = pd.read_csv(TRUE_CSV, engine="python", on_bad_lines="skip", nrows=8000)

    print("Fake columns:", fake_df.columns.tolist())
    print("True columns:", true_df.columns.tolist())

    fake_df["label"] = 0
    true_df["label"] = 1
    combined = pd.concat([fake_df, true_df], ignore_index=True)

    print("Sample total rows:", len(combined))
    print("Label counts:\n", combined["label"].value_counts())

    sample = "India wins cricket world cup"
    cleaned_sample = preprocess_text(sample)
    stems = sorted(set(cleaned_sample.split()))

    print("\nSample cleaned:", cleaned_sample)
    print("Keyword stems:", stems)

    def keyword_hits(df):
        hits = {s: 0 for s in stems}
        processed = 0
        for _, row in df.iterrows():
            cleaned = preprocess_text(row["text"])
            if not cleaned:
                continue
            processed += 1
            tokens = set(cleaned.split())
            for s in stems:
                if s in tokens:
                    hits[s] += 1
        return processed, hits

    f_processed, f_hits = keyword_hits(fake_df)
    t_processed, t_hits = keyword_hits(true_df)

    print("\nProcessed rows for keyword check:")
    print("Fake:", f_processed, " / True:", t_processed)
    print("Hits in Fake:", f_hits)
    print("Hits in True:", t_hits)

    fake_wc = fake_df["text"].astype(str).str.split().str.len()
    true_wc = true_df["text"].astype(str).str.split().str.len()

    print("\nWord count stats (original text):")
    print("Fake avg:", round(fake_wc.mean(), 2), " median:", int(fake_wc.median()))
    print("True avg:", round(true_wc.mean(), 2), " median:", int(true_wc.median()))


if __name__ == "__main__":
    main()

