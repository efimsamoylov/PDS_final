from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split


def train_tfidf_logreg(
    df: pd.DataFrame,
    task_name: str,
    random_state: int = 42,
    test_size: float = 0.2
) -> Tuple[TfidfVectorizer, LogisticRegression]:
    # Expected columns: text,label
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"[{task_name}] DataFrame must contain columns ['text','label']. "
            f"Got: {df.columns.tolist()}"
        )

    X = df["text"]
    y = df["label"]

    stratify = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify
    )

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        preprocessor=None
    )
    X_train_vec = vec.fit_transform(X_train)
    X_val_vec = vec.transform(X_val)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_val_vec)
    macro_f1 = f1_score(y_val, y_pred, average="macro")

    print(f"\n===== {task_name}: Validation on CSV (train/val split) =====")
    print(f"macro-F1: {macro_f1:.4f}")
    print(classification_report(y_val, y_pred, digits=4))

    return vec, clf
