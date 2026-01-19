import pandas as pd

from config import (
    DEPT_CSV_PATH, SEN_CSV_PATH, JSON_NOT_ANNOTATED_PATH, PRED_OUT_PATH,
    RANDOM_STATE, TEST_SIZE
)
from data_loader import load_labeled_csv, load_not_annotated_profiles
from text_processing import normalize_text
from model import train_tfidf_logreg
from inference import predict_on_not_annotated


def prepare_train_df(csv_path: str) -> pd.DataFrame:
    df = load_labeled_csv(csv_path).copy()
    df["text"] = df["text"].fillna("").astype(str).map(normalize_text)
    df["label"] = df["label"].fillna("").astype(str)
    return df


def main():
    # 1) Train on CSVs (formal evaluation ONLY on CSV split)
    dept_df = prepare_train_df(DEPT_CSV_PATH)
    sen_df = prepare_train_df(SEN_CSV_PATH)

    dept_vec, dept_clf = train_tfidf_logreg(
        dept_df, task_name="Department", random_state=RANDOM_STATE, test_size=TEST_SIZE
    )
    sen_vec, sen_clf = train_tfidf_logreg(
        sen_df, task_name="Seniority", random_state=RANDOM_STATE, test_size=TEST_SIZE
    )

    # 2) Inference on not-annotated JSON (no ground truth)
    print("\n===== Inference on not-annotated JSON (no ground truth) =====")
    profiles = load_not_annotated_profiles(JSON_NOT_ANNOTATED_PATH)
    pred_df = predict_on_not_annotated(
        profiles,
        dept_vec, dept_clf,
        sen_vec, sen_clf
    )

    pred_df.to_csv(PRED_OUT_PATH, index=False)
    print(f"Saved predictions to: {PRED_OUT_PATH}")
    print("\nSample predictions:")
    print(pred_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
