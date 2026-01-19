# Simple interpretable baseline

# Task 6 (Baseline): TF-IDF + Logistic Regression for Department + Seniority
# - Train ONLY on CSVs: department-v2.csv, seniority-v2.csv
# - Use linkedin-cvs-not-annotated.json only for inference/sanity-check and prediction export
# - Do NOT use linkedin-cvs-annotated.json here (keep it for final exam run)

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# -----------------------------
# Paths (edit if needed)
# -----------------------------
DEPT_CSV_PATH = "/Users/efim/Desktop/department-v2.csv"
SEN_CSV_PATH = '/Users/efim/Desktop/seniority-v2.csv'
JSON_NOT_ANNOTATED_PATH = '/Users/efim/Desktop/linkedin-cvs-not-annotated.json'

PRED_OUT_PATH = '/Users/efim/Desktop/Results/predictions_not_annotated.csv'

RANDOM_STATE = 42
TEST_SIZE = 0.2

# -----------------------------
# Text normalization (shared)
# -----------------------------
_PUNCT_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+", flags=re.UNICODE)


def normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).lower()

    # unify separators
    s = s.replace("&", " ").replace("/", " ").replace("|", " ")

    # strip diacritics
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))

    # remove punctuation, collapse whitespace
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# -----------------------------
# Train helpers
# -----------------------------
def train_tfidf_logreg(csv_path: str, task_name: str) -> Tuple[TfidfVectorizer, LogisticRegression]:
    df = pd.read_csv(csv_path)

    # Expected columns: text,label
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"[{task_name}] CSV must contain columns ['text','label']. "
            f"Got: {df.columns.tolist()}"
        )

    df = df.copy()
    df["text"] = df["text"].fillna("").astype(str).map(normalize_text)
    df["label"] = df["label"].fillna("").astype(str)

    X = df["text"]
    y = df["label"]

    # stratify if feasible
    stratify = y if (y.nunique() > 1 and y.value_counts().min() >= 2) else None
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify
    )

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        preprocessor=None  # already normalized
    )
    X_train_vec = vec.fit_transform(X_train)
    X_val_vec = vec.transform(X_val)

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=-1
    )
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_val_vec)
    macro_f1 = f1_score(y_val, y_pred, average="macro")

    print(f"\n===== {task_name}: Validation on CSV (train/val split) =====")
    print(f"macro-F1: {macro_f1:.4f}")
    print(classification_report(y_val, y_pred, digits=4))

    return vec, clf


# -----------------------------
# CV JSON parsing (not-annotated)
# -----------------------------
def _parse_yyyy_mm(s: Optional[str]) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    try:
        if re.fullmatch(r"\d{4}-\d{2}", s):
            return datetime.strptime(s, "%Y-%m")
        if re.fullmatch(r"\d{4}", s):
            return datetime.strptime(s, "%Y")
    except Exception:
        return None
    return None


def _is_active(exp: Dict[str, Any]) -> bool:
    st = (exp.get("status") or "").strip().upper()
    if st == "ACTIVE":
        return True
    # heuristic: no endDate often means current
    return exp.get("endDate") in (None, "", "null")


def select_current_job(experiences: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Selection rules:
    1) If there is ACTIVE: pick the one with latest startDate (parsable), else first ACTIVE.
    2) If no ACTIVE: pick the one with latest startDate (parsable), else first entry.
    """
    if not experiences:
        return None

    exps = [e for e in experiences if isinstance(e, dict)]
    if not exps:
        return None

    actives = [e for e in exps if _is_active(e)]
    pool = actives if actives else exps

    def key_fn(e: Dict[str, Any]) -> Tuple[int, datetime]:
        d = _parse_yyyy_mm(e.get("startDate"))
        return (1 if d else 0, d or datetime.min)

    pool_sorted = sorted(pool, key=key_fn, reverse=True)
    return pool_sorted[0] if pool_sorted else None


def build_cv_text_from_profile(profile: Any) -> Tuple[str, str]:
    """
    Returns (profile_id, text) where text is built from current job: position + organization.
    Handles profile formats:
      - dict with "experiences"/"positions"/"experience"
      - list of experience dicts
    """
    pid = None
    experiences = []

    if isinstance(profile, dict):
        pid = profile.get("profile_id", profile.get("id"))
        if isinstance(profile.get("experiences"), list):
            experiences = profile["experiences"]
        elif isinstance(profile.get("positions"), list):
            experiences = profile["positions"]
        elif isinstance(profile.get("experience"), list):
            experiences = profile["experience"]
        elif isinstance(profile.get("items"), list):
            experiences = profile["items"]
        else:
            # if dict itself looks like an experience
            if any(k in profile for k in ("position", "organization", "startDate", "endDate", "status")):
                experiences = [profile]
    elif isinstance(profile, list):
        experiences = profile

    current = select_current_job([e for e in experiences if isinstance(e, dict)])
    pos = (current or {}).get("position", "")
    org = (current or {}).get("organization", "")
    text = normalize_text(f"{pos} {org}")

    return (str(pid) if pid is not None else "", text)


def load_not_annotated_profiles(json_path: str) -> List[Any]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # sometimes wrapped
        for k in ("profiles", "data", "items"):
            if isinstance(data.get(k), list):
                return data[k]
        return [data]
    return []


# -----------------------------
# Inference on not-annotated
# -----------------------------
def predict_on_not_annotated(
        json_path: str,
        dept_vec: TfidfVectorizer,
        dept_clf: LogisticRegression,
        sen_vec: TfidfVectorizer,
        sen_clf: LogisticRegression,
        out_csv: str
) -> pd.DataFrame:
    profiles = load_not_annotated_profiles(json_path)

    rows = []
    for idx, profile in enumerate(profiles):
        pid, text = build_cv_text_from_profile(profile)
        if not pid:
            pid = str(idx)

        # Department
        Xd = dept_vec.transform([text])
        dept_pred = dept_clf.predict(Xd)[0]
        dept_conf = float(np.max(dept_clf.predict_proba(Xd)[0]))

        # Seniority
        Xs = sen_vec.transform([text])
        sen_pred = sen_clf.predict(Xs)[0]
        sen_conf = float(np.max(sen_clf.predict_proba(Xs)[0]))

        rows.append({
            "profile_id": pid,
            "input_text": text,
            "department_pred": dept_pred,
            "department_conf": dept_conf,
            "seniority_pred": sen_pred,
            "seniority_conf": sen_conf,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out_csv, index=False)
    return df_out


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    # 1) Train on CSVs (formal evaluation is ONLY on CSV split)
    dept_vec, dept_clf = train_tfidf_logreg(DEPT_CSV_PATH, task_name="Department")
    sen_vec, sen_clf = train_tfidf_logreg(SEN_CSV_PATH, task_name="Seniority")

    # 2) Inference on not-annotated JSON (no metrics; just sanity-check + export)
    print("\n===== Inference on not-annotated JSON (no ground truth) =====")
    pred_df = predict_on_not_annotated(
        JSON_NOT_ANNOTATED_PATH,
        dept_vec, dept_clf,
        sen_vec, sen_clf,
        out_csv=PRED_OUT_PATH
    )
    print(f"Saved predictions to: {PRED_OUT_PATH}")
    print("\nSample predictions:")
    print(pred_df.head(10).to_string(index=False))
