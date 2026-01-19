import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from department_rule_based import load_department_lexicon, predict_department_rule
from seniority_rule_based import predict_seniority_rule

# Script lives in Rule-Based/. Resolve paths relative to project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_yyyy_mm(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, "%Y-%m")
    except ValueError:
        return None


def is_active(job: Dict[str, Any]) -> bool:
    return str(job.get("status", "")).upper() == "ACTIVE"


def select_current_job(profile_jobs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    active = [j for j in profile_jobs if is_active(j)]
    if active:
        # sort by startDate desc, then by "role strength" heuristic
        def role_strength(j: Dict[str, Any]) -> int:
            p = str(j.get("position", "")).lower()
            # higher = stronger
            if re.search(r"\b(ceo|cfo|cto|cio|coo|chief|founder|owner)\b", p): return 5
            if re.search(r"\b(vp|vice president|director|head|geschäftsführer|geschäftsführung|vorstand|prokurist)\b", p): return 4
            if re.search(r"\b(manager|leitung|leiter|teamleiter)\b", p): return 3
            if re.search(r"\b(lead|principal|staff)\b", p): return 2
            return 1

        def sort_key(j: Dict[str, Any]):
            dt = parse_yyyy_mm(j.get("startDate"))
            # None dates go last
            dt_ord = dt.toordinal() if dt else -10**9
            return (dt_ord, role_strength(j))

        return sorted(active, key=sort_key, reverse=True)[0]

    # no ACTIVE -> fallback to most recent INACTIVE by endDate then startDate
    inactive = [j for j in profile_jobs if str(j.get("status", "")).upper() == "INACTIVE"]
    if not inactive:
        return None

    def sort_key_inactive(j: Dict[str, Any]):
        end_dt = parse_yyyy_mm(j.get("endDate"))
        start_dt = parse_yyyy_mm(j.get("startDate"))
        end_ord = end_dt.toordinal() if end_dt else -10**9
        start_ord = start_dt.toordinal() if start_dt else -10**9
        return (end_ord, start_ord)

    return sorted(inactive, key=sort_key_inactive, reverse=True)[0]


def build_job_text(job: Dict[str, Any]) -> str:
    pos = str(job.get("position", "")).strip()
    org = str(job.get("organization", "")).strip()
    return f"Position: {pos}. Organization: {org}."


def main():
    INPUT_JSON = PROJECT_ROOT / "data" / "linkedin-cvs-annotated.json"
    LEXICON_JSON = PROJECT_ROOT / "data" / "department_lexicon.json"
    OUT_CSV = PROJECT_ROOT / "artifacts" / "predictions_rule.csv"

    if not INPUT_JSON.exists():
        raise FileNotFoundError(f"Annotated JSON not found: {INPUT_JSON}")
    if not LEXICON_JSON.exists():
        raise FileNotFoundError(f"Department lexicon not found: {LEXICON_JSON}")

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        profiles = json.load(f)  # expected: list[list[dict]]

    lexicon = load_department_lexicon(str(LEXICON_JSON))

    rows = []
    y_true_dept, y_pred_dept = [], []
    y_true_sen, y_pred_sen = [], []

    for i, profile_jobs in enumerate(profiles):
        if not isinstance(profile_jobs, list) or not profile_jobs:
            continue

        job = select_current_job(profile_jobs)
        if not job:
            continue

        text = build_job_text(job)

        dept_pred, dept_dbg = predict_department_rule(text, lexicon)
        sen_pred, sen_dbg = predict_seniority_rule(text)

        dept_true = job.get("department")
        sen_true = job.get("seniority")

        rows.append({
            "profile_idx": i,
            "organization": job.get("organization"),
            "position": job.get("position"),
            "startDate": job.get("startDate"),
            "status": job.get("status"),
            "dept_true": dept_true,
            "dept_pred": dept_pred,
            "sen_true": sen_true,
            "sen_pred": sen_pred,
            "dept_best_score": dept_dbg.get("best_score"),
        })

        if dept_true is not None:
            y_true_dept.append(str(dept_true))
            y_pred_dept.append(str(dept_pred))
        if sen_true is not None:
            y_true_sen.append(str(sen_true))
            y_pred_sen.append(str(sen_pred))

    df = pd.DataFrame(rows)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved predictions: {OUT_CSV}")
    print()

    # Department metrics
    if y_true_dept:
        print("=== Department ===")
        print("Accuracy:", accuracy_score(y_true_dept, y_pred_dept))
        labels = sorted(set(y_true_dept) | set(y_pred_dept))
        print("Labels:", labels)
        print("Confusion matrix:\n", confusion_matrix(y_true_dept, y_pred_dept, labels=labels))
        print("\nClassification report:\n", classification_report(y_true_dept, y_pred_dept, labels=labels, zero_division=0))
        print()

    # Seniority metrics
    if y_true_sen:
        print("=== Seniority ===")
        print("Accuracy:", accuracy_score(y_true_sen, y_pred_sen))
        labels = sorted(set(y_true_sen) | set(y_pred_sen))
        print("Labels:", labels)
        print("Confusion matrix:\n", confusion_matrix(y_true_sen, y_pred_sen, labels=labels))
        print("\nClassification report:\n", classification_report(y_true_sen, y_pred_sen, labels=labels, zero_division=0))
        print()


if __name__ == "__main__":
    main()