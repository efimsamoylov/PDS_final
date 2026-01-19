import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from current_job import select_current_job
from text_processing import build_job_text
from model import predict_department_rule, predict_seniority_rule


def load_profiles(json_path: Path) -> List[List[Dict[str, Any]]]:
    """Load profiles from JSON file."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("profiles", "data", "items"):
            if isinstance(data.get(k), list):
                return data[k]
        return [data]
    return []


def dept_confidence_from_debug(dept_dbg: Dict[str, Any]) -> Dict[str, float]:
    """Calculate department confidence metrics from debug info."""
    scores = (dept_dbg or {}).get("scores", {}) or {}
    best_score = float((dept_dbg or {}).get("best_score") or 0.0)

    vals = sorted([float(v) for v in scores.values()], reverse=True)
    second_best = float(vals[1]) if len(vals) > 1 else 0.0
    margin = best_score - second_best

    # Simple normalization: higher score -> closer to 1
    conf = min(1.0, max(0.0, best_score / (best_score + 2.0)))

    return {
        "dept_best_score": best_score,
        "dept_second_best_score": second_best,
        "dept_margin": margin,
        "dept_confidence": conf,
    }


def sen_confidence_from_debug(sen_dbg: Dict[str, Any]) -> float:
    """Calculate seniority confidence from debug info."""
    matched_terms = (sen_dbg or {}).get("matched_terms", [])
    if matched_terms:
        return 0.9
    return 0.4


def predict_on_not_annotated(
    profiles: List[List[Dict[str, Any]]],
    dept_lexicon: Dict[str, List[str]],
    sen_lexicon: Dict[str, List[str]],
    dept_params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Predict department and seniority for not-annotated profiles.

    Args:
        profiles: List of profile job histories
        dept_lexicon: Department lexicon
        sen_lexicon: Seniority lexicon
        dept_params: Department prediction parameters

    Returns:
        DataFrame with predictions
    """
    rows: List[Dict[str, Any]] = []

    for i, profile_jobs in enumerate(profiles):
        if not isinstance(profile_jobs, list) or not profile_jobs:
            continue

        job = select_current_job(profile_jobs)
        if not job:
            continue

        text = build_job_text(job)

        # Filter parameters to only include those accepted by predict_department_rule
        dept_predict_args = {k: v for k, v in dept_params.items() if k != "sen_default"}
        sen_default = dept_params.get("sen_default", "Professional")

        dept_pred, dept_dbg = predict_department_rule(text, dept_lexicon, **dept_predict_args)
        sen_pred, sen_dbg = predict_seniority_rule(text, sen_lexicon, default_label=sen_default)

        dept_conf = dept_confidence_from_debug(dept_dbg)
        sen_conf = sen_confidence_from_debug(sen_dbg)

        rows.append({
            "profile_idx": i,
            "organization": job.get("organization"),
            "position": job.get("position"),
            "startDate": job.get("startDate"),
            "endDate": job.get("endDate"),
            "status": job.get("status"),
            "linkedin": job.get("linkedin"),
            "dept_pred": dept_pred,
            "sen_pred": sen_pred,
            **dept_conf,
            "sen_confidence": sen_conf,
        })

    return pd.DataFrame(rows)
