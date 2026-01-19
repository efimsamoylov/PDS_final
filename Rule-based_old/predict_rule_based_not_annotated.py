import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from department_rule_based import load_department_lexicon, predict_department_rule
from seniority_rule_based import predict_seniority_rule

# переиспользуем твою логику выбора current job, чтобы не дублировать правила
from run_rule_based_baseline import select_current_job

# Resolve paths relative to the project root (works even if script is run from Rule-Based/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_job_text(job: Dict[str, Any]) -> str:
    pos = str(job.get("position", "")).strip()
    org = str(job.get("organization", "")).strip()
    li = str(job.get("linkedin", "")).strip()

    parts = [f"Position: {pos}.", f"Organization: {org}."]
    if li:
        parts.append(f"LinkedIn: {li}.")
    return " ".join(parts)


def dept_confidence_from_debug(dept_dbg: Dict[str, Any]) -> Dict[str, float]:
    """
    Heuristic confidence for rule-based department:
    - best_score: top similarity/score
    - second_best_score: second highest
    - margin: best - second_best
    - confidence: bounded mapping to [0..1]
    """
    scores = (dept_dbg or {}).get("scores", {}) or {}
    best_score = float((dept_dbg or {}).get("best_score") or 0.0)

    vals = sorted([float(v) for v in scores.values()], reverse=True)
    second_best = float(vals[1]) if len(vals) > 1 else 0.0
    margin = best_score - second_best

    # простая нормализация: чем больше score, тем ближе к 1
    conf = min(1.0, max(0.0, best_score / (best_score + 2.0)))

    return {
        "dept_best_score": best_score,
        "dept_second_best_score": second_best,
        "dept_margin": margin,
        "dept_confidence": conf,
    }


def sen_confidence_from_debug(sen_dbg: Dict[str, Any]) -> float:
    """
    Heuristic confidence for rule-based seniority:
    - 0.9 если сработал regex/паттерн
    - 0.4 если сработал fallback
    """
    if (sen_dbg or {}).get("matched_pattern"):
        return 0.9
    return 0.4


def main():
    parser = argparse.ArgumentParser(description="Rule-based inference for NOT annotated LinkedIn CVs.")
    parser.add_argument("--input", default=str(PROJECT_ROOT / "data" / "linkedin-cvs-not-annotated.json"))
    parser.add_argument("--lexicon", default=str(PROJECT_ROOT / "data" / "department_lexicon.json"))
    parser.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "predictions_rule_not_annotated.csv"))
    args = parser.parse_args()

    input_path = Path(args.input)
    lexicon_path = Path(args.lexicon)
    out_path = Path(args.out)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input JSON not found: {input_path}\n"
            f"Tip: check your project structure. Expected under: {PROJECT_ROOT / 'data'}"
        )
    if not lexicon_path.exists():
        raise FileNotFoundError(
            f"Department lexicon not found: {lexicon_path}\n"
            f"Tip: expected file: {PROJECT_ROOT / 'data' / 'department_lexicon.json'}"
        )

    with open(input_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)  # list[list[dict]]

    lexicon = load_department_lexicon(str(lexicon_path))

    rows: List[Dict[str, Any]] = []

    for i, profile_jobs in enumerate(profiles):
        if not isinstance(profile_jobs, list) or not profile_jobs:
            continue

        job = select_current_job(profile_jobs)
        if not job:
            continue

        text = build_job_text(job)

        dept_pred, dept_dbg = predict_department_rule(text, lexicon)
        sen_pred, sen_dbg = predict_seniority_rule(text)

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

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Loaded: {input_path}")
    print(f"Saved:  {out_path}")
    print(f"Rows:   {len(df)}")


if __name__ == "__main__":
    main()