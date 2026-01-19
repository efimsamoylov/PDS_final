import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import (
    ANNOTATED_JSON_PATH,
    NOT_ANNOTATED_JSON_PATH,
    DEPT_LEXICON_PATH,
    SEN_LEXICON_PATH,
    PRED_ANNOTATED_PATH,
    PRED_NOT_ANNOTATED_PATH,
    DEPT_BIGRAM_WEIGHT,
    DEPT_UNIGRAM_WEIGHT,
    DEPT_MIN_SCORE,
    DEPT_DEFAULT_LABEL,
    SEN_DEFAULT_LABEL,
)
from text_processing import load_lexicon, build_job_text
from current_job import select_current_job
from model import predict_department_rule, predict_seniority_rule
from inference import load_profiles, predict_on_not_annotated


def evaluate_on_annotated(
    profiles_path,
    dept_lexicon,
    sen_lexicon,
    dept_params,
    output_path
):
    """
    Evaluate rule-based model on annotated data.

    Args:
        profiles_path: Path to annotated JSON
        dept_lexicon: Department lexicon
        sen_lexicon: Seniority lexicon
        dept_params: Department prediction parameters
        output_path: Path to save predictions CSV

    Returns:
        DataFrame with predictions and ground truth
    """
    profiles = load_profiles(profiles_path)

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

        # Separate department parameters from seniority default
        dept_predict_args = {k: v for k, v in dept_params.items() if k != "sen_default"}
        sen_default = dept_params.get("sen_default", "Professional")

        dept_pred, dept_dbg = predict_department_rule(text, dept_lexicon, **dept_predict_args)
        sen_pred, sen_dbg = predict_seniority_rule(text, sen_lexicon, default_label=sen_default)

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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    return df, y_true_dept, y_pred_dept, y_true_sen, y_pred_sen


def print_metrics(y_true, y_pred, task_name):
    """Print classification metrics."""
    if not y_true:
        print(f"\n{task_name}: No ground truth available")
        return

    print(f"\n=== {task_name} ===")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    labels = sorted(set(y_true) | set(y_pred))
    print(f"Labels: {labels}")
    print(f"\nConfusion matrix:\n{confusion_matrix(y_true, y_pred, labels=labels)}")
    print(f"\nClassification report:\n{classification_report(y_true, y_pred, labels=labels, zero_division=0)}")


def main():
    """Main pipeline execution."""
    print("=" * 60)
    print("Rule-Based Classification Pipeline")
    print("=" * 60)

    # Load lexicons
    print("\n[1/4] Loading lexicons...")
    dept_lexicon = load_lexicon(DEPT_LEXICON_PATH)
    sen_lexicon = load_lexicon(SEN_LEXICON_PATH)
    print(f"  Department lexicon: {len(dept_lexicon)} classes")
    print(f"  Seniority lexicon: {len(sen_lexicon)} classes")

    # Department prediction parameters
    dept_params = {
        "bigram_weight": DEPT_BIGRAM_WEIGHT,
        "unigram_weight": DEPT_UNIGRAM_WEIGHT,
        "min_score": DEPT_MIN_SCORE,
        "default_label": DEPT_DEFAULT_LABEL,
        "sen_default": SEN_DEFAULT_LABEL,
    }

    # Evaluate on annotated data
    print("\n[2/4] Evaluating on annotated data...")
    df_annotated, y_true_dept, y_pred_dept, y_true_sen, y_pred_sen = evaluate_on_annotated(
        ANNOTATED_JSON_PATH,
        dept_lexicon,
        sen_lexicon,
        dept_params,
        PRED_ANNOTATED_PATH
    )
    print(f"  Saved: {PRED_ANNOTATED_PATH}")
    print(f"  Rows: {len(df_annotated)}")

    # Print metrics
    print("\n[3/4] Evaluation metrics:")
    print_metrics(y_true_dept, y_pred_dept, "Department")
    print_metrics(y_true_sen, y_pred_sen, "Seniority")

    # Predict on not-annotated data
    print("\n[4/4] Predicting on not-annotated data...")
    profiles_not_annotated = load_profiles(NOT_ANNOTATED_JSON_PATH)
    df_not_annotated = predict_on_not_annotated(
        profiles_not_annotated,
        dept_lexicon,
        sen_lexicon,
        dept_params
    )
    df_not_annotated.to_csv(PRED_NOT_ANNOTATED_PATH, index=False)
    print(f"  Saved: {PRED_NOT_ANNOTATED_PATH}")
    print(f"  Rows: {len(df_not_annotated)}")

    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)

    print("\nSample predictions (not-annotated):")
    print(df_not_annotated.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
