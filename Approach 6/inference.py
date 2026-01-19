from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from current_job import select_current_job
from text_processing import normalize_text


def build_cv_text_from_profile(profile: Any) -> Tuple[str, str]:
    """
    Returns (profile_id, text) where text is built from current job: position + organization.
    Handles profile formats:
      - dict with "experiences"/"positions"/"experience"/"items"
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
            if any(k in profile for k in ("position", "organization", "startDate", "endDate", "status")):
                experiences = [profile]
    elif isinstance(profile, list):
        experiences = profile

    current = select_current_job([e for e in experiences if isinstance(e, dict)])
    pos = (current or {}).get("position", "")
    org = (current or {}).get("organization", "")
    text = normalize_text(f"{pos} {org}")

    return (str(pid) if pid is not None else "", text)


def predict_on_not_annotated(
    profiles: List[Any],
    dept_vec: TfidfVectorizer,
    dept_clf: LogisticRegression,
    sen_vec: TfidfVectorizer,
    sen_clf: LogisticRegression
) -> pd.DataFrame:
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

    return pd.DataFrame(rows)
