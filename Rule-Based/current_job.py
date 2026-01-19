import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _parse_yyyy_mm(s: Optional[str]) -> Optional[datetime]:
    """Parse date string in YYYY-MM or YYYY format."""
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
    """Check if experience is active."""
    st = (exp.get("status") or "").strip().upper()
    if st == "ACTIVE":
        return True
    # Heuristic: no endDate often means current
    return exp.get("endDate") in (None, "", "null")


def select_current_job(experiences: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Select current job from list of experiences.

    Rules:
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
