import json
import re
from pathlib import Path
from typing import Any, Dict, List


def normalize_text(text: str) -> str:
    """Normalize text: lowercase and collapse whitespace."""
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_lexicon(path: Path) -> Dict[str, List[str]]:
    """Load lexicon from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_job_text(job: Dict[str, Any]) -> str:
    """Build text representation from job dict."""
    pos = str(job.get("position", "")).strip()
    org = str(job.get("organization", "")).strip()
    li = str(job.get("linkedin", "")).strip()

    parts = [f"Position: {pos}.", f"Organization: {org}."]
    if li:
        parts.append(f"LinkedIn: {li}.")
    return " ".join(parts)
