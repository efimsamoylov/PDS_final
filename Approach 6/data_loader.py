import json
from pathlib import Path
from typing import Any, List

import pandas as pd


def load_labeled_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"CSV must contain columns ['text','label']. Got: {df.columns.tolist()}"
        )
    return df


def load_not_annotated_profiles(json_path: str) -> List[Any]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for k in ("profiles", "data", "items"):
            if isinstance(data.get(k), list):
                return data[k]
        return [data]
    return []
