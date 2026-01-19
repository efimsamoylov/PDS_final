from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data & outputs folders
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs"

# Input data
DEPT_CSV_PATH = DATA_DIR / "department-v2.csv"
SEN_CSV_PATH = DATA_DIR / "seniority-v2.csv"
JSON_NOT_ANNOTATED_PATH = DATA_DIR / "linkedin-cvs-not-annotated.json"

# Output
PRED_OUT_PATH = OUT_DIR / "predictions_not_annotated.csv"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2