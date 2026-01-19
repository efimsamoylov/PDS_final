from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Data & outputs folders
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "artifacts"

# Input data
ANNOTATED_JSON_PATH = DATA_DIR / "linkedin-cvs-annotated.json"
NOT_ANNOTATED_JSON_PATH = DATA_DIR / "linkedin-cvs-not-annotated.json"

# Lexicons
DEPT_LEXICON_PATH = DATA_DIR / "department_lexicon.json"
SEN_LEXICON_PATH = DATA_DIR / "seniority_lexicon.json"

# Outputs
PRED_ANNOTATED_PATH = OUT_DIR / "predictions_rule.csv"
PRED_NOT_ANNOTATED_PATH = OUT_DIR / "predictions_rule_not_annotated.csv"

# Rule-based model parameters
DEPT_BIGRAM_WEIGHT = 2.0
DEPT_UNIGRAM_WEIGHT = 1.0
DEPT_MIN_SCORE = 2.0
DEPT_DEFAULT_LABEL = "Other"

SEN_DEFAULT_LABEL = "Professional"
