import re
from typing import Tuple, Dict, Any

PATTERNS = [
    ("C-Level", r"\b(ceo|cfo|cto|cio|coo|chief|founder|co-founder|owner|inhaber|geschäftsführer|geschäftsführung|vorstand|partner|president)\b"),
    ("Director", r"\b(vp|vice president|director|head of|bereichsleiter|direktor|directeur|directrice|svp|evp)\b"),
    # Переносим Lead и Senior ВЫШЕ Manager, чтобы "Senior Manager" определялся как Senior или Lead
    ("Lead", r"\b(lead|team lead|principal|staff|leitung|teamleitung|teamleiter|lead manager)\b"),
    ("Senior", r"\b(senior|sr\.?|expert)\b"),
    ("Manager", r"\b(manager|gerente|responsable)\b"),
    ("Junior", r"\b(junior|jr\.?|associate|entry|beginner)\b"),
    ("Intern", r"\b(intern|internship|trainee|praktikant|praktikum|working student|werkstudent|étudiant)\b"),
]

def predict_seniority_rule(text: str, default_label: str = "Professional") -> Tuple[str, Dict[str, Any]]:
    t = str(text).lower()
    t = re.sub(r"\s+", " ", t)

    for label, pattern in PATTERNS:
        if re.search(pattern, t):
            return label, {"matched_pattern": pattern}

    return default_label, {"matched_pattern": None}