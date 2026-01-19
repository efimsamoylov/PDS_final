import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Пути к твоим данным
CSV_PATH = "../data/seniority-v2.csv" # Путь к размеченному CSV для Seniority
OUT_PATH = "../data/seniority_lexicon.json"

# Используем тот же набор шума, что и для департаментов
STOP = {"und", "oder", "of", "and", "the", "for", "with", "in", "at"}

def clean_term(t: str) -> str:
    return re.sub(r"\s+", " ", str(t).strip().lower())

# 1. Загружаем данные
df = pd.read_csv(CSV_PATH)
texts = df["text"].astype(str).tolist()
labels = df["label"].astype(str).tolist()
classes = df["label"].unique()

# 2. TF-IDF (важно: берем униграммы и биграммы)
vec = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    token_pattern=r"(?u)\b[\w\-\+\.]{2,}\b"
)
X = vec.fit_transform(texts)
terms = np.array(vec.get_feature_names_out())

# 3. Считаем специфичность терминов для уровней Seniority
lexicon = {}
for c in classes:
    in_mask = (df["label"].values == c)
    if not in_mask.any(): continue
    
    mean_in = np.asarray(X[in_mask].mean(axis=0)).ravel()
    mean_out = np.asarray(X[~in_mask].mean(axis=0)).ravel()
    score = mean_in - mean_out
    
    scored_idx = np.argsort(-score)
    top_terms = []
    for idx in scored_idx:
        term = clean_term(terms[idx])
        if score[idx] <= 0 or len(term) < 3 or term in STOP:
            continue
        top_terms.append(term)
        if len(top_terms) >= 50: # Для Seniority 50 слов на класс за глаза
            break
    lexicon[c] = top_terms

# 4. Сохраняем
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(lexicon, f, ensure_ascii=False, indent=2)

print(f"Seniority lexicon saved to {OUT_PATH}")
