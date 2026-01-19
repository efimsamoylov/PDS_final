import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Пути к твоим данным
CSV_PATH = "../data/seniority-v2.csv"  # Путь к размеченному CSV для Seniority
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

# 3. Считаем специфичность (существующий цикл)
raw_lexicon = {}
term_appearance_count = {} # Считаем, в скольких классах появилось слово

for c in classes:
    # ... расчет mean_in, mean_out, score ...
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
        term_appearance_count[term] = term_appearance_count.get(term, 0) + 1
        if len(top_terms) >= 100: # Берем чуть больше для фильтрации
            break
    raw_lexicon[c] = top_terms

# 4. Фильтруем: оставляем только те слова, которые специфичны ТОЛЬКО для одного класса
final_lexicon = {}
for label, terms_list in raw_lexicon.items():
    # Оставляем слово, только если оно не встречается в 2+ классах одновременно
    unique_terms = [t for t in terms_list if term_appearance_count[t] == 1]
    final_lexicon[label] = unique_terms[:50]

# 5. Сохраняем
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(final_lexicon, f, ensure_ascii=False, indent=2)

print(f"Seniority lexicon saved to {OUT_PATH}")
