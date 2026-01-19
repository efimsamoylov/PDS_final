import json
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

CSV_PATH = "/Users/efim/Desktop/department-v2.csv"
OUT_PATH = "/Users/efim/Desktop/Results/department_lexicon.json"

# Мини-стоплист для мультиязычных "мусорных" слов (расширишь по ходу)
STOP = {
    "und", "oder", "der", "die", "das", "im", "in", "am", "an", "auf", "für", "mit", "von", "zu",
    "de", "la", "le", "les", "du", "des", "et", "en", "au", "aux",
    "of", "and", "the", "to", "for", "with", "on", "at", "from", "as",
    "bei", "im", "ins", "zum", "zur", "vom", "über", "ueber", "unter", "zwischen",
    "avec", "pour", "par", "dans", "sur", "sans", "chez",
    "or", "is", "are", "was", "were", "be", "been", "being"
}

# Generic seniority / job-level words that are not department-specific
GENERIC_JOB_WORDS = {
    # English
    "manager", "management", "senior", "junior", "director", "head", "lead", "officer",
    "assistant", "specialist", "consultant", "analyst", "executive", "coordinator",
    "project", "projects", "strategy", "developer", "application", "applications",
    "operations", "business", "development", "vp", "vice", "president", "chief", "ceo", "cfo", "cto", "cio", "coo", "founder", "cofounder", "co-founder", "owner", "partner", "sr", "sen",

    # German (role/seniority words that are not department-specific)
    "leiter", "leiterin", "leitung", "teamleiter", "teamleitung", "bereichsleiter", "bereichsleitung",
    "geschäftsführer", "geschäftsführerin", "geschäftsführung", "geschäftsleitung", "vorstand",
    "prokurist", "prokuristin", "assistenz", "assistent", "assistentin", "sekretariat", "sekretärin",

    # French
    "chef", "responsable", "directeur", "directrice", "administrateur", "administratrice"
}

# Company legal forms / boilerplate tokens that should not drive department prediction
LEGAL_FORM_WORDS = {
    "gmbh", "ag", "kg", "kgaa", "gbr", "ohg", "eg", "ev", "e.v", "bv", "nv",
    "inc", "ltd", "llc", "sarl", "s.a", "sa", "plc", "co"
}

# Frequent brand/region noise that should not drive department prediction
BRAND_REGION_NOISE = {
    "volkswagen", "keysight", "jaguar", "rover", "land rover", "jaguar land", "swiss",
    "dach", "emea", "eemea", "europe", "west europe", "south west", "dach central",
    "life sciences", "sciences", "beauty", "speaker", "printmedien",
    "central eastern", "eastern europe", "eastern", "northern europe", "dach region", "europe region",
    "west", "south", "north", "european",
    "crm sharepoint", "salesforce chez", "sap salesforce", "systems cutting", "cutting tools", "service digital", "digital systems",
    "healthcare it"
}

# Generic non-department tokens that are too broad and harm rule-based department scoring
GENERIC_NON_DEPT_NOISE = {
    "service", "services", "tool", "tools", "event", "events", "market", "markets",
    "international", "global", "regional", "region", "country",
    "customer", "clients", "account", "accounts", "solutions", "solution",
    "digital", "transformation", "innovation", "platform", "products", "product",
    "commercial", "retail", "industry", "industries", "business"
}

def clean_term(t: str) -> str:
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t

def is_bad_term(t: str) -> bool:
    t = clean_term(t)
    if t in STOP:
        return True

    parts = t.split()

    # Drop bigrams where ANY token is a stopword (e.g., "of marketing", "marketing und")
    if len(parts) == 2 and (parts[0] in STOP or parts[1] in STOP):
        return True

    # Drop legal/company-form boilerplate (e.g., "gmbh", "co kg", "gmbh co")
    if any(p in LEGAL_FORM_WORDS for p in parts):
        return True

    # Drop frequent brand/region noise (helps prevent repeated blocks leaking into many departments)
    if t in BRAND_REGION_NOISE:
        return True

    # Drop overly broad non-department tokens
    if t in GENERIC_NON_DEPT_NOISE:
        return True

    # Drop terms that contain mostly digits/punctuation (IDs, codes)
    if sum(ch.isalnum() for ch in t) < 3:
        return True

    # Drop terms that are dominated by generic job-level words (better handled in seniority rules)
    if any(p in GENERIC_JOB_WORDS for p in parts):
        return True

    # Drop very short abbreviations that often appear as noise (except common domain abbreviations)
    if len(parts) == 1 and len(parts[0]) <= 2:
        return True

    # Too short / uninformative
    if len(t) < 3:
        return True

    return False

df = pd.read_csv(CSV_PATH)
df["text"] = df["text"].astype(str)
df["label"] = df["label"].astype(str)

texts = df["text"].tolist()
labels = df["label"].tolist()
classes = sorted(df["label"].unique())

# TF-IDF, включая биграммы; token_pattern оставляет слова/дефисы/плюсы/точки
vec = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8,
    token_pattern=r"(?u)\b[\w\-\+\.]{2,}\b"
)
X = vec.fit_transform(texts)
terms = np.array(vec.get_feature_names_out())

# Считаем, насколько термин специфичен для класса:
# score = mean_tfidf_in_class - mean_tfidf_outside_class
lexicon = {}
for c in classes:
    in_mask = (df["label"].values == c)
    X_in = X[in_mask]
    X_out = X[~in_mask]

    mean_in = np.asarray(X_in.mean(axis=0)).ravel()
    mean_out = np.asarray(X_out.mean(axis=0)).ravel()
    score = mean_in - mean_out

    # Build a scored candidate list, then filter and keep more than needed (we'll prune overlaps later)
    scored_idx = np.argsort(-score)
    scored_terms = []
    for idx in scored_idx:
        term = clean_term(str(terms[idx]))
        if is_bad_term(term):
            continue
        scored_terms.append((term, float(score[idx])))
        if len(scored_terms) >= 1000:  # keep a larger pool per class for later overlap pruning
            break

    # Keep as list of (term, score) for post-processing
    lexicon[c] = scored_terms

# Post-process: remove terms that appear in >=2 classes (reduces ambiguity)
term_counts = {}
for c, items in lexicon.items():
    for t, _ in items:
        term_counts[t] = term_counts.get(t, 0) + 1

final_lexicon = {}
for c, items in lexicon.items():
    unique_only = [(t, s) for (t, s) in items if term_counts.get(t, 0) == 1]

    seen = set()
    out = []

    # 1) Prefer unique terms
    for t, _ in unique_only:
        if t in seen:
            continue
        out.append(t)
        seen.add(t)
        if len(out) == 60:
            break

    # 2) If still short, fill with low-overlap terms (appear in at most 3 classes), skipping known noise
    if len(out) < 60:
        for t, _ in items:
            if t in seen:
                continue
            if t in BRAND_REGION_NOISE:
                continue
            if term_counts.get(t, 0) > 3:
                continue
            out.append(t)
            seen.add(t)
            if len(out) == 60:
                break

    # 3) Final fallback: if still short, fill with remaining terms even if they overlap,
    # while still avoiding obvious noise lists.
    if len(out) < 60:
        for t, _ in items:
            if t in seen:
                continue
            if t in BRAND_REGION_NOISE or t in GENERIC_NON_DEPT_NOISE:
                continue
            out.append(t)
            seen.add(t)
            if len(out) == 60:
                break

    final_lexicon[c] = out

lexicon = final_lexicon

# Сохраняем
import os
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(lexicon, f, ensure_ascii=False, indent=2)

print(f"Saved: {OUT_PATH}")
example_key = "Marketing" if "Marketing" in lexicon else (list(lexicon.keys())[0] if lexicon else None)
if example_key:
    print(f"Example ({example_key}):", lexicon.get(example_key, [])[:15])