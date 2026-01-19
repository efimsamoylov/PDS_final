import json
import re
from typing import Dict, List, Tuple, Any


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text


def load_department_lexicon(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def predict_department_rule(
    text: str,
    lexicon: Dict[str, List[str]],
    bigram_weight: float = 2.0,
    unigram_weight: float = 1.0,
    min_score: float = 2.0,
    default_label: str = "Other"
) -> Tuple[str, Dict[str, Any]]:
    """
    Returns (predicted_label, debug_info).
    debug_info contains scores and matched terms per class.
    """
    t = normalize_text(text)

    scores: Dict[str, float] = {}
    matched: Dict[str, List[str]] = {}

    # Precompute a padded string to reduce false substring matches for unigrams.
    # We'll treat bigrams as substring match, unigrams as word-boundary regex match.
    for label, terms in lexicon.items():
        score = 0.0
        m: List[str] = []
        for term in terms:
            term_n = term.strip().lower()
            if not term_n:
                continue

            if " " in term_n:  # bigram/phrase
                if term_n in t:
                    score += bigram_weight
                    m.append(term_n)
            else:  # unigram
                # word boundary match: avoids matching "it" inside "digital"
                if re.search(rf"(?<!\\w){re.escape(term_n)}(?!\\w)", t):
                    score += unigram_weight
                    m.append(term_n)

        scores[label] = score
        matched[label] = m

    best_label = max(scores, key=scores.get) if scores else default_label
    best_score = scores.get(best_label, 0.0)

    if best_score < min_score:
        best_label = default_label if default_label in lexicon else best_label

    debug = {
        "best_score": best_score,
        "scores": scores,
        "matched_terms": {k: v for k, v in matched.items() if v},
    }
    return best_label, debug