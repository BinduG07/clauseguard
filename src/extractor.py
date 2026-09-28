"""
STEP 4: Extractor Agent
------------------------
Splits a raw contract (plain text, already extracted from PDF) into
individual clauses. Contracts are inconsistently formatted, so this uses a
heading-pattern heuristic first, then falls back to paragraph splitting.

This is the piece you should evaluate against a hand-labeled test set to get
your "segmentation accuracy" resume metric.
"""

import re
from typing import List, Dict

# Matches headings like "1. Termination", "Section 4 - Confidentiality",
# "ARTICLE III: INDEMNIFICATION", "4.2 Governing Law"
HEADING_PATTERN = re.compile(
    r"^\s*(?:(?:ARTICLE|Section|Clause)\s+)?"
    r"(?:\d+(?:\.\d+)*|[IVXLC]+)[\.\)\:\-]?\s+"
    r"([A-Z][A-Za-z0-9 ,\-\&/]{2,60})\s*$",
    re.MULTILINE,
)


def extract_clauses(contract_text: str) -> List[Dict]:
    contract_text = contract_text.replace("\r\n", "\n")
    matches = list(HEADING_PATTERN.finditer(contract_text))

    clauses = []
    if matches:
        for i, match in enumerate(matches):
            heading = match.group(1).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(contract_text)
            body = contract_text[start:end].strip()
            if len(body) > 15:  # skip empty/near-empty sections
                clauses.append({"heading": heading, "clause_text": body})
    else:
        # Fallback: split on double newlines (paragraph breaks)
        paragraphs = [p.strip() for p in contract_text.split("\n\n") if len(p.strip()) > 40]
        for i, p in enumerate(paragraphs):
            clauses.append({"heading": f"Paragraph {i+1}", "clause_text": p})

    return clauses


def evaluate_segmentation(pred_clauses: List[Dict], gold_clauses: List[Dict]) -> float:
    """
    Rough segmentation accuracy metric: fraction of gold clause texts that
    have a predicted clause with >90% text overlap. Use this against a
    hand-labeled set of ~20-30 contracts for your resume metric.
    """
    matched = 0
    for gold in gold_clauses:
        gold_text = gold["clause_text"]
        for pred in pred_clauses:
            overlap = len(set(gold_text.split()) & set(pred["clause_text"].split()))
            if overlap / max(len(gold_text.split()), 1) > 0.9:
                matched += 1
                break
    return matched / max(len(gold_clauses), 1)


if __name__ == "__main__":
    sample = """
    1. Confidentiality
    Each party agrees to keep all shared information confidential for a period of 5 years.

    2. Termination
    Either party may terminate this agreement with 30 days written notice.

    3. Governing Law
    This agreement shall be governed by the laws of the State of Delaware.
    """
    for c in extract_clauses(sample):
        print(c["heading"], "->", c["clause_text"][:60])
