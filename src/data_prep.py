"""
STEP 1: Data Preparation
------------------------
Downloads / loads the CUAD (Contract Understanding Atticus Dataset) and
converts it into a flat table of (clause_text, category, contract_id, is_risky).

CUAD source: https://github.com/TheAtticusProject/cuad
Download the file `CUAD_v1.json` from their releases and place it in:
    clauseguard/data/CUAD_v1.json

CUAD format: SQuAD-style JSON. Each "context" is a contract (or chunk of one),
and each "qas" entry asks about a specific clause TYPE (e.g. "Governing Law",
"Termination For Convenience"). The "answers" field gives the exact clause
text (if that clause type exists in the contract).

We flatten this into: one row per (contract, clause_type) with the actual
clause text (if found) so we can train a classifier on clause_text -> risk label.
"""

import json
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_PATH = DATA_DIR / "CUAD_v1.json"
OUT_PATH = DATA_DIR / "clauses_flat.csv"

# Clause categories we treat as "high risk" if PRESENT and unfavorable-sounding,
# or "risky if MISSING" (e.g. no cap on liability). You will refine this list
# after looking at real data — this is a reasonable starting point.
HIGH_RISK_IF_PRESENT = {
    "Uncapped Liability",
    "Non-Compete",
    "Exclusivity",
    "No-Solicit Of Employees",
    "Liquidated Damages",
    "Change Of Control",
    "Most Favored Nation",
}

RISKY_IF_MISSING = {
    "Termination For Convenience",
    "Cap On Liability",
    "Governing Law",
    "Insurance",
}


def load_cuad(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(
            f"CUAD_v1.json not found at {raw_path}.\n"
            "Download it from https://github.com/TheAtticusProject/cuad "
            "(see their 'data' release) and place it in the data/ folder."
        )

    with open(raw_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    rows = []
    for doc in raw["data"]:
        contract_title = doc["title"]
        for para in doc["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                # qa["id"] looks like "<contract>__<Clause Category>"
                clause_type = qa["id"].split("__")[-1].replace("_", " ")
                answers = qa.get("answers", [])
                is_answerable = len(answers) > 0

                if is_answerable:
                    for ans in answers:
                        rows.append(
                            {
                                "contract_id": contract_title,
                                "clause_type": clause_type,
                                "clause_text": ans["text"].strip(),
                                "present": True,
                            }
                        )
                else:
                    rows.append(
                        {
                            "contract_id": contract_title,
                            "clause_type": clause_type,
                            "clause_text": "",
                            "present": False,
                        }
                    )

    df = pd.DataFrame(rows)
    return df


def label_risk(row) -> str:
    """Turn (clause_type, present) into a 4-way risk label."""
    if row["present"] and row["clause_type"] in HIGH_RISK_IF_PRESENT:
        return "High-Risk"
    if not row["present"] and row["clause_type"] in RISKY_IF_MISSING:
        return "Missing"
    if row["present"]:
        return "Standard"
    return "Not-Applicable"


def build_dataset() -> pd.DataFrame:
    df = load_cuad()
    df["risk_label"] = df.apply(label_risk, axis=1)

    # For training the text classifier we only want rows that actually HAVE
    # clause text (present == True) — "Missing" rows have no text to embed,
    # they're handled by the rule engine instead (see risk_flagger.py).
    df_text = df[df["present"] & (df["clause_text"].str.len() > 20)].copy()
    df_text = df_text.drop_duplicates(subset=["clause_text"])

    DATA_DIR.mkdir(exist_ok=True)
    df_text.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df_text)} labeled clauses to {OUT_PATH}")
    print(df_text["risk_label"].value_counts())
    return df_text


if __name__ == "__main__":
    build_dataset()
