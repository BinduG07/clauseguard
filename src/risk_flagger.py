"""
STEP 5: Risk-Flagger Agent
---------------------------
Combines three layers, same pattern as the UPI project:
    1. Rule engine       -> hard, deterministic red flags (fast, no LLM cost)
    2. ML classifier      -> trained risk_classifier.joblib (from train_classifier.py)
    3. LLM fallback       -> only called when ML confidence is low / clause is ambiguous,
                             and REQUIRED to quote the exact clause text it's judging
                             (this is your hallucination-control mechanism)
"""

import os
import json
import joblib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Explicitly point to the .env file in the project root (one level above src/)
# so this works regardless of which folder you run the script from.
ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=ENV_PATH)

MODEL_DIR = Path(__file__).parent.parent / "models"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

RULE_KEYWORDS = {
    "High-Risk": ["unlimited liability", "sole discretion", "non-compete", "perpetual",
                  "irrevocable", "no liability cap"],
}

CONFIDENCE_THRESHOLD = 0.6  # below this, escalate to LLM


class RiskFlagger:
    def __init__(self, groq_api_key: str | None = None):
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)
        self.clf = joblib.load(MODEL_DIR / "risk_classifier.joblib")
        self.label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
        self.groq_client = Groq(api_key=groq_api_key or os.environ.get("GROQ_API_KEY"))

    def rule_check(self, clause_text: str) -> str | None:
        text_l = clause_text.lower()
        for label, keywords in RULE_KEYWORDS.items():
            if any(kw in text_l for kw in keywords):
                return label
        return None

    def ml_predict(self, clause_text: str):
        vec = self.embedder.encode([clause_text])
        probs = self.clf.predict_proba(vec)[0]
        best_idx = int(np.argmax(probs))
        label = self.label_encoder.inverse_transform([best_idx])[0]
        return label, float(probs[best_idx])

    def llm_fallback(self, clause_text: str) -> dict:
        """
        LLM is required to quote (cite) the exact clause text back in its
        reasoning. If it can't ground its answer in the given text, we treat
        that as a failure and fall back to a safe default rather than trust
        an ungrounded answer -- this is the citation-verification /
        hallucination-control mechanism worth writing up as a finding.
        """
        prompt = f"""You are a contract risk reviewer. Read the clause below and classify it
as one of: Standard, Unusual, High-Risk.

Clause:
\"\"\"{clause_text}\"\"\"

Respond ONLY in JSON with keys: label, quoted_evidence, reasoning.
"quoted_evidence" MUST be an exact substring copied from the clause above.
"""
        try:
            resp = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = resp.choices[0].message.content
            parsed = json.loads(content)

            if parsed.get("quoted_evidence", "") not in clause_text:
                return {
                    "label": "Needs-Human-Review",
                    "reasoning": "LLM evidence did not match source text (possible hallucination); escalated.",
                    "source": "llm_rejected",
                }
            parsed["source"] = "llm"
            return parsed

        except Exception as e:
            return {
                "label": "Needs-Human-Review",
                "reasoning": f"LLM call failed ({type(e).__name__}); escalated for manual review.",
                "source": "llm_failure",
            }

    def flag(self, clause_text: str) -> dict:
        rule_label = self.rule_check(clause_text)
        if rule_label:
            return {"label": rule_label, "source": "rule_engine", "confidence": 1.0}

        ml_label, confidence = self.ml_predict(clause_text)
        if confidence >= CONFIDENCE_THRESHOLD:
            return {"label": ml_label, "source": "ml_classifier", "confidence": confidence}

        llm_result = self.llm_fallback(clause_text)
        llm_result["confidence"] = confidence
        return llm_result


if __name__ == "__main__":
    flagger = RiskFlagger()
    test_clause = "The Company shall have unlimited liability for any breach of this agreement."
    print(flagger.flag(test_clause))