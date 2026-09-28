"""
STEP 3: Train the Risk Classifier (the ML core of the project)
----------------------------------------------------------------
Trains a model that maps clause_text -> risk_label (Standard / High-Risk /
Not-Applicable), using Sentence-BERT embeddings as features.

Also runs the ablation study your resume bullet is built on:
    (a) rules-only baseline
    (b) ML-only (this classifier)
    (c) ML + rules combined
and reports precision/recall/F1 for each, plus a SHAP summary plot for
explainability (like the UPI project's SHAP usage).
"""

import numpy as np
import pandas as pd
import shap
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
CLAUSES_CSV = DATA_DIR / "clauses_flat.csv"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_features():
    df = pd.read_csv(CLAUSES_CSV)
    embedder = SentenceTransformer(EMBED_MODEL_NAME)
    X = embedder.encode(df["clause_text"].tolist(), show_progress_bar=True)
    y = df["risk_label"]
    return df, np.array(X), y


def train():
    df, X, y_text = load_features()

    # XGBoost's sklearn API requires numeric class labels, not strings.
    # We encode "Standard" -> 0, "High-Risk" -> 1, etc. and save the encoder
    # so risk_flagger.py can decode predictions back to readable labels.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=42,
    )

    # Cross-validated score -> this is your "0.821 ± 0.009 AUC"-style bullet
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="f1_weighted")
    print(f"Cross-val F1 (weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Decode back to text labels for a readable report
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    print("\n=== Classification Report (held-out test set) ===")
    print(classification_report(y_test_labels, y_pred_labels))
    print("Confusion matrix:\n", confusion_matrix(y_test_labels, y_pred_labels))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(clf, MODEL_DIR / "risk_classifier.joblib")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.joblib")

    # ---- SHAP explainability ----
    print("\nComputing SHAP values (this explains WHY a clause was flagged)...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test[:50])  # sample for speed
    joblib.dump(
        {"explainer": explainer, "sample_X": X_test[:50], "shap_values": shap_values},
        MODEL_DIR / "shap_artifacts.joblib",
    )
    print("Saved SHAP artifacts for report generation.")

    return clf, (X_test, y_test, y_pred)


def rule_only_baseline(df: pd.DataFrame):
    """Simple keyword-rule baseline, used only for the ablation comparison table."""
    RISKY_KEYWORDS = ["unlimited liability", "sole discretion", "non-compete", "perpetual"]

    def rule_predict(text: str) -> str:
        text_l = text.lower()
        if any(kw in text_l for kw in RISKY_KEYWORDS):
            return "High-Risk"
        return "Standard"

    preds = df["clause_text"].apply(rule_predict)
    return preds


def run_ablation():
    """
    Reproduces the "rules-only vs ML-only vs combined" comparison table
    for your resume bullet / evaluation section.
    """
    df, X, y = load_features()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    test_idx = train_test_split(df.index, test_size=0.2, random_state=42, stratify=y)[1]
    df_test = df.loc[test_idx]

    clf = joblib.load(MODEL_DIR / "risk_classifier.joblib")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    ml_preds = label_encoder.inverse_transform(clf.predict(X_test))
    rule_preds = rule_only_baseline(df_test)

    # Combined: trust rules for a clear keyword hit, else fall back to ML
    combined_preds = [
        r if r == "High-Risk" else m for r, m in zip(rule_preds, ml_preds)
    ]

    print("\n=== Ablation: Rules-only ===")
    print(classification_report(y_test, rule_preds, zero_division=0))
    print("\n=== Ablation: ML-only ===")
    print(classification_report(y_test, ml_preds, zero_division=0))
    print("\n=== Ablation: Combined (Rules + ML) ===")
    print(classification_report(y_test, combined_preds, zero_division=0))


if __name__ == "__main__":
    train()
    run_ablation()
