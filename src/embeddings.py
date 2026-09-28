"""
STEP 2: Embeddings + FAISS Retrieval Index
-------------------------------------------
Converts every reference clause into a Sentence-BERT embedding and builds a
FAISS index so the Retriever Agent can find the most similar "standard"
clause for any new clause it sees.
"""

import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent.parent / "data"
MODEL_DIR = Path(__file__).parent.parent / "models"
CLAUSES_CSV = DATA_DIR / "clauses_flat.csv"
INDEX_PATH = MODEL_DIR / "clause_index.faiss"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for clause similarity


def build_index():
    df = pd.read_csv(CLAUSES_CSV)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print(f"Embedding {len(df)} clauses with {EMBED_MODEL_NAME} ...")
    embeddings = model.encode(
        df["clause_text"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,  # so we can use inner product = cosine similarity
    )
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product on normalized vecs = cosine sim
    index.add(embeddings)

    MODEL_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))
    df.to_csv(MODEL_DIR / "clause_lookup.csv", index=False)  # row i <-> index i
    print(f"Saved FAISS index ({index.ntotal} vectors) to {INDEX_PATH}")


class ClauseRetriever:
    """Loads the prebuilt index and lets you query it at inference time."""

    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_PATH))
        self.lookup = pd.read_csv(MODEL_DIR / "clause_lookup.csv")

    def query(self, clause_text: str, k: int = 3):
        vec = self.model.encode([clause_text], normalize_embeddings=True).astype("float32")
        scores, idxs = self.index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            row = self.lookup.iloc[idx]
            results.append(
                {
                    "matched_clause_type": row["clause_type"],
                    "matched_text": row["clause_text"],
                    "matched_risk_label": row["risk_label"],
                    "similarity": float(score),
                }
            )
        return results


if __name__ == "__main__":
    build_index()
