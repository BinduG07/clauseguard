# ClauseGuard — Multi-Agent Legal Contract Reviewer

AI system that extracts clauses from a contract, retrieves similar standard
clauses, flags risk (Rules + ML classifier + LLM fallback), and generates a
structured risk report.

## Architecture
Extractor → Retriever (FAISS + Sentence-BERT) → Risk-Flagger (Rules + XGBoost + Groq LLM) → Summarizer
Orchestrated with **LangGraph**.

## Setup

```bash
cd clauseguard
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get a free Groq API key from https://console.groq.com and set it:
```bash
export GROQ_API_KEY=your_key_here     # Windows: set GROQ_API_KEY=your_key_here
```

## Get the dataset
Download `CUAD_v1.json` from https://github.com/TheAtticusProject/cuad
(see their releases / data folder) and place it at:
```
clauseguard/data/CUAD_v1.json
```

## Run order (do these once, in order)

```bash
cd src
python data_prep.py          # Step 1: flatten CUAD into clauses_flat.csv
python embeddings.py         # Step 2: build FAISS index over reference clauses
python train_classifier.py   # Step 3: train XGBoost risk classifier + SHAP + ablation
```

## Try the pipeline on a sample contract
```bash
python pipeline.py           # runs the 4-agent graph on a built-in sample
```

## Launch the demo UI
```bash
cd ..
streamlit run app.py
```

## What to measure for your report / resume bullets
- **Segmentation accuracy** — hand-label ~20-30 real contracts, compare
  against `extractor.py`'s output using `evaluate_segmentation()`.
- **Classifier performance** — cross-val F1 and held-out precision/recall,
  printed by `train_classifier.py`.
- **Ablation table** — rules-only vs ML-only vs combined, also printed by
  `train_classifier.py` (`run_ablation()`).
- **Hallucination rate** — log how often `risk_flagger.py`'s LLM step gets
  rejected by the citation-verification check (`source: llm_rejected`) vs
  accepted, before/after you tighten the prompt.

## Folder structure
```
clauseguard/
├── data/                 # CUAD_v1.json goes here; clauses_flat.csv generated
├── models/               # FAISS index, trained classifier, SHAP artifacts
├── src/
│   ├── data_prep.py      # Step 1
│   ├── embeddings.py     # Step 2
│   ├── train_classifier.py  # Step 3
│   ├── extractor.py      # Extractor agent
│   ├── risk_flagger.py   # Risk-Flagger agent (rules + ML + LLM)
│   └── pipeline.py       # LangGraph orchestration (all 4 agents)
├── app.py                # Streamlit demo
└── requirements.txt
```
