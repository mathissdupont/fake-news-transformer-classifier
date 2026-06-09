# CENG454 Fake News Detection Project

This repository contains the final project for CENG454.

## Project Goal

Compare a TF-IDF baseline with transformer-based sentence embeddings for fake news detection.

## Planned Structure

- `data/`: raw and processed dataset files
- `src/`: preprocessing, training, and evaluation code
- `notebooks/`: exploratory analysis and experiments
- `models/`: saved model artifacts
- `reports/`: figures, tables, and the final PDF report
- `docs/`: project notes, AI usage log, and decisions

## AI Usage

All AI assistance used during planning, writing, or debugging must be logged in `docs/ai-usage.md`.

## Team Workflow

The three-person work split and the AI usage standard are documented in `docs/team-split.md` and `AGENTS.md`.

## Initial Workflow

1. Prepare the dataset and clean labels.
2. Train the TF-IDF + Logistic Regression baseline.
3. Train embedding-based models.
4. Evaluate with accuracy, precision, recall, F1-score, and confusion matrices.
5. Write the final report and bibliography in IEEE format.

Quick Run (minimal)

```bash
# create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt

# prepare data (will download from HF)
python scripts/download_prepare_dataset.py

# train TF-IDF baseline
python src/train_tfidf.py

# train embedding-based classifiers
python src/train_embeddings.py

# evaluate and collect results
python src/evaluate.py
```

## UI Demo

The project includes a Streamlit interface with a ChatGPT-style analysis panel.
Paste a news text into the chat box to see the selected model's real/fake
probabilities and a short explanation of the prediction.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The UI can switch between available saved models. The TF-IDF model is available
when these artifacts exist:

- `models/tfidf_model.pkl`
- `models/tfidf_vectorizer.pkl`

If these files are missing, run:

```bash
python src/train_tfidf.py
```

Embedding model options appear after running:

```bash
python src/train_embeddings.py
```
