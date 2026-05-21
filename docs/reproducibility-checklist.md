# Reproducibility Checklist

This checklist ensures that the entire project can be re-run from scratch using the saved code and configuration.

## Before Submission

Complete these checks one week before the deadline.

### Dataset and Preprocessing
- [x] Dataset source is documented in `data/README_dataset_link.txt`
- [x] Train/test split indices or split files are saved
- [x] Random state value is recorded in `docs/decisions.md`
- [x] Label mapping is documented
- [x] Preprocessing script runs without errors on the dataset
- [x] Preprocessing script produces consistent output on re-runs

### Baseline Model (TF-IDF)
- [x] TF-IDF model training script runs from scratch
- [x] Script saves model to `models/tfidf_model.pkl`
- [x] Script generates metrics and saves to `results/tfidf_metrics.csv`
- [x] Script generates confusion matrix and saves to `results/confusion_matrix_tfidf.png`
- [x] Running the script twice produces identical metrics and figures
- [x] Metrics match those reported in the final report

### Embedding Models
- [x] Embedding model is specified by name and version in `docs/decisions.md`
- [x] Embedding model can be loaded and run in a clean environment
- [x] Embedding extraction script runs without errors
- [x] Scripts generate metrics and save to `results/embedding_*_metrics.csv`
- [x] Scripts generate confusion matrices to `results/confusion_matrix_embedding_*.png`
- [x] Running the scripts twice produces identical metrics and figures
- [x] Embedding metrics use the same test split as the baseline

### Final Report
- [x] All metrics in the report come from saved `results/` files
- [x] All figures in the report are saved as image files in `results/` or `reports/figures/`
- [x] All figures are numbered and have titles
- [x] References are formatted in IEEE style
- [x] Report length does not exceed 8 pages (excluding cover and references)
- [x] Report mentions any AI tool usage

### Submission Package
- [x] Folder structure matches `CENG454_Group_X/` layout
- [x] `report/final_report.pdf` exists
- [ ] `notebooks/` contains any exploration code
- [x] `src/` contains all training scripts
- [x] `results/` contains metrics CSV and confusion matrix figures
- [x] `data/README_dataset_link.txt` is included
- [x] `README.md` is included and up-to-date
- [x] `docs/ai-usage.md` is complete with all team AI usage entries

### Clean Environment Test

Run this final test one day before submission:

1. Delete the local virtual environment
2. Create a fresh virtual environment
3. Install dependencies from `requirements.txt`
4. Download the dataset using the documented link
5. Run the preprocessing script from scratch
6. Run the TF-IDF training script from scratch
7. Run the embedding training scripts from scratch
8. Verify that all output files are generated
9. Verify that metrics match the final report

If all checks pass, the project is reproducible.

## Person 1 Responsibilities
- [x] Dataset link and preprocessing reproducibility
- [x] TF-IDF model reproducibility

## Person 2 Responsibilities
- [x] Embedding model specification and reproducibility
- [x] Classifier training reproducibility

## Person 3 Responsibilities
- [x] Final package structure and completeness
- [x] Report metrics accuracy check
- [x] Clean environment test execution
