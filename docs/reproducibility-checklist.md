# Reproducibility Checklist

This checklist ensures that the entire project can be re-run from scratch using the saved code and configuration.

## Before Submission

Complete these checks one week before the deadline.

### Dataset and Preprocessing
- [ ] Dataset source is documented in `data/README_dataset_link.txt`
- [ ] Train/test split indices or split files are saved
- [ ] Random state value is recorded in `docs/decisions.md`
- [ ] Label mapping is documented
- [ ] Preprocessing script runs without errors on the dataset
- [ ] Preprocessing script produces consistent output on re-runs

### Baseline Model (TF-IDF)
- [ ] TF-IDF model training script runs from scratch
- [ ] Script saves model to `models/tfidf_model.pkl`
- [ ] Script generates metrics and saves to `results/tfidf_metrics.csv`
- [ ] Script generates confusion matrix and saves to `results/confusion_matrix_tfidf.png`
- [ ] Running the script twice produces identical metrics and figures
- [ ] Metrics match those reported in the final report

### Embedding Models
- [ ] Embedding model is specified by name and version in `docs/decisions.md`
- [ ] Embedding model can be loaded and run in a clean environment
- [ ] Embedding extraction script runs without errors
- [ ] Scripts generate metrics and save to `results/embedding_*_metrics.csv`
- [ ] Scripts generate confusion matrices to `results/confusion_matrix_embedding_*.png`
- [ ] Running the scripts twice produces identical metrics and figures
- [ ] Embedding metrics use the same test split as the baseline

### Final Report
- [ ] All metrics in the report come from saved `results/` files
- [ ] All figures in the report are saved as image files in `results/` or `reports/figures/`
- [ ] All figures are numbered and have titles
- [ ] References are formatted in IEEE style
- [ ] Report length does not exceed 8 pages (excluding cover and references)
- [ ] Report mentions any AI tool usage

### Submission Package
- [ ] Folder structure matches `CENG454_Group_X/` layout
- [ ] `report/final_report.pdf` exists
- [ ] `notebooks/` contains any exploration code
- [ ] `src/` contains all training scripts
- [ ] `results/` contains metrics CSV and confusion matrix figures
- [ ] `data/README_dataset_link.txt` is included
- [ ] `README.md` is included and up-to-date
- [ ] `docs/ai-usage.md` is complete with all team AI usage entries

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
- [ ] Dataset link and preprocessing reproducibility
- [ ] TF-IDF model reproducibility

## Person 2 Responsibilities
- [ ] Embedding model specification and reproducibility
- [ ] Classifier training reproducibility

## Person 3 Responsibilities
- [ ] Final package structure and completeness
- [ ] Report metrics accuracy check
- [ ] Clean environment test execution
