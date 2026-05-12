# Three-Person Work Split

This split is designed so each person can work largely independently with similar workload.

## Project Starting Point

The project should begin with one shared decision: which dataset will be used.

## Before Coding Starts

All three members should do a short research pass before writing code.

### Shared Research Goals

- Understand fake news detection as a binary text classification problem.
- Compare TF-IDF baselines with transformer-based embeddings.
- Confirm that the chosen dataset is reproducible and suitable for the deadline.
- Collect at least 10 relevant academic sources for the final bibliography.
- Write down candidate datasets, candidate models, and candidate evaluation metrics before implementation.

### Suggested Search Keywords

- fake news detection transformer embeddings
- Sentence-BERT text classification
- Turkish fake news detection dataset
- multilingual fake news classification
- TF-IDF logistic regression text classification
- misinformation detection survey
- news classification BERT Turkish
- WELFake dataset fake news
- Turkish NLP BERTurk classification
- text classification evaluation F1 confusion matrix

### Example Paper Themes To Look For

- surveys on fake news detection and misinformation analysis
- papers comparing traditional ML and transformer embeddings for text classification
- Sentence-BERT and sentence embedding papers
- Turkish NLP papers and BERTurk-related work
- dataset papers or dataset cards for fake news datasets
- papers about class imbalance in text classification
- papers about explainability or error analysis in misinformation detection

### Pre-Implementation Research Checklist

- read 2 to 3 fake news detection survey papers
- read 2 papers about Sentence-BERT or embedding-based classification
- read 2 papers about Turkish NLP or multilingual classification if the dataset is Turkish
- find 1 to 2 dataset sources that look usable
- compare the pros and cons of TF-IDF vs embeddings in a short note
- record every useful source in the bibliography draft or a notes file

### Preferred Dataset Sources

- Hugging Face Datasets: search for Turkish fake news or binary news classification datasets.
- Kaggle Datasets: search for Turkish fake news datasets and the WELFake dataset as a backup.
- Google Dataset Search: use it to discover additional Turkish or multilingual news datasets.
- Papers With Code: check whether any dataset cards or benchmark pages are linked.

### Preferred Dataset Choice Order

1. Turkish fake news dataset with news text and a binary fake/real label.
2. If the Turkish data is not reliable, incomplete, or inaccessible, use WELFake as the backup.
3. If both are unavailable, document the issue and select the closest reproducible news classification dataset with clear labeling.

### Dataset Selection Criteria

- Text column must contain full news content, not only headlines.
- Labels must be clearly mappable to exactly two classes.
- Dataset should have enough samples for a meaningful train/test split.
- Missing values, duplicate rows, and label noise must be manageable.
- The dataset should be reproducible through a public link or downloadable source.

### Dataset Documentation Requirement

- Save the final dataset link or source note in `data/README_dataset_link.txt`.
- Record the chosen dataset name, download date, and source URL in `docs/decisions.md`.
- Note any label mapping rules in `README.md` or `docs/decisions.md` before training begins.

## Global Rules

- Use one shared dataset contract: `text` column for news content and `label` column for fake/real class.
- Freeze the label mapping before any training starts and document it in the README or decisions file.
- Use one shared `random_state` value for all splits and experiments that need reproducibility.
- Store every generated artifact using agreed filenames so other members can reuse them without renaming.
- Every AI-assisted task must be logged immediately in `docs/ai-usage.md`.

## Shared Contracts

- Dataset schema: one text column and one binary label column.
- Label mapping must be fixed and documented before training starts.
- Train/test split must use one shared `random_state` value.
- Every script must save outputs using agreed filenames in `results/` or `reports/`.
- All raw data should stay in `data/raw/` and cleaned outputs in `data/processed/`.
- The final zip should include the dataset link file if the dataset itself cannot be redistributed.

## Person 1: Data and Baseline

Primary responsibility:

Research tasks before coding:

- search for dataset-specific papers or dataset cards and write down what kind of labels and columns they use
- compare at least two candidate datasets and decide which one is easier to reproduce
- look for preprocessing examples in papers that use fake news datasets or Turkish news classification
- collect keywords about label cleaning, text normalization, and train/test splitting for text classification
- note any dataset license or access restriction before downloading

Code tasks:

- search the agreed dataset sources and shortlist candidate datasets
- verify that each candidate has a public link, enough samples, and a binary label structure
- download the selected raw dataset to `data/raw/`
- create a small dataset summary: source, size, language, label names, missing values, duplicates
- inspect column names and confirm which column is the text field and which is the label field
- decide the exact label mapping, for example fake/real or 0/1 to fake/real
- remove empty text rows, duplicated rows, malformed rows, and invalid labels
- convert text values to string type and standardize whitespace
- create the shared train/test split with stratified sampling and fixed `random_state`
- save the split indices or split files so the same split can be reused by everyone
- write the preprocessing helper functions used by both the baseline and embedding pipelines
- build the TF-IDF + Logistic Regression baseline
- optionally test one extra TF-IDF configuration to verify the baseline is reasonable
- save predictions, metrics, and the confusion matrix under standard filenames
- write a short note describing preprocessing choices and any data quality issues

Deliverables:

- cleaned dataset description
- preprocessing script
- TF-IDF results table
- baseline confusion matrix
- dataset quality summary
- exact preprocessing checklist
- dataset source note
- saved split indices or split files

Detailed output checklist:

- `data/raw/<dataset_file>`
- `data/processed/train.csv` and `data/processed/test.csv` or equivalent split files
- `results/tfidf_metrics.csv`
- `results/confusion_matrix_tfidf.png`
- `docs/decisions.md` entry for the dataset choice

## Person 2: Embeddings and Models

Primary responsibility:

Research tasks before coding:

- read papers or documentation about Sentence-BERT and multilingual sentence embeddings
- compare embedding models that work well for Turkish or multilingual text if needed
- search for papers on transformer embeddings used as features with classical classifiers
- collect keywords about embedding pooling, classifier choice, and runtime tradeoffs
- check whether there are Turkish transformer models or sentence embedding models suitable for this project

Code tasks:

- choose the embedding model after confirming the dataset language and size
- test that the selected SentenceTransformer model loads correctly in the environment
- confirm the embedding model name and version in `docs/decisions.md`
- build the embedding extraction pipeline separately for train and test data
- cache embeddings if they are expensive to recompute or large enough to slow the workflow
- compare at least Logistic Regression and SVM on the embeddings
- optionally run Random Forest only if the result adds value and time permits
- save trained classifiers to `models/` with clear filenames
- generate predictions on the exact same test split used by Person 1
- compare every embedding result directly against the TF-IDF baseline metrics
- inspect false positives and false negatives and write a short error analysis note
- record runtime, memory use, and any model loading issues
- verify that the embedding pipeline can be rerun from a clean environment

Deliverables:

- embedding extraction script
- embedding-based training script
- model comparison results
- confusion matrix figures
- cached embeddings or model artifacts if needed
- short error analysis notes
- model loading note
- runtime and memory note

Detailed output checklist:

- `models/embedding_lr.pkl` or equivalent
- `models/embedding_svm.pkl` or equivalent
- `results/embedding_metrics.csv`
- `results/confusion_matrix_embedding_lr.png`
- `results/confusion_matrix_embedding_svm.png`
- `results/error_analysis.md`

## Person 3: Evaluation, Report, and Reproducibility

Primary responsibility:

Research tasks before coding:

- read report-writing examples for short AI/data science final projects
- collect citation details for the benchmark papers and dataset sources used by the team
- find papers or guides about evaluation metrics for imbalanced binary classification
- search for examples of confusion matrix analysis and error analysis sections
- review IEEE formatting examples so the report can be written correctly from the start

Code tasks:

- collect the outputs from Person 1 and Person 2 and merge them into one results table
- verify that every reported metric comes from real saved predictions, not copied values
- build or maintain the evaluation script that reads saved predictions and writes metrics tables
- check that confusion matrices, classification reports, and tables all use the same split and label order
- format tables and figures so they fit the report template cleanly
- give every figure a number, title, and a one- or two-sentence explanation
- validate that the final outputs are reproducible from the saved scripts and configuration files
- check that filenames, paths, and saved artifacts match the project conventions
- write the introduction and motivation section from the actual project scope
- write the background and related work section with at least 10 real sources
- write the dataset and preprocessing section using the final dataset and cleaning steps
- write the proposed approach and implementation section using the exact model pipeline
- write the evaluation, discussion, limitations, and conclusion sections
- check the final PDF length against the 8-page limit before submission
- format all references in IEEE style
- prepare the submission zip folder structure and verify every required file is present
- verify that the final zip contains code, report, dataset note, and all required result files
- maintain the AI usage log throughout the project
- add the final reproducibility checklist and submission checklist
- confirm the final report mentions any AI use explicitly

Deliverables:

- final report PDF
- bibliography
- reproducibility checklist
- completed AI usage log
- submission zip checklist
- final report structure draft
- evaluation script or aggregation script
- reproducibility verification note
- final package validation checklist

Technical responsibilities:

- run the final metric aggregation step
- cross-check metric values against saved prediction files
- confirm that all figures were generated from the same experiment outputs
- verify the final submission package before upload

## Why This Split Works

- Person 1 can work from raw data to baseline without waiting for model experiments.
- Person 2 can build the embedding pipeline from the same cleaned dataset contract.
- Person 3 can draft the report and documentation in parallel once the shared outputs are defined.
- The workload is balanced because each person owns one major technical slice and one documentation slice.

## AI Logging Standard

Every time AI is used for any task, add a new row to `docs/ai-usage.md` with:

1. date
2. tool name
3. purpose
4. a short summary of the prompt or request
5. one line saying how a human verified the result

If AI changes the project direction, also add a short note to `docs/decisions.md`.

## Done Criteria For Each Person

- Person 1 is done only when the cleaned dataset, split files, and baseline metrics exist and can be reused by others.
- Person 2 is done only when the embedding experiments run on the same split and produce saved metrics, figures, and model outputs.
- Person 3 is done only when the report draft, final bibliography, AI log, and submission checklist are complete.

## Immediate Next Files To Create

- `data/README_dataset_link.txt`
- `results/` summary template
- `docs/decisions.md` entry for the final dataset source
- `docs/reproducibility-checklist.md`
- `docs/research-notes.md`
