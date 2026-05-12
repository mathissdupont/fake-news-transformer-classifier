# Project Decisions

## Working Assumptions

- Use Python for the implementation.
- Treat the task as an experimental machine learning project, not a full production system.
- Compare a TF-IDF baseline with transformer-based embeddings.
- Keep all results reproducible with fixed random seeds.

## Open Choices

- Turkish dataset or English backup dataset
- Final classifier set for embeddings
- Whether to add Random Forest as an optional comparison

## Collaboration Standard

- The project is split into three mostly independent work packages: data/baseline, embeddings/models, and evaluation/reporting.
- Every AI-assisted action must be logged in `docs/ai-usage.md` immediately after it happens.
- Any AI-influenced project decision should also be noted here if it changes the plan or deliverables.
