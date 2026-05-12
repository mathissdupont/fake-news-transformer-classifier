# AI Usage Log

This file records every meaningful use of AI tools during the project.

## Standard

- Add a new row immediately after any AI-assisted planning, coding, debugging, summarization, or documentation work.
- Never leave AI help undocumented, even if the change seems small.
- Keep each entry short, factual, and verifiable.
- If AI influenced a decision, also note it in `docs/decisions.md`.

## Current Entries

| Date | Tool | Purpose | What was asked | Human review |
|------|------|---------|----------------|--------------|
| 2026-04-24 | Copilot | Project planning | Repo structure and submission strategy for CENG454 fake news detection | Plan reviewed manually |
| 2026-04-25 | Copilot | Documentation reading | Summarize the two project PDFs and extract requirements | Requirements checked against the source PDFs |
| 2026-04-26 | Copilot | Repository setup | Create the initial repo scaffold and AI note file | Files reviewed manually |
| 2026-04-26 | Copilot | Workflow design | Split the project into three independent work packages and define the AI logging standard | Split reviewed against project scope and requirements |
| 2026-05-01 | Copilot | Task expansion | Add dataset sourcing rules and more explicit responsibilities for each teammate | Updated split reviewed against the project document |
| 2026-05-05 | Copilot | Role planning | Add pre-coding research tasks, code responsibilities, and example keywords for all teammates | Section reviewed in the team split document |
| 2026-05-06 | Copilot | Template creation | Create dataset link README, results template, reproducibility checklist, and research notes | Templates reviewed against project requirements |
| 2026-05-07 | Copilot | Documentation | Write comprehensive Turkish project overview explaining all technical terms and project goals | Overview reviewed for clarity and completeness |
| 2026-05-07 | Copilot | Sprint planning | Create 6-day sprint plan with daily tasks, milestones, and success criteria | Plan reviewed against deadline and team capacity |
| 2026-05-07 | Copilot | Dataset search | Write step-by-step guide for finding datasets from 5 sources (HF, Kaggle, Google, PWC, TÜİK) | Guide reviewed for clarity and completeness |
| 2026-05-08 | Copilot | Dataset download & preprocessing | Load `isakulaksiz/turkish-fake-news-detection` via `datasets.load_dataset`, saved raw CSV and created processed `train.csv`/`test.csv` (80/20 stratified) | Human verified saved files in `data/raw` and `data/processed` |

## Reporting Rule

If AI is used again for brainstorming, code generation, debugging, text editing, or summarization, add a new row here with the date, tool, purpose, and a short human verification note.
