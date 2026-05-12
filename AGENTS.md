# Project Instructions

These instructions apply to every AI-assisted task in this repository.

## AI Usage Rule

- Any meaningful AI use must be recorded in `docs/ai-usage.md`.
- Log the use immediately after planning, writing, debugging, summarizing, or editing with AI help.
- Do not finish a task that used AI without adding a fresh row to the log.
- Use the standard fields: `Date`, `Tool`, `Purpose`, `What was asked`, and `Human review`.

## Work Style

- Keep work reproducible and traceable.
- Prefer small, documented changes over hidden or implicit edits.
- When AI contributes to a decision, note the decision in `docs/decisions.md` if it affects the project direction.

## Project Scope

- This project is an experimental fake news detection study.
- The main comparison is TF-IDF versus transformer-based sentence embeddings.
- All final metrics must come from real runs, not generated values.
