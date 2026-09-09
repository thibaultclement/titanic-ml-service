# Titanic ML Service — Agent Instructions

## Project context

This is an existing Data Science / Machine Learning project based on the Kaggle Titanic dataset.

The repository already contains:
- a Data Science and feature-engineering pipeline;
- a serialized sklearn model;
- FastAPI inference endpoints;
- a Streamlit frontend;
- SHAP / PDP / ICE explainability;
- CLI scripts;
- deployment configuration managed externally;
- notebooks used for experimentation.

Do not rebuild the project from scratch.

## Working principles

- Prefer small, focused, reviewable changes.
- Do not perform unrelated refactoring or cleanup.
- Do not silently change existing behavior.
- Explain important behavioral or architectural changes before applying them.
- Preserve the existing project structure unless a task explicitly requires changing it.

## Model and data

- `models/model.joblib` is the current frozen model artifact.
- Do not retrain, replace, regenerate, or modify the model artifact unless explicitly requested.
- Do not modify raw Titanic data unless explicitly requested.
- Do not change feature semantics without explicit approval.
- Keep training/evaluation logic clearly separated from online inference logic.
- Treat training/inference parity as an important project constraint.

## Python and dependencies

- Use `uv` for dependency management.
- Do not use `pip` directly unless explicitly requested.
- Preserve the existing `src/` package layout.
- Prefer imports through the installed `titanic` package rather than relying on the repository working directory.

## Scope control

- Inspect only the files needed for the current task when possible.
- Do not broaden the scope of a task without explaining why.
- Avoid speculative improvements unrelated to the requested change.
- If you discover an important issue outside the current scope, report it instead of fixing it automatically.

## Validation

After relevant Python code changes:
- run the relevant tests;
- run `ruff check --no-cache .`;
- report any failing checks clearly.

Do not modify code solely to make unrelated existing failures disappear.

## Git

- Keep changes easy to review.
- Before finishing, summarize the files changed and the purpose of each change.
- Do not create commits unless explicitly requested.
