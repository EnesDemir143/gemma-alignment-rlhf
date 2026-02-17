# Project Environment Rules

## Package Manager
- **uv** is used as the package manager (not pip directly)
- Install packages: `uv pip install <package> --python .venv/bin/python`
- Add to project: `uv add <package>`

## Python Environment
- Virtual environment: `.venv/` (project root)
- Python binary: `.venv/bin/python` (Python 3.13)
- Run scripts: `.venv/bin/python -m src.<module>` (from project root)
- **Never use system python** — always use the venv binary

## Project Structure
- Source code: `src/`
- Data: `data/raw/`, `data/processed/`
- Notebooks: `notebooks/`
- Docs: `docs/`

## Language
- Code comments and docstrings: **English**
- User-facing print messages: **Turkish** (emoji prefixed)
