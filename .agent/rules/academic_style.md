# Academic Style Rules

## Language

- All code, comments, markdown cells, and documentation must be written in **English**
- Use formal, concise academic prose — avoid colloquialisms and filler words

## Notebook Structure

- Every notebook must begin with a **title** (H1) and a one-sentence **purpose statement**
- Organize content into numbered sections with clear **H2 headers** (e.g., `## 1. Data Loading`)
- Each section should have a brief markdown introduction before any code

## Code Style

- Use inline comments sparingly; prefer self-documenting variable names
- Keep code cells focused: one logical operation per cell
- Remove all trial-and-error / debugging code before finalizing

## Figures & Plots

- Use `scienceplots` with `['science', 'ieee']` style for publication-quality figures
- Every plot must have labeled axes and a descriptive title
- Use consistent color palettes across all figures

## Configuration

- No hardcoded paths — all paths and parameters come from `config.yaml`
- Use `src/config.py` dataclass for typed, validated access to configuration
