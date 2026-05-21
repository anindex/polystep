# Contributing to polystep

Thank you for your interest in contributing to polystep!

## Getting Started

```bash
git clone https://github.com/anindex/polystep.git
cd polystep
pip install -e ".[dev]"
```

## Development Workflow

1. Create a branch from `main`
2. Make your changes
3. Run tests: `pytest tests/ -v -m "not slow"`
4. Run linting: `ruff check src/polystep/`
5. Submit a pull request

## Running Tests

```bash
# Fast tests only (recommended during development)
pytest tests/ -v -m "not slow"

# Full test suite
pytest tests/ -v

# With coverage
pytest tests/ --cov=polystep --cov-report=term-missing
```

## Code Style

- Follow existing code conventions
- Use type hints for public functions
- Add docstrings with Args/Returns sections for public APIs
- Run `ruff check` before submitting

## Reporting Issues

Please include:
- Python and PyTorch versions
- GPU model (if relevant)
- Minimal reproduction script
- Full error traceback
