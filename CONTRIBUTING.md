# Contributing

Install the development dependencies and run the same checks as CI:

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest tests/ -q
```

Create a branch, make the change, add regression coverage where needed, and open a
pull request. Keep comments specific and public APIs typed.

## Tests

The default suite excludes `slow` and `gpu` tests. To select them:

```bash
pytest tests/ -m slow
pytest tests/ -m gpu
pytest tests/ -m ""
```

Tests use one CPU thread by default; `POLYSTEP_TEST_THREADS` overrides it. Slow tests
may download MNIST. Tests importing `experiments/` must skip when that directory is
absent, since it is excluded from distributions.

Pushes and pull requests test Python 3.11 with PyTorch 2.8 and Python 3.14 with
current PyTorch. Scheduled and manual runs add Python 3.12/3.13, coverage, real-data
learning checks, and the shipped tests against the installed wheel. Every run
builds both distributions and checks the wheel's imports and type marker.

Before a release:

```bash
uv build
uvx twine check --strict dist/*
```

## Releases

Update `polystep.__version__`, the version and date in `CITATION.cff`, and
`CHANGELOG.md`. `pyproject.toml` reads the package version automatically.

Publishing a GitHub Release tagged `vX.Y.Z` triggers the PyPI workflow. It checks
version consistency, lint, tests, and package metadata before publishing. The
release also runs the shipped tests against the wheel. Only the separate publish
job has OIDC permission; it uses the `pypi` environment and PyPI Trusted Publishing.

## Issues

Include a minimal reproduction, the traceback, Python and PyTorch versions, and
hardware details relevant to the failure.
