# Contributing

From the repository root, install the development dependencies and run the CI checks:

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest tests/ -q -n auto --maxprocesses=4
```

A bug fix should have a small regression test of observable behavior. Extend an
existing case when it already covers the same path; avoid separate tests for trivial
defaults, getters, or repeated shapes. Keep public APIs typed. Comments should
explain constraints, numerical choices, or non-obvious behavior.

## Tests

The default suite excludes `slow` and `gpu` tests. To select them:

```bash
pytest tests/ -m slow
pytest tests/ -m gpu
pytest tests/ -m ""
```

Capping parallel test workers at four limits startup and memory costs. The fast
suite uses one CPU thread per worker; `POLYSTEP_TEST_THREADS` overrides it. Slow
tests use `ncpu - 8` threads, clamped to 2-16 and split across xdist workers, and may
download datasets. Keep downloads and training-quality checks in the slow suite;
use synthetic inputs for unit tests.

Tests importing `experiments/` must skip when that directory is absent, since it is
excluded from distributions. Install `pip install -e ".[dev,experiments]"` to run
tests that need optional experiment dependencies. Missing optional dependencies and
unavailable hardware are valid reasons to skip a test; unsupported parameter pairs
should not be collected. Tests run without automatic retries.

Pushes and pull requests test Python 3.11 with PyTorch 2.8 and Python 3.14 with
current PyTorch. Scheduled and manual runs add Python 3.12/3.13, coverage, optional
experiment dependencies, real-data learning checks, and the shipped tests against
the installed wheel. Every run builds both distributions and checks the wheel's
imports and type marker.

## Releases

Update `polystep.__version__`. Set the same version and the release date in
`CITATION.cff`. Add a `CHANGELOG.md` entry with the heading `## X.Y.Z - YYYY-MM-DD`.
`pyproject.toml` reads the package version automatically.
If the project's maturity changes, update its `Development Status` classifier.

```bash
uv build
uvx twine check --strict dist/*
```

After the full manual CI workflow passes on the release commit, publish a GitHub
Release tagged `vX.Y.Z`. The release workflow checks versions, lint, metadata, and
the installed wheel before publishing. Only the publish job has OIDC permission;
it uses the `pypi` environment and PyPI Trusted Publishing.

## Issues

Include a minimal reproduction, the traceback, Python and PyTorch versions, and
hardware details relevant to the failure.
