# Contributing

From the repository root, install the development dependencies and run the CI checks:

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest tests/ -q -n auto --maxprocesses=4
```

Include a small regression test with each bug fix. Extend an existing case when it
already covers the same path; avoid separate tests for trivial defaults, getters, or
repeated shapes. Keep public APIs typed. Comments should explain constraints,
numerical choices, or non-obvious behavior.

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

Tests importing `experiments/` or `examples/` must skip when that directory is absent,
since both are excluded from distributions. Install `pip install -e ".[dev,experiments]"` to run
tests that need optional experiment dependencies. Missing optional dependencies and
unavailable hardware are valid reasons to skip a test; unsupported parameter pairs
should not be collected. Tests run without automatic retries.

Pushes and pull requests test Python 3.11 with PyTorch 2.8 and Python 3.14 with
current PyTorch. Weekly and manual runs also test Python 3.12/3.13, coverage, and
optional experiment dependencies. They run real-data learning checks and test the
installed wheel with the shipped tests. Every run builds both distributions and
checks the wheel's imports and type marker.

## Releases

Update `polystep.__version__`. Set the same version and the release date in
`CITATION.cff`. Put the new `CHANGELOG.md` entry first, with the heading
`## X.Y.Z - YYYY-MM-DD`. Use the same date as `CITATION.cff`.
`pyproject.toml` reads the package version automatically.
If the project's maturity changes, update its `Development Status` classifier.

```bash
uv build
uvx twine check --strict dist/*
```

Check the pinned action versions before each release. Dependabot is disabled.

Push the release commit and tag `vX.Y.Z`. Run the full manual CI workflow on that
commit. After it passes, publish the GitHub Release with the changelog entry as its
notes. The release workflow checks the version, date, lint, metadata, and installed
wheel before publishing. Only the publish job can request an OIDC token. It uses
the `pypi` environment and PyPI Trusted Publishing.

## Issues

Include a minimal reproduction, the traceback, Python and PyTorch versions, and
hardware details relevant to the failure.
