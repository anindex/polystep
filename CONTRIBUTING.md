# Contributing

Install the development dependencies. Then run the CI checks:

```bash
pip install -e ".[dev]"
ruff check .
ruff format --check .
pytest tests/ -q -n auto
```

Work on a branch. Add a regression test when the change needs one.
Open a pull request after the checks pass. Test observable behavior.
Keep one small regression test per bug.
Extend an existing case when it already covers the same path; avoid separate tests
for trivial defaults, getters, or repeated shapes. Keep comments specific and public
APIs typed.

## Tests

The default suite excludes `slow` and `gpu` tests. To select them:

```bash
pytest tests/ -m slow
pytest tests/ -m gpu
pytest tests/ -m ""
```

The fast suite uses one CPU thread; `POLYSTEP_TEST_THREADS` overrides it. Slow tests
use `ncpu - 8` threads, clamped to 2-16 and split across xdist workers, and may
download datasets. Keep downloads and training-quality checks in the slow suite; use
synthetic inputs for unit tests.
Tests importing `experiments/` must skip when that directory is absent, since it is
excluded from distributions.

Pushes and pull requests test Python 3.11 with PyTorch 2.8 and Python 3.14 with
current PyTorch. Scheduled and manual runs add Python 3.12/3.13, coverage, real-data
learning checks, and the shipped tests against the installed wheel. Every run
builds both distributions and checks the wheel's imports and type marker.

## Releases

Update `polystep.__version__`. Set the same version and the release date in
`CITATION.cff`. Add a `CHANGELOG.md` entry with the heading `## X.Y.Z - YYYY-MM-DD`.
`pyproject.toml` reads the package version automatically.
If the project's maturity changes, update its `Development Status` classifier.

Build the distributions. Then check their metadata:

```bash
uv build
uvx twine check --strict dist/*
```

Run the full manual CI workflow on the release commit.
This covers the Python matrix and repository-only experiment tests.
After CI passes, publish a GitHub Release with the tag `vX.Y.Z`.
This triggers the PyPI workflow. The workflow checks version consistency, lint, and
package metadata. It tests the installed wheel before it publishes the package.
Only the separate publish job has OIDC permission; it uses the `pypi` environment
and PyPI Trusted Publishing.

## Issues

Include a minimal reproduction, the traceback, Python and PyTorch versions, and
hardware details relevant to the failure.
