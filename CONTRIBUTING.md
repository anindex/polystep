# Contributing to polystep

## Getting Started

```bash
git clone https://github.com/anindex/polystep.git
cd polystep
pip install -e ".[dev]"
```

## Development Workflow

1. Create a branch from `main`
2. Make your changes
3. Run the checks below; they are the lint and test commands CI runs
4. Submit a pull request

## Running the checks

```bash
# Lint and formatting, exactly as CI runs them
ruff check .
ruff format --check .

# Fast tests (recommended during development). The pyproject addopts deselect both
# the slow and the gpu markers, so neither is collected.
pytest tests/ -v

# Exactly what CI selects, in parallel. About 9s.
pytest tests/ -m "not slow and not gpu" -n auto

# Every test, including the ones marked slow. Those train on MNIST; about 90 s total.
pytest tests/ -m ""

# With coverage
pytest tests/ --cov=polystep --cov-report=term-missing
```

`conftest.py` pins `torch.set_num_threads(1)` for everything except the `slow` tests,
which train at batch 512 and get `cpu_count() - 8`. The fast tests are small-tensor
bound, where torch's intra-op pool costs more than it saves; unpinned the suite is both
slower and erratic, which was tripping the timeout on the Sinkhorn-heavy tests. Serial it
runs in about 18 s. `POLYSTEP_TEST_THREADS` overrides the fast count.

CI runs the fast suite on Python 3.11-3.14 per push, and the `slow` marker nightly (or on
`workflow_dispatch`), since those legs download MNIST.

It also builds the sdist, checks that it collects, and runs `tests/test_api.py` from the
unpacked tarball. To reproduce that:

```bash
uv build --sdist
mkdir -p /tmp/sdist && tar xzf dist/*.tar.gz -C /tmp/sdist --strip-components=1
cd /tmp/sdist && uv pip install --system ".[dev]" && pytest tests/test_api.py -q
```

Tests that read or import `experiments/` request the `require_experiments` fixture, which
skips them outside a repo checkout.

## Cutting a release

`release.yml` publishes on a GitHub Release through PyPI trusted publishing. It refuses
to publish unless the tag, `src/polystep/__init__.py`, and `CITATION.cff` all carry the
same version and `CHANGELOG.md` has a section for it, so update all four together:

```bash
# 1. bump __version__ in src/polystep/__init__.py
# 2. bump version and date-released in CITATION.cff
# 3. add a "## X.Y.Z - YYYY-MM-DD" section to CHANGELOG.md
# 4. tag as vX.Y.Z and publish the GitHub Release
```

`pyproject.toml` reads the version from `polystep.__version__`, so it needs no edit.

## Code Style

- Follow existing code conventions
- Use type hints for public functions
- Add docstrings with Args/Returns sections for public APIs
- Keep comments short and specific; explain why, not what

The package ships `py.typed`, so annotations on the public surface are part of the
contract. `mypy src/polystep/` currently reports errors, mostly narrowing complaints on
the duck-typed `subspace` argument, so it is not yet a CI gate. New code should not add
to the count.

## Reporting Issues

Please include:
- Python and PyTorch versions
- GPU model (if relevant)
- Minimal reproduction script
- Full error traceback
