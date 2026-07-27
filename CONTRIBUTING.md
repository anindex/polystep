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
3. Run the checks below; they mirror what CI runs, so a green local run means a green CI run
4. Submit a pull request

## Running the checks

```bash
# Lint and formatting, exactly as CI runs them
ruff check src/polystep/ tests/ examples/ experiments/scripts/
ruff format --check src/polystep/ tests/ examples/ experiments/scripts/

# Fast tests (recommended during development). The pyproject addopts deselect the
# slow tests; the gpu-marked ones are still collected and self-skip without CUDA.
pytest tests/ -v

# Exactly what CI selects, in parallel. About 9s.
pytest tests/ -m "not slow and not gpu" -n auto

# Every test, including the ones marked slow. These train on MNIST and take 15-30 min.
pytest tests/ -v -m ""

# With coverage
pytest tests/ --cov=polystep --cov-report=term-missing
```

`conftest.py` pins `torch.set_num_threads(1)` for everything except the `slow` tests.
The fast tests are small-tensor bound, where torch's intra-op pool costs more than it
saves. Unpinned the suite is both slower and erratic, which was tripping the 60s timeout
on the Sinkhorn-heavy tests; pinned it runs in about 9s. The `slow` tests train at batch
512 and get the machine default back. Override with `POLYSTEP_TEST_THREADS`.

CI also builds the sdist and runs the shipped test suite from the unpacked tarball. To
reproduce that:

```bash
uv build --sdist
mkdir -p /tmp/sdist && tar xzf dist/*.tar.gz -C /tmp/sdist --strip-components=1
cd /tmp/sdist && pip install ".[dev]" && pytest tests/ -q
```

Tests that read or import `experiments/` request the `require_experiments` fixture, which
skips them outside a repo checkout.

## Code Style

- Follow existing code conventions
- Use type hints for public functions
- Add docstrings with Args/Returns sections for public APIs
- Keep comments short and specific; explain why, not what

The package ships `py.typed`, so annotations on the public surface are part of the
contract. `mypy src/polystep/` currently reports errors, mostly narrowing complaints on
the deliberately duck-typed `subspace` argument, so it is not yet a CI gate. New code
should not add to the count.

## Reporting Issues

Please include:
- Python and PyTorch versions
- GPU model (if relevant)
- Minimal reproduction script
- Full error traceback
