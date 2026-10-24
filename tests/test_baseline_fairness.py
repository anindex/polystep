"""Evaluation budgets, search representations, seed control, and baseline behavior."""

from __future__ import annotations

import os

import pytest
import torch

# experiments/ is the paper reproduction harness and is not part of the distributed
# package, so this module is unimportable when the tests run from an sdist.
pytest.importorskip("experiments.runners")


def test_fairness_table_matches_budgets_representation_and_trajectory():
    """One run per method checks matched budgets, search spaces and validation-only trajectories."""
    pytest.importorskip("cma", reason="pycma drives the CMA-ES baseline")
    import torch.nn as nn

    from experiments.runners.fairness import FAIR_METHODS, make_subspace, run_baseline
    from polystep.cost_nn import NNCostEvaluator
    from polystep.transform import ParamLayout

    budget, rank, seed = 256, 4, 0
    torch.manual_seed(seed)
    X, Y = torch.randn(8, 6), torch.randint(0, 3, (8,))

    table = {}
    for method in FAIR_METHODS:
        torch.manual_seed(seed)
        model = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))
        layout = ParamLayout.from_module(model)
        evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
        table[method] = run_baseline(
            method,
            model=model,
            layout=layout,
            loss_batch=lambda stacked: evaluator.evaluate(stacked, X, Y),
            budget=budget,
            val_fn=lambda m: -float(nn.functional.cross_entropy(m(X), Y)),
            test_fn=lambda m: -float(nn.functional.cross_entropy(m(X), Y)),
            mode="max",
            seed=seed,
            subspace=make_subspace(layout, rank=rank, seed=seed, method=method),
            subspace_rank=rank,
            probe_scale=0.5,
            log_points=8,
        )
    budgets = {m: r["hyperparameters"]["eval_budget"] for m, r in table.items()}
    assert set(budgets.values()) == {budget}, f"budgets differ across the table: {budgets}"
    for method, r in table.items():
        used = r["hyperparameters"]["evals_used"]
        assert used == r["metrics"]["function_evals"] <= budget, f"{method} used {used} of {budget}"
    ranks = {m: r["hyperparameters"]["subspace_rank"] for m, r in table.items()}
    assert set(ranks.values()) == {rank}, f"ranks differ across the table: {ranks}"

    classes = {m: r["hyperparameters"]["subspace_class"] for m, r in table.items()}
    # EGGROLL's rank-r A B^T needs matrix-structured coordinates, which HybridSubspace
    # does not have; FactoredSubspace is that parameterization. The exception is
    # allowed only because it is recorded in the JSON, which is what this asserts.
    assert classes.pop("eggroll") == "FactoredSubspace", classes
    assert set(classes.values()) == {"HybridSubspace"}, f"classes differ across the table: {classes}"
    for method, r in table.items():
        traj = r["step_logs"]
        assert len(traj) >= 2, f"{method} recorded {len(traj)} trajectory points"
        evals = [p["evals"] for p in traj]
        assert evals == sorted(evals), f"{method} trajectory is not monotone in evals"
        assert evals[-1] <= budget
        assert all("val_accuracy" in p for p in traj)
        # Validation curve only: a per-point test score would leak test into the paper's figure.
        assert not any("test_accuracy" in p for p in traj), f"{method} logs test per point"
        assert all(p.get("scored_at") in {"iterate", "best_x"} for p in traj)


def test_eval_budget_matches_polystep_spend():
    """The shared budget is derived from PolyStep, so it must equal what it spends."""
    import torch.nn as nn

    from experiments.runners.fairness import make_subspace, polystep_eval_budget
    from polystep.optimizer import PolyStepOptimizer
    from polystep.transform import ParamLayout

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 8), nn.Linear(8, 3))
    subspace = make_subspace(ParamLayout.from_module(model), rank=4, seed=0)
    opt = PolyStepOptimizer(model, subspace=subspace, solver="softmax", seed=0, num_probe=1)

    steps = 3
    spent = 0

    def closure(stacked):
        nonlocal spent
        n = next(iter(stacked.values())).shape[0]
        spent += n
        return torch.randn(n)

    for _ in range(steps):
        opt.step(closure)
    assert polystep_eval_budget(opt, steps) == spent


def test_theory_mode_selects_analysed_config():
    """Jitter on, schedules off, orthoplex, no acceleration -- and the tuned config
    is left alone, because the point is to report the gap between the two."""
    from experiments.runners.fairness import THEORY_GAMMA, THEORY_JITTER, apply_theory_mode
    from polystep.epsilon import PowerDecay

    tuned = {
        "rank": 8,
        "epsilon_init": 10.0,
        "epsilon_target": 0.1,
        "step_radius_init": 5.0,
        "step_radius_target": 1.0,
        "probe_radius_init": 10.0,
        "probe_radius_target": 2.0,
        "amortize_steps": 3,
        "amortize_ema": 0.7,
        "use_momentum": True,
        "biased_rotation": True,
    }
    theory = apply_theory_mode(tuned)

    assert tuned["amortize_steps"] == 3, "apply_theory_mode must not mutate its input"
    assert theory["probe_radius_jitter"] == THEORY_JITTER
    assert theory["probe_radius_jitter_dist"] == "smooth"
    assert theory["polytope_type"] == "orthoplex"
    assert theory["biased_rotation"] is False
    assert theory["use_momentum"] is False
    assert theory["amortize_steps"] == 1 and theory["anderson_depth"] == 0
    # Flat epsilon, flat probe radius, decaying step radius r_0 (t+1)^-(1/2+gamma).
    # Assert radii in physical units: a bare ``probe_radius`` is an epsilon
    # multiplier, while a ``*_init`` pair is the physical radius itself.
    from polystep.epsilon import resolve_radius

    assert theory["epsilon"] == 0.1 and "epsilon_init" not in theory
    assert "probe_radius_init" not in theory and "probe_radius_target" not in theory
    assert resolve_radius(theory["probe_radius"], 0, theory["epsilon"]) == pytest.approx(2.0)
    assert theory["probe_radius_realized"] == pytest.approx(2.0)
    r = theory["step_radius"]
    assert isinstance(r, PowerDecay) and r.gamma == THEORY_GAMMA
    assert resolve_radius(r, 0, theory["epsilon"]) == pytest.approx(5.0)
    assert r.at(3) == pytest.approx(5.0 * 4 ** -(0.5 + THEORY_GAMMA))


def test_theory_mode_keeps_scalar_physical_radii():
    """A scalar radius is an epsilon multiplier and must transplant at the same physical radius."""
    from experiments.runners.fairness import apply_theory_mode
    from polystep.epsilon import resolve_radius

    tuned = {"epsilon": 0.5, "step_radius": 1.0, "probe_radius": 1.0}
    theory = apply_theory_mode(tuned)
    eps = theory["epsilon"]
    assert resolve_radius(theory["step_radius"], 0, eps) == pytest.approx(0.5)
    assert resolve_radius(theory["probe_radius"], 0, eps) == pytest.approx(0.5)


def test_budget_charges_only_ot_steps():
    """``amortize_steps=a`` means PolyStep is charged for probes on 1 step in ``a`` (ceiled)."""
    from experiments.runners.fairness import polystep_eval_budget

    class _Fake:
        def __init__(self, a):
            self.amortize_steps = a
            self._state = type("S", (), {"X": torch.zeros(4, 2)})()
            self._polytope_vertices = torch.zeros(3, 2)
            self.num_probe = 1

    assert polystep_eval_budget(_Fake(1), 30) == 12 * 30
    assert polystep_eval_budget(_Fake(3), 30) == 12 * 10
    # ceil: 31 steps at a=3 is 11 OT steps, not 10.
    assert polystep_eval_budget(_Fake(3), 31) == 12 * 11


def test_probe_scale_is_per_coordinate():
    """PolyStep's radius is a displacement norm; a baseline's sigma is per-coordinate.

    ``dim`` divides it out so both probes have the same displacement norm.
    """
    import math

    from experiments.runners.fairness import probe_scale_of

    cfg = {"probe_radius": 2.0}
    dim = 1242
    assert probe_scale_of(cfg, dim=dim) == pytest.approx(2.0 / math.sqrt(dim))
    # A Gaussian at the returned sigma has the radius back as its norm.
    assert probe_scale_of(cfg, dim=dim) * math.sqrt(dim) == pytest.approx(2.0)
    # A scheduled radius transfers its floor, still per-coordinate.
    sched = {"probe_radius_init": 10.0, "probe_radius_target": 2.0}
    assert probe_scale_of(sched, dim=dim) == pytest.approx(2.0 / math.sqrt(dim))
    # ...and epsilon must NOT be applied to a schedule: the runner reads an
    # init/target pair as the physical radius.
    sched_eps = {"probe_radius_init": 10.0, "probe_radius_target": 2.0, "epsilon_target": 0.1}
    assert probe_scale_of(sched_eps, dim=dim) == pytest.approx(2.0 / math.sqrt(dim))
    # A *scalar* radius is a multiplier on epsilon, and that rule still holds.
    assert probe_scale_of({"probe_radius": 2.0, "epsilon": 0.5}, dim=dim) == pytest.approx(1.0 / math.sqrt(dim))
    # No dim means no division, for callers with no dimension to hand.
    assert probe_scale_of(cfg) == 2.0


def test_reseed_loaders_makes_repeated_iteration_reproducible():
    """A shared loader must give every run the same minibatch stream.

    ``seeded_loader_kwargs`` seeds each loader's generator once, at construction, so
    the shuffle order stops depending on global RNG consumption. It keeps depending on
    how many times the loader has already been iterated, because the generator advances
    per epoch. A sweep that builds its loaders once and reuses them across configurations
    therefore trains cell *i* on the *i*-th stream and ranks configurations partly by
    their position in the sweep.
    """
    from torch.utils.data import DataLoader, TensorDataset

    from experiments.runners.common import reseed_loaders
    from polystep.benchmarks.utils import seeded_loader_kwargs

    data = TensorDataset(torch.arange(64).unsqueeze(1).float(), torch.zeros(64))
    loader = DataLoader(data, batch_size=8, shuffle=True, **seeded_loader_kwargs(42))

    def order():
        return torch.cat([x.flatten() for x, _ in loader]).tolist()

    first, second = order(), order()
    assert first != second, "expected the generator to advance between epochs; without that this test asserts nothing"

    reseed_loaders(42, loader)
    assert order() == first
    reseed_loaders(42, loader)
    assert order() == first


def test_reseed_loaders_accepts_none_and_unshuffled():
    """Runners pass val/test loaders that may be None or unshuffled."""
    from torch.utils.data import DataLoader, TensorDataset

    from experiments.runners.common import reseed_loaders

    plain = DataLoader(TensorDataset(torch.zeros(4, 1), torch.zeros(4)), batch_size=2)
    reseed_loaders(7, None, plain)


def test_eggroll_factors_over_factored_coordinates_only():
    """EGGROLL's low-rank sampler needs ``shapes`` to say where the matrices are.

    ``FactoredSubspace``'s projected coordinates *are* a ``(d_out, rank)`` matrix;
    ``HybridSubspace``'s are a QR'd Gaussian projection with no matrix structure. Both
    mark projected specs with ``is_projected``, so reporting every projected block as a
    matrix would make EGGROLL factor Hybrid coordinates that do not factor, and
    reporting none of them would make the EGGROLL arm run dense Gaussian ES while
    still paying FactoredSubspace's dimension penalty.
    """
    import math

    import torch.nn as nn

    from polystep.baselines.core import Objective
    from polystep.baselines.methods import _lowrank_noise
    from polystep.factored_subspace import FactoredSubspace
    from polystep.hybrid_subspace import HybridSubspace
    from polystep.transform import ParamLayout

    model = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 10))
    layout = ParamLayout.from_module(model)
    base = {k: v.detach() for k, v in model.state_dict().items()}

    def loss_batch(sd):
        return torch.zeros(next(iter(sd.values())).shape[0])

    factored = FactoredSubspace.from_layout(layout, rank=4, seed=0)
    obj = Objective.from_subspace(factored, base, loss_batch, budget=100)
    assert obj.shapes[0] == (32, 4) and obj.shapes[2] == (10, 4)
    assert obj.shapes[1] == (32,) and obj.shapes[3] == (10,), "1D entries stay flat"

    hybrid = HybridSubspace.from_layout(layout, rank=4, seed=0)
    flat = Objective.from_subspace(hybrid, base, loss_batch, budget=100)
    assert all(len(s) == 1 for s in flat.shapes), "Hybrid coordinates do not factor"

    # And the sampler actually produces rank-<=1 blocks on the factored objective.
    gen = torch.Generator().manual_seed(0)
    E = _lowrank_noise(obj.shapes, 1, 4, obj.dim, gen, torch.device("cpu"), torch.float32)
    off = 0
    for shape in obj.shapes:
        n = math.prod(shape)
        if len(shape) >= 2:
            assert torch.linalg.matrix_rank(E[0, off : off + n].reshape(shape)).item() <= 1
        off += n


def test_wallclock_deadline_stops_every_method():
    """A deadline stops a run through ``remaining``, without touching any method."""

    from polystep.baselines.core import Objective

    obj = Objective(lambda X: torch.zeros(X.shape[0]), 4, budget=10**9, deadline_s=0.2)
    assert obj.elapsed_s == 0.0, "clock starts on the first call, not at construction"
    while obj.remaining > 0:
        obj(torch.zeros(2, 4))
    assert obj.out_of_time and obj.remaining == 0 and obj.evals > 0
    evals = obj.evals
    obj(torch.zeros(2, 4))
    assert obj.evals == evals + 2, "a generation that started before the deadline finishes"


def _spsa_run(**kwargs):
    import torch.nn as nn

    from experiments.runners.fairness import run_baseline
    from polystep.cost_nn import NNCostEvaluator
    from polystep.transform import ParamLayout

    torch.manual_seed(0)
    X, Y = torch.randn(8, 6), torch.randint(0, 3, (8,))
    model = nn.Sequential(nn.Linear(6, 8), nn.Tanh(), nn.Linear(8, 3))
    evaluator = NNCostEvaluator(model, loss_fn=nn.CrossEntropyLoss())
    out = run_baseline(
        "spsa",
        model=model,
        layout=ParamLayout.from_module(model),
        loss_batch=lambda stacked: evaluator.evaluate(stacked, X, Y),
        budget=200,
        val_fn=lambda m: -float(nn.functional.cross_entropy(m(X), Y)),
        test_fn=lambda m: -float(nn.functional.cross_entropy(m(X), Y)),
        probe_scale=0.5,
        log_points=4,
        **kwargs,
    )
    return out["hyperparameters"], [p["loss"] for p in out["step_logs"]]


def test_spsa_gain_offset_follows_polystep_steps():
    """Without it, budgeted runs keep the library's ``budget // 2`` offset."""
    generic_hp, generic = _spsa_run()
    assert "A" not in generic_hp
    assert _spsa_run(hp={"A": 0.1 * (200 // 2)})[1] == generic
    assert _spsa_run(polystep_steps=100)[1] == generic

    steps_hp, steps = _spsa_run(polystep_steps=7)
    assert steps_hp["A"] == pytest.approx(0.7)
    assert steps != generic
    assert _spsa_run(polystep_steps=7, hp={"A": 3.0})[0]["A"] == 3.0


def test_spsa_deadline_requires_steps_or_A():
    """A deadline gives no iteration count to derive ``A`` from."""
    with pytest.raises(ValueError, match="polystep_steps"):
        _spsa_run(deadline_s=60.0)
    assert _spsa_run(deadline_s=60.0, polystep_steps=7)[0]["A"] == pytest.approx(0.7)
    assert _spsa_run(deadline_s=60.0, hp={"A": 3.0})[0]["A"] == 3.0


def test_progress_line_prints_validation_score(capsys):
    """The evaluation-budget progress line reads the ``val_`` key the trajectory stores."""
    _spsa_run()
    assert "accuracy=" in capsys.readouterr().out


def test_matched_budget_records_used_axis():
    """Each of the three axes returns its own name, and wall-clock needs a PolyStep run."""
    import json as _json
    import tempfile

    from experiments.runners.fairness import budget_for_method, matched_budget

    assert budget_for_method("openai_es", 10_000, 100, showcase="snn")[1] == "evals"
    assert budget_for_method("spsa", 10_000, 100, showcase="snn")[1] == "steps"
    assert budget_for_method("openai_es", 10_000, 100, showcase="snn", match="wallclock")[1] == "wallclock"

    with tempfile.TemporaryDirectory() as d:
        with pytest.raises(FileNotFoundError):
            matched_budget(
                "openai_es",
                showcase="argmax",
                seed=42,
                polystep_steps=100,
                eval_budget=10_000,
                results_dir=d,
                match="wallclock",
            )
        with open(os.path.join(d, "argmax_polystep_42.json"), "w") as fh:
            _json.dump({"metrics": {"wall_time_seconds": 1234.5}}, fh)
        budget, axis, deadline = matched_budget(
            "openai_es",
            showcase="argmax",
            seed=42,
            polystep_steps=100,
            eval_budget=10_000,
            results_dir=d,
            match="wallclock",
        )
        assert axis == "wallclock" and deadline == 1234.5 and budget > 10**9
