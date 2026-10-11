# Changelog

## 1.0.0 (unreleased)

- Remove `CMAAdaptiveSubspace`, `compute_cma_hyperparameters`,
  `use_covariance_adaptation`, and the CMA fields of `SolverState`; use
  `AdaptiveSubspace`.
- `PolyStepOptimizer` raises `TypeError` for a subspace that is not an
  `AdaptiveSubspace`, `HybridSubspace`, or `FactoredSubspace`.
- Remove `adaptive_probes` and `adaptive_probes_threshold`. Runs that passed a
  constant `objective_token` now evaluate candidates on every OT step instead of
  replaying cached probe costs. Amortized momentum steps still skip evaluation.
- `objective_token` now only decides whether the pending trust-region comparison
  still applies.
- Remove `TemperedSoftmaxSolver`, `solver='tempered_softmax'`, and
  `tempered_softmax_tau`; use `solver='softmax'` with `ent_epsilon` for a fixed
  temperature.
- Remove `data_dependent_init` from `SinkhornSolver` and `PolyStepOptimizer`; cold
  Sinkhorn solves start from zero duals.
- Remove `auto_epsilon` and `auto_epsilon_config`; pass a `ProgressiveEpsilon` as
  `epsilon`. The old defaults were `init=e`, `target=max(0.01, 0.1 * e)`,
  `max_epsilon=5 * e` for a float `e`; the schedule's `init` and `target` with
  `max_epsilon=5 * init` for a `LinearEpsilon`; and `init=1.0`, `target=0.01`,
  `max_epsilon=5.0` for a `CosineEpsilon`.
- `ProgressiveEpsilon` with `ent_epsilon` raises `ValueError` in
  `PolyStepOptimizer` and `PolyStep`.
- `ProgressiveEpsilon` ignores solvers without a convergence threshold, so a
  one-particle `PolyStep` no longer pushes epsilon to `max_epsilon`.
- Remove the `subspace`, `nn_evaluator`, `train_inputs`, and `train_targets` fields
  of `PolyStep` and the `base_params` argument of `PolyStep.init_state`; train
  networks with `PolyStepOptimizer`.
- Remove `TrainConfig.batch_size` and `TrainConfig.log_every`, which `train()` never
  read; the `DataLoader` sets the batch size and `LoggingCallback(log_every=...)` the
  logging interval.
- Remove `create_hybrid_blocks` and the `LayerProjectionSpec` alias; use
  `polystep.subspace.ProjectionSpec`.
- Remove the unused `compile` field of `SoftmaxSolver`, `MinCostGreedySolver`,
  `TopKMeanSolver`, and `KLSoftmaxSolver`.
- `polystep` no longer exports `CompiledFunctions`, `try_compile`, `apply_momentum`,
  `update_radius_multiplier`, `update_stagnation`, `compute_momentum_coefficient`,
  `BlockConfig`, `create_per_layer_blocks`, `create_grouped_blocks`,
  `auto_detect_chunk_size`, `ProjectionSpec`, `create_generator`, `ParamEntry`,
  `compute_cost_matrix`, or `scale_cost_matrix`; import them from `polystep._compiled`,
  `polystep.dynamics`, `polystep.blockwise`, `polystep.cost_nn`, `polystep.subspace`,
  `polystep.transform`, or `polystep.costs`.
- Soft-label cross-entropy on Linear-only models scores on the batched path, so its
  costs can differ from the vmap path in the last bit.
- `FactoredSubspace.apply_perturbation` returns only the subspace's entries, so a
  frozen parameter edited outside the optimizer keeps the edit.
- Steps write into the model's current parameters, so a swapped-in layer trains.
  Call `resync_from_model()` after the swap, or the step overwrites it with the old
  weights.
- Optimizer checkpoints use format 6 and restore `candidate_evals`. Older formats
  raise `ValueError`; rebuild the optimizer from the saved model weights instead.
- Subspace runs with `block_strategy='per_layer'` or `'grouped'` take the monolithic
  step and its fast evaluation paths. Sinkhorn and finite-`lam` KL solves still split
  the particles into up to eight row segments; one-sided solvers run unsplit and warn.
  Seeded runs draw a different random stream.
- In those runs, trust region, the quadratic model, screening, probe-count
  adaptation, and `rank_schedule` now apply instead of being ignored. The adaptive
  radius follows the best-vertex mean, `scale_cost='mean'` and `'max_cost'` use one
  divisor for a one-sided solve instead of one per block, and `biased_rotation` warns
  when it falls back to the transport direction.
- In those runs, a nonfinite candidate gets one penalty for the whole step instead
  of one per block. Sinkhorn and KL row segments each set their own penalty, as they
  set their own scale, since every segment is a separate OT problem. One-sided
  solves recenter the whole cost matrix at once, and `_saturation_log` gets one
  record per step instead of one per block.
- With FP16 or BF16 parameters, `HybridSubspace` and `FactoredSubspace` skip the
  subspace-delta and factored evaluators, and the site-local path builds the changed
  weight and the shared ones from the base weights as `reconstruct_batch` does, so
  candidates are scored on the weights they round to. Factored runs in half precision
  materialize their candidates.
- Remove `create_subspace_blocks`, `split_subspace_to_blocks`, and
  `reassemble_blocks_to_subspace`.
- Remove `split_particles` and `reassemble_blocks` from `polystep.blockwise`.
- An absorb keeps the step's best-vertex mean as the stagnation baseline. Resetting
  it to the full cost mean made the next step look like progress, which shrank the
  adaptive radius.
- Full-space `per_layer` and `grouped` steps raise `ValueError` when the closure does
  not return one loss per candidate.
- With a registered evaluator, those steps also score grouped and padded blocks on
  the site paths. Particles that straddle two parameters or hold padding still go
  to the closure.
- Closure batches in those steps no longer round up to a whole candidate group, so
  they stay within `chunk_size`.
- Those steps drive the adaptive radius with the best-vertex mean, like the
  monolithic step; `state.costs` still records the full cost mean, now averaged on
  the host in double precision.
- A step whose vertex costs are all nonfinite records an infinite cost, reports no
  convergence, leaves the adaptive radius alone, and keeps the previous warm start
  in every driver, instead of reporting the sanitize penalty as a loss. A vertex
  with any nonfinite probe counts as nonfinite.
- Newton refinement runs before momentum, so momentum carries the refined step
  instead of being overwritten by it, and Sinkhorn keeps its warm-start duals. A
  particle whose refinement comes out nonfinite keeps its barycentric step; the
  others stay refined.
- `adaptive_num_probe` defaults to off under orthoplex `newton_refinement`, which
  needs `num_probe>=2`. Passing `True` with it warns.
- The screen calibrates dropped vertices on the finite full-fidelity costs only. A
  nonfinite kept cost used to give every dropped vertex of its particle the penalty.
- With a registered evaluator, every evaluation path screens on the
  `screen_fidelity` slice of the batch; the site-vmap path used to screen on all of
  it. `candidate_evals` and `state.evals` count screen forwards at that fraction on
  every path.
- A `HybridSubspace` or `FactoredSubspace` absorb keeps the current basis instead
  of reverting to the seeded one, unless `absorb_aligned_active` redraws it, and an
  absorb that lands on a rotation step also rotates.
- The trust-region comparison survives absorbs and rotations, which leave the
  weights unchanged, so `AdaptiveSubspace` with its default rotation every step now
  adapts the radius.
- The trust region applies its updated radius on the step that measured the ratio.
  Every update used to land one step late, so a step that raised the loss was
  followed by another at the old radius.
- `PolyStepOptimizer` warns when an `AdaptiveSubspace` that rotates every step is
  combined with `amortize_steps > 1`, `biased_rotation`, or `use_momentum`, whose
  state each rotation clears.
- Sparse `HybridSubspace` projections keep a narrower parameter's dtype instead of
  promoting it to the coordinates' dtype.
- `resync_from_model()` keeps each per-layer projection at its parameter's dtype, so
  a mixed-dtype model keeps training after `train()` restores its best weights.
  Later rotations and `rank_schedule` transitions draw projections at those dtypes
  too. It raises `ValueError` when the model has moved to another device.
- `train()` scores the weights for `restore_best` and callbacks through the
  evaluator: inputs are cast to the model's dtype, `candidate_autocast` applies, and
  an unreduced loss is averaged. FP64 and `mixed_precision` models fed FP32 batches
  no longer fail there.
- Half-precision candidate outputs, from an FP16 or BF16 model or from
  `candidate_autocast`, reduce in FP32 when the loss is an `nn.CrossEntropyLoss`,
  `nn.MSELoss`, or `nn.L1Loss` left at its defaults apart from `reduction`, so
  candidates about 1e-3 apart no longer tie. With `candidate_autocast` on CUDA, the eager in-place and loop costs
  and the `train()` loss now match the compiled path. Other losses and FP64 models
  score as before.
- A `FactoredSubspace` step after `release_evaluator()` scores through the closure
  instead of failing on the released batch. The release also frees the cast copy of
  that batch kept by the registered, delta, and factored evaluators.
- With inputs of more than two dimensions, a Sequential MLP without a leading
  `nn.Flatten` skips the sparse-delta, subspace-delta, and factored evaluators, which
  flatten each sample, so the step no longer fails on a shape mismatch. Full-space
  and hybrid candidates go to the site-local path; factored candidates are
  materialized.
- `PolyStepES` without a `seed` draws rotations from PyTorch's global RNG, so
  `torch.manual_seed` controls it. Every unseeded run used to replay one fixed stream.
- `PowerDecay` requires `0 < gamma <= 0.5`. A larger `gamma` makes the radii
  summable, which the convergence analysis does not allow.
- `CosineEpsilon` raises `ValueError` for a negative `decay`, as `LinearEpsilon` does.
- `rank_schedule` raises `ValueError` at construction unless the subspace is a
  `HybridSubspace`. Other subspaces used to warn and keep their rank.
- `max_subspace_dim` also scales weight matrices the rank leaves at full width.
  They used to count as fixed, so the README quickstart got 132 coordinates and a
  warning instead of 128. The RL configs' cap of 24 now binds; the recorded
  CartPole (114 coordinates, full space) and Acrobot (68) runs used the old split.
- `AdaptiveSubspace` draws its bases on its generator's own device instead of
  reseeding a CPU generator from a CUDA or MPS one, and `HybridSubspace` runs its
  projection QR on the model's device. Seeded runs on non-CPU devices change; CPU
  runs are unchanged.
- `SparseRandomProjection` keeps a single CSR copy with 32-bit indices and, when
  a row map is smaller than full row pointers, stores only the rows its columns touch.
  Projections are unchanged. With 1e8 rows and 256 columns it holds 82 MB instead
  of 902 MB and projects twice as fast on CUDA; on one CPU thread it projects 3 to
  6 times faster at 1e6 to 1e7 rows. `memory_bytes` counts everything it holds, so
  it reads higher than before.
- `fused_softmax_project` in `polystep._compiled` drops its `a` and
  `scale_cost_mean` arguments and returns weights whose rows sum to one; callers
  recenter and scale the cost first. `SolverState` drops the unused marginal `a`.
- `CompiledFunctions` hands Python floats straight to the compiled kernels, which
  recompile at most once for a new radius or epsilon. Seeded `biased_rotation` runs
  on the softmax solver without the quadratic model can change in the last bit.
- `SinkhornSolver(compile=True)` also compiles the iteration when it checks
  convergence, not only when `threshold <= 0`. Each check costs one host sync
  instead of two.
- At `omega=1` without Anderson or adaptive omega, the convergence check reads only
  the row marginal, since the column update leaves the column exact up to rounding.
  `errors` loses that rounding floor, so a solve whose floor sat above `threshold`
  now stops on the row residual.
- `KLSoftmaxSolver` takes `check_every` (default 10) and checks convergence that
  often instead of every `max_iterations // 20` iterations. Budgets of 220 or more are
  checked more often and can stop sooner; budgets under 200 are checked less
  often and can stop later. Seeded KL runs change.
- `SolverResult` drops `cost`, which repeated `ent_reg_cost`, and no longer takes
  `ent_reg_cost` as an argument. That value and
  `KLSoftmaxSolver.last_marginal_violation` are read from the device only when
  accessed, so a solve no longer waits on them. `last_marginal_violation` is now
  a read-only property.
- The barycentre, the direction `biased_rotation` uses without the quadratic
  model, `ess`, and the saturation log divide each transport row by its own sum,
  however small. A row an unconverged Sinkhorn solve left with under 1e-12 of
  mass used to drag its particle toward the origin. An empty or nonfinite row
  counts as uniform: it leaves its particle in place, and `ess` no longer
  reports a huge value for it.
- `min_cost_greedy` and `top_k_mean` in `PolyStepOptimizer` skip any vertex with a
  nonfinite probe. They used to rank it by its finite penalty, so `top_k_mean`
  averaged infeasible vertices into the step when fewer than `k` were feasible, and
  with several probes either solver could pick one. The weighted solvers now give
  such a vertex one penalty, set from the finite vertex costs, instead of averaging
  per-probe penalties into it. Full-space `per_layer` and `grouped` steps record
  that penalty in the loss they return; the monolithic step still records the
  per-probe average, or a penalty when that average overflows.
- `MinCostGreedySolver` and `TopKMeanSolver` rank infeasible vertices as `+inf`, so
  a finite cost at the dtype maximum no longer ties with the penalty and spreads
  mass onto them.
- `SinkhornSolver`, `SoftmaxSolver`, and `KLSoftmaxSolver` compute the
  `scale_cost='mean'` and `'max_cost'` divisors over the finite costs, so the penalty
  for a NaN or infinite cost no longer sets them. The penalty itself is set after
  scaling, so a narrow finite spread cannot inflate it past FP32 precision and break
  the Sinkhorn marginals of an infeasible row. Steps with a nonfinite cost change
  in `PolyStepOptimizer` and `PolyStep` under either mode, and in `PolyStepES`, which
  defaults to `'mean'`.
- With `multifidelity_screen`, a NaN or infinite screen loss no longer enters the
  offset that calibrates the dropped vertices, where it could drag them far below
  the scored ones. A dropped vertex whose screen loss is nonfinite is infeasible.
- `PolyStep` records the mean of the sanitized costs, as `PolyStepOptimizer` does,
  so one infeasible probe no longer ends `run()` early as diverged. The recorded
  cost is `inf` only when every vertex cost is nonfinite, and FP16/BF16 objectives
  record an FP32 mean. A nonfinite particle in `X` still records NaN cost and
  displacement, so `run()` stops as diverged instead of converging on a stuck state.
- `biased_rotation` checks the previous descent direction one particle at a time,
  so a nonfinite row leaves only its own frame unbiased instead of every particle's.
  Checkpoints no longer store `_prev_descent_direction_finite`, and one that has it
  still loads.
- `apply_biased_rotation` turns each frame with one Householder reflection instead
  of a Gram-Schmidt loop, 10 times faster at 8 dimensions and over 100 times at 32
  on CUDA. A bias along another axis of the frame is now applied; it used to fall
  back to the unbiased frame. Seeded `biased_rotation` runs change, orthoplex ones
  included, because the remaining axes differ. Runs without it are unchanged.
- CPU QRs of subspace bases with 1e8 or more `rows * cols**2` work use the whole
  intra-op pool instead of one thread: 1.7 times faster at 50000x64 on 2 threads,
  3 times on 24. Seeded runs with such bases round differently and depend on the
  thread count. Rotation QRs stay on one thread.
- A candidate forward that draws its own random numbers stays batched. vmap retries
  it once with a separate draw per candidate, compiled or not, instead of falling
  back to a sequential loop with a warning. Full-space steps on a small noisy MLP
  run 3.5 times faster on CPU and 40 times on CUDA. Seeded runs of such models draw
  a different random stream.
- Buffers that a refused vmap forward wrote are restored before the retry or the
  sequential fallback runs.
- The sparse-delta and subspace-delta evaluators share one layer walk. Hybrid biases
  and full-width weights now change only the output units their coordinates touch,
  instead of taking a dense correction or the site-local path. Hybrid runs can change
  in the last bit, as can full-space runs whose `particle_dim` of 3 or more puts
  several entries on one output unit. On one CPU thread, hybrid steps and full-space
  steps with a `particle_dim` of 8 run 7 to 15% faster; full-space steps at the
  default `particle_dim` run at the same speed.
- `FactoredSubspace` steps with a registered Sequential MLP score candidates on the
  subspace-delta path, which moves only the output units a candidate's coordinates
  belong to. Runs that span two parameters, screened candidates and the incumbent
  keep the low-rank evaluator. Factored runs can change in the last bit. A
  784-128-10 rank-8 step runs 13 to 25 times faster on one CPU thread. On CUDA at
  the default chunk size it runs 1.5 to 2 times faster with a fifteenth of the peak
  memory. With `chunk_size=72` it needs a quarter to half the peak memory but runs
  1.3 to 1.6 times slower, since such small chunks are bound by kernel launches.
- Those factored steps keep whole particles in each chunk, as hybrid steps already
  do, so `chunk_size` drops to a multiple of the vertex count times `num_probe` and
  never goes below one such group.
- CPU `AdaptiveSubspace` steps with a dense projection build each candidate from the
  incumbent plus the projection columns of the one particle it moves, instead of
  multiplying every candidate's full coordinates. The padded last particle, screened
  candidates, the incumbent, sparse projections and FP16/BF16 runs keep the full
  product. Adaptive runs can change in the last bit. A 784-256-10 rank-128 step runs
  1.1 times faster on one CPU thread with the default rotation every step, and 1.5
  times faster with `rotation_interval=10`. CUDA steps keep the full product, since
  the offsets measured up to 13% slower there. These CPU steps also keep whole
  particles in each chunk, so `chunk_size` rounds as it does for hybrid and factored
  steps.
- CPU adaptive steps with a dense FP32 or FP64 projection and no screen write every
  chunk, the padded particle and the incumbent included, into one candidate buffer
  that the optimizer keeps between steps, so one chunk of candidate weights stays
  allocated. The same 784-256-10 step runs 1.7 times faster with
  `rotation_interval=10` at batch 32, 1.1 times at batch 512, and 1.06 times with
  rotation every step. A closure that keeps a candidate tensor after it returns must
  clone it, as full-space closures already had to.
- With `use_inplace=True` and a registered evaluator, a full-space or
  `HybridSubspace` candidate that changes one parameter rewrites only that
  parameter, and in full space only its own entries, instead of swapping in every
  weight. These candidates are scored with the registered evaluator's loss on the
  registered batch, as on the other registered paths; full-space ones used to go
  through the closure. An evaluator that picks the in-place path for memory takes
  the same route on a model with tied weights. Full-space steps score the same
  weights as before; hybrid steps can change in the last bit. Screened hybrid
  candidates, the hybrid incumbent, FP16/BF16 hybrid weights and non-contiguous
  parameters keep the full swap. A 784-1024-256-10 hybrid step at rank 8 runs 6
  times faster on CUDA, and small CNN and attention hybrid steps 1.5 to 2.3 times
  faster.
- Hybrid candidates on the site-local path with a sparse projection or an
  unprojected parameter now scatter-add only the entries their particle's columns
  touch, instead of projecting a full-width coordinate row. A dense projection fuses
  its offset into the incumbent with one `baddbmm` instead of a separate add. These
  runs can change in the last bit. On one CPU thread, small CNN hybrid steps run 14
  to 18% faster, and scoring candidates of a 32,768-entry weight runs 1.15 times
  faster with a dense projection and twice as fast with a sparse one. On CUDA the
  sparse case runs 1.6 times faster and the rest at the same speed; with a sparse
  projection, repeated CUDA runs match bit for bit only under deterministic
  algorithms.
- Candidates that a delta, site or offsets chunk sends back to materialization are
  scored in batches of at most `chunk_size` instead of a whole particle group.
- Delta and site paths give the padded last particle a chunk of its own, so the
  particles before it stay on those paths instead of falling back with it.
- The automatic chunk size counts the widest layer output, measured by one forward
  when an evaluator is registered, instead of the widest layer's channel count, and
  delta and site chunks no longer charge a full candidate. On CUDA, a one-conv model
  on 256x256 inputs at batch 64 peaked at 6.9 GiB and now peaks at 384 MiB; the
  benchmark's 8x8 CNN steps take 2 to 14 percent longer from the extra chunks.
- FP16/BF16 hybrid steps on the in-place path no longer build an incumbent weight
  set that the path never reads. On CUDA, a step of a BF16 2048-2048-4 MLP at rank 4
  with `use_inplace=True` allocates at most 34 MiB above the model instead of 42 MiB.
- FP16/BF16 hybrid weights round the projected change before adding the base, as
  materialized candidates do, whether they are written in place, written back after
  a step, or folded in by an absorb. One in-place step no longer scores its site and
  swapped candidates on differently rounded weights, and the model ends each step on
  the weights that were scored.
- The baselines' `Objective` returns NaN and infinite losses as `+inf`. Rank and
  z-score shaping give them the lowest utility, and SPSA and MeZO skip a step when
  either loss of the pair is nonfinite. A NaN loss used to get the top rank or turn
  the iterate into NaN.
- `zscore` rescales losses large enough to overflow its mean or variance, so
  float32 losses near `1e38` no longer turn every utility into NaN.
- `Objective` enforces the deadline only through `remaining`, so a generation that
  starts before the deadline finishes instead of raising `BudgetExhausted`.
- `random_search` scores the incumbent and the candidate in the same call, two
  evaluations per step, so both see the same minibatch. It used to compare each
  candidate with the incumbent's loss on an earlier batch, which kept the luckiest
  draw and stalled the search. Random search results change.
- `GymVectorEvaluator` and `CartPoleEvaluator` count `env_steps` only in
  `loss_for_stacked_params`, so evaluation rollouts no longer add to it.
- `GymVectorEvaluator` keeps one vector env per candidate count instead of
  rebuilding it whenever the count changes, and `close()` closes them all.
- `CartPoleEvaluator` takes the same `activation` argument as `GymVectorEvaluator`.
- The Gym `random_policy_baseline` counts reaching the goal before the horizon as
  success on Acrobot and MountainCar. It used to report the timeout rate.
- In the RL runner, ES averages the ranks of tied returns, so a population whose
  returns all tie no longer moves. Every CartPole arm except PPO and DQN runs on
  the analytic simulator. `rl_env_steps` counts training steps only: the real
  steps for PolyStep and ES, `total_timesteps` for PPO and DQN, and 0 for random
  policies. PPO and DQN success on Acrobot means reaching the goal. RL results
  change.
- The experiment runners set SPSA's gain offset `A` to a tenth of PolyStep's step
  count, the schedule the gallery tuning uses, on every match axis. On the
  evaluation and wall-clock axes it came from the evaluation budget or the
  wall-clock cap, so SPSA took far smaller steps than the schedule it was tuned
  on. An SPSA run with a deadline and neither `polystep_steps` nor `A` raises
  `ValueError`. SPSA results on those axes change; the library's `spsa` keeps its
  default.
- `download_file` in `polystep.benchmarks.utils` times out after 60 seconds without
  data and writes through its own temporary file next to the target, so an
  interrupted MNIST download no longer leaves a truncated file that later runs load,
  and two processes fetching the same file no longer collide. The test fixtures use
  the same downloader and readers, so tests run from the sdist no longer skip the
  MNIST checks.
- `VmapSafeMultiHeadAttention` on CUDA skips the fully masked row guard when no mask
  is given. Outputs are unchanged, and unmasked forwards of 16 vmapped candidates
  run 1.4 times faster at sequence length 64 and twice as fast at 128.
- Remove `BenchmarkResult` from `polystep.benchmarks`; nothing used it.
- Remove the unused `experiments/baselines/openai_es.py` and `spsa.py`, which
  `polystep.baselines` replaced, and the flat-parameter helpers, `get_loss_fn`,
  `FunctionEvalCounter`, and Tonic dataset loaders from
  `experiments/runners/common.py`.
- The fairness runner's progress line shows the validation score. It looked up a
  key the trajectory never stores, so the score was never printed.

## 0.13.0 - 2026-09-22

- Fix KL row mass, nonfinite inputs, evaluator hooks, and chunk handling.
- Reuse shared MLP forward work and load only requested MNIST images.
- Fix example checkpoint selection and evaluation budgets; trim demo dependencies.

## 0.12.0 - 2026-09-17

- Fix cross-entropy targets, stale caches, attention masks, nonfinite best values,
  and mixed-dtype evaluation.
- Correct simplex gradients and whole-model trust predictions; migrate checkpoints
  to format 5.
- Reuse MLP prefixes, use native CPU attention, and compile site-local evaluation.
- Average tied baseline ranks; add controlled experiments and forward benchmarks.
- Validate sparse dimensions and factored ranks; simplify documentation.

## 0.11.0 - 2026-08-14

- Remove `LowRankSubspace`, `FactorSpec`, and `LinearSubspace`; use `HybridSubspace`
  or `FactoredSubspace`. Replace `absorb_every` with `absorb_mode`.
- Remove `polystep.benchmarks.baselines`, `evotorch`, `SinkhornSolver.solve(init_eps)`,
  and solver warm-start methods. `CMAAdaptiveSubspace.base` is removed.
- Require `dim` in `probe_scale_of`; correct baseline probe scaling.
- Fix nonfinite costs, zero-mass rows, quadratic evaluation counts, and resume state.
- Add tight-frame quadratic fits and cache radius-jitter sampling.

## 0.10.1 - 2026-08-01

- Require `objective_token` for probe reuse and reject invalid optimizer settings.
- Fix blockwise evaluation, checkpoint aliasing, dtype changes, and tied weights.
- Add site-local evaluation to blockwise mode and reduce subspace allocations.

## 0.10.0 - 2026-07-30

- Default to simplex sampling; retune radii when upgrading.
- Change seeded rotation streams and clear momentum on basis changes.
- Enable forward compilation on the in-place path by default.
- Remove `projection_mode`, `use_csa`, low-level block strategies, and unused exports.
- Move NumPy to optional extras; validate schedules and constructor arguments.

## 0.9.0 - 2026-07-27

- Add sparse-delta evaluation, `FactoredSubspace`, greedy solvers, and evaluator
  registration helpers.
- Fix probe reuse, screening, subspace state, tied weights, and best-weight restoration.
- Extend checkpoint state and reduce projection and evaluation allocations.

## 0.8.0 - 2026-07-24

- Add optimizer checkpointing and forward-compilation controls.
- Fix mixed precision, scheduled epsilon, and invalid solver inputs.

## 0.7.0 - 2026-07-20

- Publish on PyPI and reuse blockwise candidate buffers.
- Fix constant-offset cost stability and biased rotations.

## 0.6.1 - 2026-07-15

- Add the hard decision-tree example.
- Fix mixed precision, barycentric normalization, and candidate memory estimates.
- Cache hybrid projections between rotations.

## 0.6.0 - 2026-07-12

- Add direct F1 optimization and optimizer-variant experiments.
- Fix CMA scaling, solver validation, and trust-region state.

## 0.5.0 - 2026-07-07

- Add `PolyStepES`, `minimize`, and the binary-network example.
- Fix Anderson acceleration, Newton selection, screening, and step adaptation.

## 0.4.0 - 2026-06-21

- Share solver cost handling and remove low-rank Sinkhorn.
- Fix cost scaling, dual recentering, and activation semantics.

## 0.3.0 - 2026-05-27

- Reduce GPU synchronization in solvers and blockwise updates.
- Fix CMA rotation forwarding and API examples.

## 0.2.2 - 2026-05-21

- Use unit-radius vertex generators with keyword-only `radius`.
- Fix CMA transport weighting and stale API examples.

## 0.2.1 - 2026-05-14

- Add the Loihi-style SNN adaptation example.

## 0.2.0 - 2026-05-14

- Require Python 3.11, PyTorch 2.8, and NumPy 2.0 for optional extras.
- Reduce solver synchronization and use inference mode for evaluation.

## 0.1.0 - 2026-05-03

- Initial release: gradient-free training, OT solvers, subspaces, vmap-compatible
  layers, examples, and experiment runners.
