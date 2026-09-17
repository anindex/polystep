"""Check two tuned optimizer updates per task on CUDA."""

import gc
import json
import time
import torch
from experiments.runners.run_practical import make_engine, host_data, points, OUT
from experiments.runners.run_controlled import atomic_json


def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    rows = []
    for task in ("snn", "dvs", "int8"):
        data = host_data(task)[0]
        n = 32 if task == "dvs" else 128
        batch = (data[0][:n].cuda().float(), data[1][:n].cuda())
        torch.manual_seed(2026)
        e = make_engine(task, "polystep_tuned", list(points(task, "polystep_tuned"))[13], 2026, 512, True)
        before = {k: p.clone() for k, p in e.params().items()}
        start = time.perf_counter()
        for j in range(2):
            e.progress = 0.5 * j
            count = e.step(batch)
            assert count == e.candidates_per_step
        assert any(not torch.equal(before[k], p) for k, p in e.params().items())
        assert all(torch.isfinite(p).all() for p in e.params().values())
        row = dict(
            task=task,
            candidates=count,
            coordinates=e.active,
            compiled_fallback=e.evaluator._compile_failed,
            two_updates_seconds=time.perf_counter() - start,
            status="passed",
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
        del e
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(OUT / "tuned_gpu_integration.json", rows)


if __name__ == "__main__":
    main()
