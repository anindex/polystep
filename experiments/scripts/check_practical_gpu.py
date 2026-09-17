"""One-update integration checks for every forward-only task and method."""

import gc
import json
import time
from experiments.runners.run_practical import ForwardRun, host_data, points, OUT
from experiments.runners.run_controlled import atomic_json
import torch


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    records = []
    for task in ("snn", "dvs", "int8"):
        data = host_data(task)[0]
        n = 32 if task == "dvs" else 128
        batch = (data[0][:n].cuda().float(), data[1][:n].cuda())
        for method in ("polystep", "openai_es", "eggroll", "spsa", "mezo", "random_search"):
            if task == "int8" and method not in ("polystep", "mezo"):
                continue
            torch.manual_seed(2026)
            cfg = list(points(task, method))[13]
            if "population" in cfg:
                cfg["population"] = 32
            start = time.perf_counter()
            engine = ForwardRun(task, method, cfg, 2026, 128, False)
            before = engine.z.clone()
            spent = engine.step(batch)
            assert spent == engine.candidates_per_step
            assert torch.isfinite(engine.z).all()
            assert torch.equal(engine.z[engine.active :], before[engine.active :])
            # Candidate scoring must restore the module's original parameters.
            assert all(torch.equal(p, engine.base[k]) for k, p in engine.model.named_parameters())
            row = dict(
                task=task,
                method=method,
                active_dimension=engine.active,
                candidates=spent,
                initialization_and_update_seconds=time.perf_counter() - start,
                displacement_norm=(engine.z - before).norm().item(),
                status="passed",
            )
            records.append(row)
            atomic_json(OUT / "gpu_integration.json", records)
            print(json.dumps(row), flush=True)
            del engine
            gc.collect()
            torch.cuda.empty_cache()
    print("All task/method one-update checks passed", flush=True)


if __name__ == "__main__":
    main()
