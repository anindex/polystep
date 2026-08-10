#!/usr/bin/env python
"""Fine-tune GPT-2 124M on SST-2 with polystep vs an Adam baseline.

Covers full fine-tuning, head-only training, and the hard-quantized head
comparison. ``--tune`` sweeps hyperparameters on the validation split only;
during a sweep the test split is a ``fairness.TestSplitTripwire`` that raises
if anything reads it.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from experiments.runners.common import (
    evaluate_accuracy,
    save_result,
    set_seed,
    track_gpu_memory,
)
from experiments.runners.fairness import (
    DEFAULT_SELECTION_PATH,
    TUNING_GRID,
    TestSplitTripwire,
    apply_point,
    apply_polystep_multipliers,
    load_selection,
    tuning_cost,
    write_selection,
)
from experiments.runners.nondiff_models import BinaryLinear, QuantizedLinear

from polystep.layers import VmapSafeMultiHeadAttention


BENCHMARK = "gpt2_finetune"

GPT2_FINETUNE_CONFIG = {
    "vocab_size": 50257,
    "max_seq_len": 128,
    "embed_dim": 768,
    "num_heads": 12,
    "num_layers": 12,
    "ff_dim": 3072,
    "dropout": 0.0,  # 0 for vmap compatibility
    "num_classes": 2,
}

POLYSTEP_CONFIG = {
    "subspace_dim": 128,
    "step_radius": 2.0,
    "probe_radius": 1.0,
    "epsilon": 0.1,
    "num_probe": 2,
    "chunk_size": 4,
    "sinkhorn_max_iters": 50,
}

ADAM_CONFIG = {
    "lr": 2e-5,
    "epochs": 3,
}

NUM_STEPS = 100
BATCH_SIZE = 8
MAX_TRAIN = 5000
MAX_SEQ_LEN = 128

# Head-only configs: freeze the backbone, train the classifier head only.
HEADONLY_POLYSTEP_CONFIG = {
    "epsilon_init": 5.0,
    "epsilon_target": 0.5,
    "step_radius_init": 2.0,
    "step_radius_target": 0.5,
    "probe_radius_init": 2.0,
    "probe_radius_target": 0.5,
    "num_probe": 1,
    "chunk_size": None,
    "sinkhorn_max_iters": 50,
    "use_momentum": True,
    "momentum_init": 0.5,
    "momentum_final": 0.9,
}
HEADONLY_ADAM_CONFIG = {
    "lr": 1e-3,
    "epochs": 3,
}
HEADONLY_EPOCHS = 50  # epochs, not steps: 1538 params make epoch training cheap
HEADONLY_BENCHMARK = "gpt2_headonly"


class GPT2TransformerBlock(nn.Module):
    """Single pre-norm Transformer block using VmapSafeMultiHeadAttention."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = VmapSafeMultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(approximate="tanh"),  # GPT-2's gelu_new
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]

        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=x.device, dtype=x.dtype),
            diagonal=1,
        )

        normed = self.norm1(x)
        attn_out = self.attention(normed, normed, normed, attn_mask=causal_mask)
        x = x + self.dropout_layer(attn_out)

        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out

        return x


class GPT2Small(nn.Module):
    """GPT-2 Small (124M) with masked mean pooling and a linear classification head."""

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 128,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.0,
        num_classes: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [GPT2TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)]
        )

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if seq_len > self.max_seq_len:
            input_ids = input_ids[:, : self.max_seq_len]
            seq_len = self.max_seq_len
            if attention_mask is not None:
                attention_mask = attention_mask[:, : self.max_seq_len]

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.layer_norm(x)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        logits = self.classifier(x)
        return logits


def load_gpt2_weights(model: GPT2Small, hf_model_name: str = "gpt2") -> dict:
    """Load pretrained HuggingFace GPT-2 weights into GPT2Small.

    Handles Conv1D transposes (HF stores [in, out], nn.Linear [out, in]), the
    fused QKV split, and position-embedding truncation to max_seq_len.
    """
    from transformers import GPT2Model as HF_GPT2

    hf = HF_GPT2.from_pretrained(hf_model_name)
    hf_sd = hf.state_dict()

    mapping = {}

    mapping["token_embedding.weight"] = hf_sd["wte.weight"]  # [50257, 768]
    mapping["position_embedding.weight"] = hf_sd["wpe.weight"][: model.max_seq_len]  # truncate

    mapping["layer_norm.weight"] = hf_sd["ln_f.weight"]
    mapping["layer_norm.bias"] = hf_sd["ln_f.bias"]

    for i in range(12):
        pfx = f"h.{i}"
        lpfx = f"layers.{i}"

        # Fused QKV -> separate Q, K, V (split BEFORE transposing)
        c_attn_w = hf_sd[f"{pfx}.attn.c_attn.weight"]  # [768, 2304]
        c_attn_b = hf_sd[f"{pfx}.attn.c_attn.bias"]  # [2304]
        q_w, k_w, v_w = c_attn_w.split(768, dim=1)  # each [768, 768]
        q_b, k_b, v_b = c_attn_b.split(768, dim=0)  # each [768]

        mapping[f"{lpfx}.attention.W_q.weight"] = q_w.T
        mapping[f"{lpfx}.attention.W_q.bias"] = q_b
        mapping[f"{lpfx}.attention.W_k.weight"] = k_w.T
        mapping[f"{lpfx}.attention.W_k.bias"] = k_b
        mapping[f"{lpfx}.attention.W_v.weight"] = v_w.T
        mapping[f"{lpfx}.attention.W_v.bias"] = v_b

        # Output projection (Conv1D transpose)
        mapping[f"{lpfx}.attention.W_o.weight"] = hf_sd[f"{pfx}.attn.c_proj.weight"].T
        mapping[f"{lpfx}.attention.W_o.bias"] = hf_sd[f"{pfx}.attn.c_proj.bias"]

        # FFN (Conv1D transpose)
        mapping[f"{lpfx}.ff.0.weight"] = hf_sd[f"{pfx}.mlp.c_fc.weight"].T
        mapping[f"{lpfx}.ff.0.bias"] = hf_sd[f"{pfx}.mlp.c_fc.bias"]
        mapping[f"{lpfx}.ff.2.weight"] = hf_sd[f"{pfx}.mlp.c_proj.weight"].T
        mapping[f"{lpfx}.ff.2.bias"] = hf_sd[f"{pfx}.mlp.c_proj.bias"]

        # LayerNorm (direct copy, no transpose)
        mapping[f"{lpfx}.norm1.weight"] = hf_sd[f"{pfx}.ln_1.weight"]
        mapping[f"{lpfx}.norm1.bias"] = hf_sd[f"{pfx}.ln_1.bias"]
        mapping[f"{lpfx}.norm2.weight"] = hf_sd[f"{pfx}.ln_2.weight"]
        mapping[f"{lpfx}.norm2.bias"] = hf_sd[f"{pfx}.ln_2.bias"]

    # Load with strict=False (classifier.weight/bias are randomly initialized for new task)
    missing, unexpected = model.load_state_dict(mapping, strict=False)
    assert set(missing) == {"classifier.weight", "classifier.bias"}, f"Unexpected missing keys: {missing}"
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

    # Free HF model memory
    del hf, hf_sd
    gc.collect()

    return mapping


def get_sst2_gpt2_loaders(
    max_seq_len: int = 128,
    batch_size: int = 8,
    max_train: int = 0,
) -> tuple:
    """Load SST-2 dataset with GPT-2 BPE tokenizer.

    Uses HuggingFace datasets for SST-2 and GPT-2 tokenizer for BPE encoding.
    The GPT-2 tokenizer has no pad token by default; we set it to EOS token.

    Args:
        max_seq_len: Maximum sequence length for tokenization.
        batch_size: Batch size for DataLoaders.
        max_train: Maximum training samples (0 = use all).

    Returns:
        tuple: (train_loader, val_loader) where each yields
            (input_ids, attention_mask, labels) batches.
    """
    from transformers import GPT2Tokenizer
    from datasets import load_dataset

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    # Namespaced: huggingface_hub >= 1.0 rejects the bare "glue" id.
    ds = load_dataset("nyu-mll/glue", "sst2")
    train_texts = list(ds["train"]["sentence"])
    train_labels = list(ds["train"]["label"])
    val_texts = list(ds["validation"]["sentence"])
    val_labels = list(ds["validation"]["label"])

    if max_train > 0:
        train_texts = train_texts[:max_train]
        train_labels = train_labels[:max_train]

    train_enc = tokenizer(
        train_texts,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    val_enc = tokenizer(
        val_texts,
        max_length=max_seq_len,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    train_ds = TensorDataset(
        train_enc["input_ids"],
        train_enc["attention_mask"],
        torch.tensor(train_labels),
    )
    val_ds = TensorDataset(
        val_enc["input_ids"],
        val_enc["attention_mask"],
        torch.tensor(val_labels),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def run_polystep(
    seed: int,
    device: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    results_dir: str,
    num_steps: int = 100,
    subspace_dim: int = 128,
):
    """Fine-tune GPT-2 on SST-2 with polystep + SparseRandomProjection.

    Uses AdaptiveSubspace with random rotation mode and SparseRandomProjection
    for memory-efficient gradient-free fine-tuning of all 124M parameters.

    Args:
        seed: Random seed.
        device: Device string ('cuda' or 'cpu').
        train_loader: Training DataLoader yielding (input_ids, attention_mask, labels).
        test_loader: Validation DataLoader.
        results_dir: Directory to save result JSON.
        num_steps: Total number of optimizer steps (not epochs).
        subspace_dim: Dimensionality of the subspace projection.
    """
    from torch.func import functional_call, vmap

    from polystep.optimizer import PolyStepOptimizer
    from polystep.adaptive_subspace import AdaptiveSubspace

    set_seed(seed)

    print("  Creating GPT-2 Small model...")
    model = GPT2Small(**GPT2_FINETUNE_CONFIG).to(device)
    load_gpt2_weights(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    # Create AdaptiveSubspace with sparse projection
    subspace = AdaptiveSubspace.auto_from_params(
        model,
        compression_target=0.001,
        max_rank=subspace_dim,
    )
    object.__setattr__(subspace, "rotation_mode", "random")

    optimizer = PolyStepOptimizer(
        model,
        seed=seed,
        subspace=subspace,
        projection_type="sparse",
        step_radius=POLYSTEP_CONFIG["step_radius"],
        probe_radius=POLYSTEP_CONFIG["probe_radius"],
        epsilon=POLYSTEP_CONFIG["epsilon"],
        num_probe=POLYSTEP_CONFIG["num_probe"],
        chunk_size=POLYSTEP_CONFIG["chunk_size"],
        compile=False,
        sinkhorn_max_iters=POLYSTEP_CONFIG["sinkhorn_max_iters"],
    )

    criterion = nn.CrossEntropyLoss()
    buffers = dict(model.named_buffers())

    epoch_logs = []
    step_logs = []
    best_accuracy = 0.0
    step_count = 0
    start_time = time.time()
    train_iter = iter(train_loader)

    def get_batch():
        nonlocal train_iter
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        return batch

    print(f"  Running {num_steps} optimizer steps (subspace_dim={subspace_dim})...")

    with track_gpu_memory() as mem:
        for step in range(1, num_steps + 1):
            step_start = time.time()

            input_ids, attention_mask, labels = get_batch()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            def make_closure(_ids, _mask, _labels):
                def closure(batched_params):
                    was_training = model.training
                    model.eval()
                    try:

                        def single_forward(params):
                            full_dict = {**params, **buffers}
                            logits = functional_call(model, full_dict, (_ids, _mask))
                            return criterion(logits, _labels)

                        losses = vmap(single_forward, in_dims=(0,))(batched_params)
                    finally:
                        if was_training:
                            model.train()
                    return losses

                return closure

            optimizer.step(make_closure(input_ids, attention_mask, labels))

            # Evaluate current loss
            with torch.no_grad():
                output = model(input_ids, attention_mask=attention_mask)
                loss = criterion(output, labels).item()

            step_time = time.time() - step_start
            step_count += 1

            # Per-20-step fine-grained tracking
            if step % 20 == 0:
                step_test_acc = evaluate_accuracy(model, test_loader, device=device)
                best_accuracy = max(best_accuracy, step_test_acc)
                step_logs.append(
                    {
                        "step": step,
                        "test_accuracy": step_test_acc,
                        "loss": loss,
                        "wall_time": time.time() - start_time,
                    }
                )

            # Periodic evaluation
            if step % 10 == 0 or step == num_steps:
                test_acc = evaluate_accuracy(model, test_loader, device=device)
                best_accuracy = max(best_accuracy, test_acc)

                epoch_logs.append(
                    {
                        "epoch": step,
                        "accuracy": test_acc,
                        "loss": loss,
                        "time": step_time,
                    }
                )
                print(
                    f"    Step {step}/{num_steps} | acc={test_acc * 100:.1f}% | loss={loss:.4f} | time={step_time:.1f}s"
                )

    wall_time = time.time() - start_time
    final_acc = evaluate_accuracy(model, test_loader, device=device)
    best_accuracy = max(best_accuracy, final_acc)

    result_path = save_result(
        benchmark=BENCHMARK,
        method="polystep",
        seed=seed,
        metrics={
            "final_accuracy": final_acc,
            "best_accuracy": best_accuracy,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": step_count,
            "total_steps": step_count,
        },
        hyperparameters={
            "total_params": total_params,
            "subspace_dim": subspace_dim,
            "projection_type": "SparseRandomProjection",
            "batch_size": BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "max_train": MAX_TRAIN,
            **POLYSTEP_CONFIG,
        },
        epoch_logs=epoch_logs,
        step_logs=step_logs,
        results_dir=results_dir,
        # SST-2 publishes no test labels, so these four legacy arms select the best
        # epoch and report it on the SAME split. That is a selection leak however
        # conservative the arm (all four report negative results), so the run is
        # stamped and the aggregator refuses to put it in a table. The head-quant
        # family below has a real train/val/test split and a TestSplitTripwire, and
        # is the GPT-2 experiment the paper reports.
        leaked=True,
    )
    print(f"  Saved: {result_path}")
    print(f"  Final accuracy: {final_acc * 100:.1f}%, Best: {best_accuracy * 100:.1f}%")
    print(f"  Wall time: {wall_time:.1f}s, Peak GPU: {mem['peak_gpu_memory_mb']:.0f} MB")

    return {
        "final_accuracy": final_acc,
        "best_accuracy": best_accuracy,
        "wall_time": wall_time,
        "peak_memory_mb": mem["peak_gpu_memory_mb"],
    }


def run_adam(
    seed: int,
    device: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    results_dir: str,
    num_epochs: int = 3,
):
    """Fine-tune GPT-2 on SST-2 with Adam optimizer (gradient baseline).

    Standard gradient-based fine-tuning with lr=2e-5 (typical for GPT-2).

    Args:
        seed: Random seed.
        device: Device string ('cuda' or 'cpu').
        train_loader: Training DataLoader.
        test_loader: Validation DataLoader.
        results_dir: Directory to save result JSON.
        num_epochs: Number of training epochs.
    """
    set_seed(seed)

    print("  Creating GPT-2 Small model...")
    model = GPT2Small(**GPT2_FINETUNE_CONFIG).to(device)
    load_gpt2_weights(model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,} ({total_params / 1e6:.1f}M)")

    optimizer = torch.optim.Adam(model.parameters(), lr=ADAM_CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()

    epoch_logs = []
    best_accuracy = 0.0
    total_steps = 0
    start_time = time.time()

    with track_gpu_memory() as mem:
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            epoch_start = time.time()

            for input_ids, attention_mask, labels in train_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1
                total_steps += 1

            avg_loss = epoch_loss / max(epoch_batches, 1)
            test_acc = evaluate_accuracy(model, test_loader, device=device)
            best_accuracy = max(best_accuracy, test_acc)
            epoch_time = time.time() - epoch_start

            epoch_logs.append(
                {
                    "epoch": epoch,
                    "accuracy": test_acc,
                    "loss": avg_loss,
                    "time": epoch_time,
                }
            )
            print(
                f"    Epoch {epoch}/{num_epochs} | acc={test_acc * 100:.1f}% | loss={avg_loss:.4f} | time={epoch_time:.1f}s"
            )

    wall_time = time.time() - start_time
    final_acc = evaluate_accuracy(model, test_loader, device=device)
    best_accuracy = max(best_accuracy, final_acc)

    result_path = save_result(
        benchmark=BENCHMARK,
        method="adam",
        seed=seed,
        metrics={
            "final_accuracy": final_acc,
            "best_accuracy": best_accuracy,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": total_steps,
            "total_steps": total_steps,
        },
        hyperparameters={
            "total_params": total_params,
            "batch_size": BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "max_train": MAX_TRAIN,
            **ADAM_CONFIG,
        },
        epoch_logs=epoch_logs,
        results_dir=results_dir,
        # Selects and reports on the same split; stamped so the aggregator refuses it.
        leaked=True,
    )
    print(f"  Saved: {result_path}")
    print(f"  Final accuracy: {final_acc * 100:.1f}%, Best: {best_accuracy * 100:.1f}%")
    print(f"  Wall time: {wall_time:.1f}s, Peak GPU: {mem['peak_gpu_memory_mb']:.0f} MB")

    return {
        "final_accuracy": final_acc,
        "best_accuracy": best_accuracy,
        "wall_time": wall_time,
        "peak_memory_mb": mem["peak_gpu_memory_mb"],
    }


def get_backbone_features(
    model: GPT2Small,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Extract pooled features from frozen GPT-2 backbone (no classifier).

    Runs: token_embedding + position_embedding -> transformer layers ->
    layer_norm -> masked mean pooling. Returns [B, 768] feature tensor.

    All computation is done under torch.no_grad() for efficiency.

    Args:
        model: GPT2Small model with pretrained weights loaded.
        input_ids: Input token IDs [B, S].
        attention_mask: Padding mask [B, S] (1 for real tokens, 0 for padding).

    Returns:
        Tensor: Pooled features [B, embed_dim].
    """
    device = input_ids.device
    batch_size, seq_len = input_ids.shape

    if seq_len > model.max_seq_len:
        input_ids = input_ids[:, : model.max_seq_len]
        seq_len = model.max_seq_len
        if attention_mask is not None:
            attention_mask = attention_mask[:, : model.max_seq_len]

    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    x = model.token_embedding(input_ids) + model.position_embedding(positions)

    for layer in model.layers:
        x = layer(x, attention_mask=attention_mask)

    x = model.layer_norm(x)

    # Masked mean pooling (same as GPT2Small.forward)
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
    else:
        x = x.mean(dim=1)

    return x  # [B, embed_dim]


def run_headonly_polystep(
    seed: int,
    device: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    results_dir: str,
    num_epochs: int = HEADONLY_EPOCHS,
):
    """Train only GPT-2 classifier head with polystep (full-space, 1538 params).

    Freezes the entire pretrained backbone and optimizes only the classifier
    head (nn.Linear(768, 2) = 1538 params) using polystep in full-space mode.
    Features are pre-extracted from the frozen backbone for efficiency: avoids
    vmapping over the full 124M parameter model.

    Args:
        seed: Random seed.
        device: Device string ('cuda' or 'cpu').
        train_loader: Training DataLoader yielding (input_ids, attention_mask, labels).
        test_loader: Validation DataLoader.
        results_dir: Directory to save result JSON.
        num_epochs: Number of training epochs over the full train set.
    """
    from torch.func import functional_call, vmap

    from polystep.epsilon import CosineEpsilon
    from polystep.optimizer import PolyStepOptimizer

    set_seed(seed)

    print("  Creating GPT-2 Small model (head-only mode)...")
    model = GPT2Small(**GPT2_FINETUNE_CONFIG).to(device)
    load_gpt2_weights(model)
    model.eval()

    # Freeze entire model
    for p in model.parameters():
        p.requires_grad_(False)

    # Create standalone classifier module for polystep (avoids vmapping 124M params)
    classifier_module = nn.Sequential(nn.Linear(768, 2)).to(device)
    classifier_module[0].weight.data.copy_(model.classifier.weight.data)
    classifier_module[0].bias.data.copy_(model.classifier.bias.data)

    trainable_params = sum(p.numel() for p in classifier_module.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  Trainable params (classifier head): {trainable_params:,}")
    assert trainable_params == 1538, f"Expected 1538 trainable params, got {trainable_params}"

    # Build CosineEpsilon schedules
    cfg = HEADONLY_POLYSTEP_CONFIG
    # Keyword, not positional: CosineEpsilon declares `target` before `init`, so the
    # positional form ran these schedules backwards.
    eps = (
        CosineEpsilon(init=cfg["epsilon_init"], target=cfg["epsilon_target"])
        if "epsilon_init" in cfg
        else cfg.get("epsilon", 2.0)
    )
    sr = (
        CosineEpsilon(init=cfg["step_radius_init"], target=cfg["step_radius_target"])
        if "step_radius_init" in cfg
        else cfg.get("step_radius", 1.0)
    )
    pr = (
        CosineEpsilon(init=cfg["probe_radius_init"], target=cfg["probe_radius_target"])
        if "probe_radius_init" in cfg
        else cfg.get("probe_radius", 1.0)
    )

    optimizer = PolyStepOptimizer(
        classifier_module,
        seed=seed,
        step_radius=sr,
        probe_radius=pr,
        epsilon=eps,
        num_probe=cfg["num_probe"],
        sinkhorn_max_iters=cfg["sinkhorn_max_iters"],
        use_momentum=cfg.get("use_momentum", False),
        momentum_init=cfg.get("momentum_init", 0.5),
        momentum_final=cfg.get("momentum_final", 0.9),
        compile=False,
    )

    criterion = nn.CrossEntropyLoss()
    buffers = dict(classifier_module.named_buffers())

    epoch_logs = []
    step_logs = []
    best_accuracy = 0.0
    total_steps = 0
    start_time = time.time()

    print(f"  Running {num_epochs} epochs (head-only polystep, full-space)...")

    with track_gpu_memory() as mem:
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            epoch_loss = 0.0
            epoch_batches = 0

            for input_ids, attention_mask, labels in train_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                # Pre-extract features from frozen backbone
                with torch.no_grad():
                    features = get_backbone_features(model, input_ids, attention_mask)

                def make_closure(_features, _labels):
                    def closure(batched_params):
                        def single_forward(params):
                            full_dict = {**params, **buffers}
                            logits = functional_call(classifier_module, full_dict, (_features,))
                            return criterion(logits, _labels)

                        losses = vmap(single_forward, in_dims=(0,))(batched_params)
                        return losses

                    return closure

                optimizer.step(make_closure(features, labels))

                # Track loss for logging
                with torch.no_grad():
                    logits = classifier_module(features)
                    batch_loss = criterion(logits, labels).item()
                epoch_loss += batch_loss
                epoch_batches += 1
                total_steps += 1

                # Per-20-step fine-grained tracking
                if total_steps % 20 == 0:
                    model.classifier.weight.data.copy_(classifier_module[0].weight.data)
                    model.classifier.bias.data.copy_(classifier_module[0].bias.data)
                    step_test_acc = evaluate_accuracy(model, test_loader, device=device)
                    step_logs.append(
                        {
                            "step": total_steps,
                            "epoch": epoch,
                            "test_accuracy": step_test_acc,
                            "loss": batch_loss,
                            "wall_time": time.time() - start_time,
                        }
                    )

            avg_loss = epoch_loss / max(epoch_batches, 1)

            # Copy classifier weights back for evaluation
            model.classifier.weight.data.copy_(classifier_module[0].weight.data)
            model.classifier.bias.data.copy_(classifier_module[0].bias.data)

            test_acc = evaluate_accuracy(model, test_loader, device=device)
            best_accuracy = max(best_accuracy, test_acc)
            epoch_time = time.time() - epoch_start

            epoch_logs.append(
                {
                    "epoch": epoch,
                    "accuracy": test_acc,
                    "loss": avg_loss,
                    "time": epoch_time,
                }
            )

            if epoch % 5 == 0 or epoch == num_epochs:
                print(
                    f"    Epoch {epoch}/{num_epochs} | acc={test_acc * 100:.1f}% | loss={avg_loss:.4f} | time={epoch_time:.1f}s"
                )

    wall_time = time.time() - start_time

    # Final evaluation
    model.classifier.weight.data.copy_(classifier_module[0].weight.data)
    model.classifier.bias.data.copy_(classifier_module[0].bias.data)
    final_acc = evaluate_accuracy(model, test_loader, device=device)
    best_accuracy = max(best_accuracy, final_acc)

    result_path = save_result(
        benchmark=HEADONLY_BENCHMARK,
        method="polystep",
        seed=seed,
        metrics={
            "final_accuracy": final_acc,
            "best_accuracy": best_accuracy,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": total_steps,
            "total_steps": total_steps,
        },
        hyperparameters={
            "total_params": total_params,
            "trainable_params": trainable_params,
            "mode": "head_only",
            "num_epochs": num_epochs,
            "batch_size": BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "max_train": MAX_TRAIN,
            **HEADONLY_POLYSTEP_CONFIG,
        },
        epoch_logs=epoch_logs,
        step_logs=step_logs,
        results_dir=results_dir,
        # Selects and reports on the same split; stamped so the aggregator refuses it.
        leaked=True,
    )
    print(f"  Saved: {result_path}")
    print(f"  Final accuracy: {final_acc * 100:.1f}%, Best: {best_accuracy * 100:.1f}%")
    print(f"  Wall time: {wall_time:.1f}s, Peak GPU: {mem['peak_gpu_memory_mb']:.0f} MB")

    return {
        "final_accuracy": final_acc,
        "best_accuracy": best_accuracy,
        "wall_time": wall_time,
        "peak_memory_mb": mem["peak_gpu_memory_mb"],
    }


def run_headonly_adam(
    seed: int,
    device: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    results_dir: str,
    num_epochs: int = HEADONLY_ADAM_CONFIG["epochs"],
):
    """Train only GPT-2 classifier head with Adam (gradient baseline).

    Freezes all backbone parameters and trains only the classifier head
    (1538 params) with Adam lr=1e-3 for 3 epochs.

    Args:
        seed: Random seed.
        device: Device string ('cuda' or 'cpu').
        train_loader: Training DataLoader.
        test_loader: Validation DataLoader.
        results_dir: Directory to save result JSON.
        num_epochs: Number of training epochs.
    """
    set_seed(seed)

    print("  Creating GPT-2 Small model (head-only Adam)...")
    model = GPT2Small(**GPT2_FINETUNE_CONFIG).to(device)
    load_gpt2_weights(model)

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad_(False)

    # Unfreeze only classifier head
    model.classifier.weight.requires_grad_(True)
    model.classifier.bias.requires_grad_(True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  Trainable params (classifier head): {trainable_params:,}")
    assert trainable_params == 1538, f"Expected 1538 trainable params, got {trainable_params}"

    optimizer = torch.optim.Adam(
        [model.classifier.weight, model.classifier.bias],
        lr=HEADONLY_ADAM_CONFIG["lr"],
    )
    criterion = nn.CrossEntropyLoss()

    epoch_logs = []
    best_accuracy = 0.0
    total_steps = 0
    start_time = time.time()

    print(f"  Running {num_epochs} epochs (head-only Adam, lr={HEADONLY_ADAM_CONFIG['lr']})...")

    with track_gpu_memory() as mem:
        for epoch in range(1, num_epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_batches = 0
            epoch_start = time.time()

            for input_ids, attention_mask, labels in train_loader:
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_batches += 1
                total_steps += 1

            avg_loss = epoch_loss / max(epoch_batches, 1)
            test_acc = evaluate_accuracy(model, test_loader, device=device)
            best_accuracy = max(best_accuracy, test_acc)
            epoch_time = time.time() - epoch_start

            epoch_logs.append(
                {
                    "epoch": epoch,
                    "accuracy": test_acc,
                    "loss": avg_loss,
                    "time": epoch_time,
                }
            )
            print(
                f"    Epoch {epoch}/{num_epochs} | acc={test_acc * 100:.1f}% | loss={avg_loss:.4f} | time={epoch_time:.1f}s"
            )

    wall_time = time.time() - start_time
    final_acc = evaluate_accuracy(model, test_loader, device=device)
    best_accuracy = max(best_accuracy, final_acc)

    result_path = save_result(
        benchmark=HEADONLY_BENCHMARK,
        method="adam",
        seed=seed,
        metrics={
            "final_accuracy": final_acc,
            "best_accuracy": best_accuracy,
            "wall_time_seconds": wall_time,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": total_steps,
            "total_steps": total_steps,
        },
        hyperparameters={
            "total_params": total_params,
            "trainable_params": trainable_params,
            "mode": "head_only",
            "num_epochs": num_epochs,
            "batch_size": BATCH_SIZE,
            "max_seq_len": MAX_SEQ_LEN,
            "max_train": MAX_TRAIN,
            **HEADONLY_ADAM_CONFIG,
        },
        epoch_logs=epoch_logs,
        results_dir=results_dir,
        # Selects and reports on the same split; stamped so the aggregator refuses it.
        leaked=True,
    )
    print(f"  Saved: {result_path}")
    print(f"  Final accuracy: {final_acc * 100:.1f}%, Best: {best_accuracy * 100:.1f}%")
    print(f"  Wall time: {wall_time:.1f}s, Peak GPU: {mem['peak_gpu_memory_mb']:.0f} MB")

    return {
        "final_accuracy": final_acc,
        "best_accuracy": best_accuracy,
        "wall_time": wall_time,
        "peak_memory_mb": mem["peak_gpu_memory_mb"],
    }


# ---------------------------------------------------------------------------
# Hard-quantized head: the comparison the smooth head cannot make
# ---------------------------------------------------------------------------
#
# The head-only runs above train an ordinary ``nn.Linear(768, 2)``. Everything
# works there -- Adam, PolyStep, and any zeroth-order method -- which is exactly
# why that setup cannot separate them. Quantize the head's weights with no
# straight-through estimator and the picture splits:
#
#   * Adam gets an exactly-zero gradient (``round``/``sign`` have zero derivative)
#     and never leaves its initialization.
#   * MeZO and EGGROLL estimate a gradient by finite differences. A probe that
#     does not push a weight across a bin boundary changes nothing, so the
#     difference is exactly 0 and the estimated gradient is exactly 0. How often
#     that happens depends on the grid: int8 at scale 0.01 is fine enough that
#     most probes do cross, ``sign()`` is not. Both variants are run, and the
#     runs record ``flat_generation_fraction`` so the claim is measured rather
#     than asserted.
#   * PolyStep never differentiates anything; the forward pass is a cost oracle.
#
# Features are extracted once from the frozen backbone and cached, so all of
# this is minutes, not hours: the backbone is never re-run during training.

HEADQUANT_BENCHMARK = "gpt2_headquant"
HEADQUANT_SCALE = 0.01  # int8 grid step; the weight grid is 0.01 * {-128..127}
#: Candidate evaluations, matched across the gradient-free methods. Evaluations are
#: cheap here (a 768x2 matmul on cached features), so this is minutes per run and
#: still buys PolyStep ~290 steps at ~1.1 evaluations per parameter per step.
HEADQUANT_BUDGET = 500_000
HEADQUANT_BATCH = 64
HEADQUANT_ADAM_EPOCHS = 30
HEADQUANT_ADAM_LR = 1e-3
HEADQUANT_PROBE_EVERY = 5_000

#: ``variant -> (head factory, backprop sees the weights, backprop sees the bias)``.
#:
#: The bias column is not decoration. ``QuantizedLinear`` quantizes its bias as well as
#: its weight, so Adam's gradient on the int8 head is *exactly* zero and the row is a
#: clean "backprop cannot move this at all". ``BinaryLinear`` does not: ``sign`` is
#: declared for the weight only, so 2 of the head's 1,538 parameters stay
#: differentiable and Adam reports ``final_grad_l1 = 1.0`` rather than 0.
#:
#: The bias is deliberately left alone rather than quantized. ``sign`` on a
#: zero-initialized bias is pinned at ``sign(0) = 0`` and snaps to +-1 on the first
#: movement -- a two-state bias that would be a different layer, in a class four other
#: experiments share. Instead the asymmetry is labelled: the binary Adam row carries
#: ``partially_differentiable_control: true`` and must be read as a partial control,
#: not as the same "no gradient exists" claim the int8 row makes. The gradient-free
#: methods are unaffected either way -- none of them differentiates anything.
HEADQUANT_VARIANTS = {
    "int8": (lambda: QuantizedLinear(768, 2, scale=HEADQUANT_SCALE), False, False),
    "binary": (lambda: BinaryLinear(768, 2), False, True),
    "smooth": (lambda: nn.Linear(768, 2), True, True),
}

HEADQUANT_POLYSTEP_CONFIG = {
    "epsilon_init": 5.0,
    "epsilon_target": 0.5,
    "step_radius_init": 2.0,
    "step_radius_target": 0.5,
    "probe_radius_init": 2.0,
    "probe_radius_target": 0.5,
    "num_probe": 1,
    "sinkhorn_max_iters": 50,
    "use_momentum": True,
    "momentum_init": 0.5,
    "momentum_final": 0.9,
}

# MeZO and EGGROLL are run at more than one probe radius on purpose, because
# picking one would decide the result. Both are finite-difference gradient
# estimators, and their analysis is an eps -> 0 argument: at a hard quantizer
# that limit is identically zero, because a small enough probe crosses no bin
# boundary and ``L(x + eps z) - L(x - eps z)`` is exactly 0. Make eps large
# enough to cross boundaries and they still move -- but then they are no longer
# estimating a gradient, they are doing crude random search. Sweeping eps shows
# both ends instead of asserting one, and ``flat_generation_fraction`` in each
# result says which regime that run was in.
#: 1e-5 is the eps -> 0 limit their analysis takes, 1e-3 is MeZO's own default,
#: 0.02 is coarse enough to cross an int8 bin most of the time.
ZO_EPS = [1e-5, 1e-3, 0.02]
HEADQUANT_MEZO = {"eps": 0.02, "lr": 1e-2}
HEADQUANT_EGGROLL = {"sigma": 0.02, "lr": 1e-2, "rank": 1, "popsize": 32}


@torch.no_grad()
def extract_features(model: GPT2Small, loader: DataLoader, device: str):
    """Run the frozen backbone once over a loader. Returns ``(features, labels)``."""
    feats, labels = [], []
    for input_ids, attention_mask, y in loader:
        f = get_backbone_features(model, input_ids.to(device), attention_mask.to(device))
        feats.append(f.float().cpu())
        labels.append(y.cpu())
    return torch.cat(feats), torch.cat(labels)


def load_headquant_features(device: str, cache: str, max_train: int = MAX_TRAIN):
    """Cached GPT-2 features for SST-2, split train / val / test.

    SST-2's own test labels are not public, so its ``validation`` split is the
    test set here and the selection split is carved out of train. The backbone
    is frozen and pretrained, so the features do not depend on the seed and the
    cache is shared by every run.
    """
    if os.path.exists(cache):
        blob = torch.load(cache)
    else:
        print("  Extracting GPT-2 backbone features (once; cached afterwards)...")
        model = GPT2Small(**GPT2_FINETUNE_CONFIG).to(device)
        load_gpt2_weights(model)
        model.eval()
        train_loader, val_loader = get_sst2_gpt2_loaders(MAX_SEQ_LEN, batch_size=32, max_train=max_train)
        tr_x, tr_y = extract_features(model, train_loader, device)
        te_x, te_y = extract_features(model, val_loader, device)
        blob = {"train_x": tr_x, "train_y": tr_y, "test_x": te_x, "test_y": te_y}
        os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
        torch.save(blob, cache)
        del model
        gc.collect()
        torch.cuda.empty_cache()

    n_val = max(1, int(0.1 * len(blob["train_y"])))
    g = torch.Generator().manual_seed(0)  # fixed split: the same data for every method
    perm = torch.randperm(len(blob["train_y"]), generator=g)
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return {
        "train": (blob["train_x"][train_idx].to(device), blob["train_y"][train_idx].to(device)),
        "val": (blob["train_x"][val_idx].to(device), blob["train_y"][val_idx].to(device)),
        "test": (blob["test_x"].to(device), blob["test_y"].to(device)),
    }


@torch.no_grad()
def _head_accuracy(head: nn.Module, split) -> float:
    x, y = split
    return (head(x).argmax(-1) == y).float().mean().item()


def _head_batches(split, batch_size: int, seed: int):
    """Endless shuffled minibatch stream over the cached features.

    Shuffles once per epoch and then hands out *slices*. Indexing per batch with
    a CPU permutation costs a host-to-device copy and a synchronize on every
    call, which was 12 ms -- 100x the actual loss computation, and MeZO calls
    this once per optimizer step.
    """
    x, y = split
    g = torch.Generator(device="cpu").manual_seed(seed)
    while True:
        perm = torch.randperm(len(y), generator=g).to(y.device)
        xs, ys = x[perm], y[perm]
        for i in range(0, len(ys), batch_size):
            yield xs[i : i + batch_size], ys[i : i + batch_size]


def _head_loss_batch(head: nn.Module, W: torch.Tensor, b: torch.Tensor, x, y) -> torch.Tensor:
    """Per-candidate loss for a stack of head weights. ``(N,2,768), (N,2) -> (N,)``.

    All three variants are ``x @ transform(W).T + transform(b)``, so this is one
    einsum instead of ``vmap(functional_call(...))``. It matters: MeZO calls the
    objective once per *step*, with two candidates, so per-call Python and
    dispatch overhead is the entire cost of the run. ``_assert_head_loss_matches``
    checks this against the module's own forward.
    """
    tw = getattr(head, "polystep_weight_transform", None)
    tb = getattr(head, "polystep_bias_transform", None)
    if tw is not None:
        W = tw(W)
    if tb is not None:
        b = tb(b)
    logits = torch.einsum("bi,noi->nbo", x, W) + b.unsqueeze(1)
    n, bs, c = logits.shape
    return F.cross_entropy(logits.reshape(n * bs, c), y.repeat(n), reduction="none").reshape(n, bs).mean(1)


@torch.no_grad()
def _assert_head_loss_matches(head: nn.Module, x, y) -> None:
    """The fast path must agree with the module it replaces, or the run is fiction."""
    fast = _head_loss_batch(head, head.weight.unsqueeze(0), head.bias.unsqueeze(0), x, y)[0]
    slow = F.cross_entropy(head(x), y)
    assert torch.allclose(fast, slow, atol=1e-5), f"head fast path {fast} != module forward {slow}"


class _SignalCounter:
    """How much signal a probe actually produced.

    ``flat`` counts generations where every candidate scored *exactly* the same
    loss -- for a two-point method that is a finite difference of zero, i.e. an
    estimated gradient of zero. ``unique`` is the average fraction of distinct
    loss values in a generation, which degrades gracefully for larger populations.
    """

    def __init__(self, antithetic: bool = False):
        self.antithetic = antithetic
        self.calls = self.sampled = 0
        self.unique = 0.0
        self._flat = self._pairs = None  # on-device; reading every call would sync

    def observe(self, losses: torch.Tensor) -> None:
        n = losses.numel()
        if n < 2:
            return
        self.calls += 1
        flat = (losses.max() == losses.min()).float()
        self._flat = flat if self._flat is None else self._flat + flat
        if self.antithetic and n % 2 == 0:
            # MeZO and EGGROLL both hand the objective ``[x + d ; x - d]``, so this
            # is the fraction of antithetic pairs whose finite difference is
            # exactly zero -- an estimated gradient of exactly zero.
            h = n // 2
            p = (losses[:h] == losses[h:]).float().mean()
            self._pairs = p if self._pairs is None else self._pairs + p
        if self.calls % 100 == 1:  # torch.unique syncs; MeZO calls this 250k times
            self.unique += int(torch.unique(losses).numel()) / n
            self.sampled += 1

    def as_metrics(self):
        return {
            "flat_generation_fraction": float(self._flat) / self.calls if self.calls else None,
            "flat_pair_fraction": float(self._pairs) / self.calls if self._pairs is not None else None,
            "mean_unique_loss_fraction": self.unique / self.sampled if self.sampled else None,
            "generations": self.calls,
        }


def _headquant_search_space(method: str, n_params: int) -> dict:
    """What space this method searched, and whether it is the same as everyone else's.

    On the head-quant experiment it *is* the same, exactly: 1,538 parameters, full
    space, no projection for anybody. That matters because it settles the EGGROLL
    subspace question here rather than deferring it.

    Inside a projected subspace, matching EGGROLL's ``FactoredSubspace`` dimension to
    the shared one forces rank 1, and a ``(d_out, 1)`` coordinate matrix is already
    rank 1, so its ``A B^T`` sampler would have nothing to factor and it would
    degenerate into dense Gaussian ES. There is no such tradeoff here: nothing is
    projected, so dimension and rank are both matched and there is no gap to report.

    The rank EGGROLL runs at is still worth recording. Its perturbation factors the
    ``(2, 768)`` weight, whose maximum rank is ``min(2, 768) = 2``, so ``rank=1`` is a
    genuine low-rank perturbation -- half the available rank, not a degenerate one.
    ``rank=2`` would be the dense control, and ``_lowrank_noise`` clips anything higher.
    """
    space = {
        "subspace_class": None,  # full 1,538-dim parameter space; no projection
        "subspace_rank": None,
        "subspace_dim": n_params,
        "subspace_dim_shared": n_params,
        "subspace_dim_matched": True,
        "matched_on": "rank+dimension (nothing is projected here)",
    }
    if method == "eggroll":
        space["eggroll_rank"] = HEADQUANT_EGGROLL["rank"]
        space["eggroll_max_useful_rank"] = 2  # min(2, 768) for the (2, 768) head weight
        space["eggroll_degenerates_to_dense_es"] = HEADQUANT_EGGROLL["rank"] >= 2
    return space


def _headquant_resolve(variant: str, method: str, point, selection: str, untuned: bool):
    """``(grid point, provenance)``: an explicit sweep trial, or the sweep's pick.

    Falls back to an empty point -- the transplanted defaults -- when nothing has been
    swept, and the result JSON records which of the three it was.
    """
    if point is not None:
        return point, None
    if untuned:
        return {}, None
    entry, provenance = load_selection("gpt2_headquant", variant, method, selection)
    return (entry["point"], provenance) if entry else ({}, None)


def run_headquant(
    variant: str,
    method: str,
    seed: int,
    device: str,
    splits,
    results_dir: str,
    budget: int = HEADQUANT_BUDGET,
    adam_epochs: int = HEADQUANT_ADAM_EPOCHS,
    probe_every: int = HEADQUANT_PROBE_EVERY,
    point=None,
    tune: bool = False,
    selection: str = DEFAULT_SELECTION_PATH,
    untuned: bool = False,
):
    """One (variant, method) run on the 1,538-parameter head over cached features.

    Protocol: train on the train slice, select on the held-out validation slice,
    score SST-2's validation split (used as test) exactly once on the selection.

    ``tune=True`` makes it a sweep trial instead: ``point`` overrides the
    hyperparameters, the test split is unreadable, and the return value is the best
    validation accuracy rather than a test number.
    """
    from polystep.baselines import METHODS, Objective
    from polystep.transform import ParamLayout

    set_seed(seed)
    head_fn, differentiable, bias_differentiable = HEADQUANT_VARIANTS[variant]
    head = head_fn().to(device)
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params == 1538, f"expected the 1538-param head, got {n_params}"
    if tune:
        splits = {**splits, "test": TestSplitTripwire()}

    # One probe scale for everybody, in the same units: PolyStep's probe radius is a
    # norm over all 1,538 coordinates, a baseline's sigma/eps is per-coordinate.
    probe_scale = HEADQUANT_POLYSTEP_CONFIG["probe_radius_target"] / math.sqrt(n_params)
    base_method = method.partition("_eps")[0]
    point, provenance = _headquant_resolve(variant, base_method, point, selection, untuned)

    trajectory = []
    best = {"val": -1.0, "sd": None}
    signal = _SignalCounter(antithetic=method.startswith(("mezo", "eggroll")))
    criterion = nn.CrossEntropyLoss()
    start = time.time()

    def record(evals: int) -> None:
        acc = _head_accuracy(head, splits["val"])
        trajectory.append({"evals": evals, "val_accuracy": acc, "wall_time": time.time() - start})
        if acc > best["val"]:
            best.update(val=acc, sd={k: v.detach().clone() for k, v in head.state_dict().items()})

    with track_gpu_memory() as mem:
        if method == "adam":
            # On a quantized head this is the point of the experiment, not a bug:
            # round()/sign() have zero derivative, so the update is zero and the
            # head stays at its initialization.
            opt = torch.optim.Adam(head.parameters(), lr=HEADQUANT_ADAM_LR)
            stream = _head_batches(splits["train"], HEADQUANT_BATCH, seed)
            steps_per_epoch = max(1, len(splits["train"][1]) // HEADQUANT_BATCH)
            evals = 0
            for epoch in range(adam_epochs):
                for _ in range(steps_per_epoch):
                    x, y = next(stream)
                    opt.zero_grad()
                    criterion(head(x), y).backward()
                    opt.step()
                    evals += 1
                # Adam's trajectory x-axis is optimizer steps, not evaluations; the
                # result carries eval_budget=null and gradient_based=true to say so.
                record(evals)
            grad_norm = sum(float(p.grad.abs().sum()) for p in head.parameters() if p.grad is not None)
            extra = {"final_grad_l1": grad_norm, "gradient_based": True}
            steps = evals
            evals = 0  # gradient method: an evaluation budget does not apply

        elif method == "polystep":
            from polystep.epsilon import CosineEpsilon
            from polystep.optimizer import PolyStepOptimizer
            from torch.func import functional_call, vmap

            cfg = apply_polystep_multipliers(HEADQUANT_POLYSTEP_CONFIG, point)
            est_steps = max(1, int(budget / (1.125 * n_params)))

            def sched(key):
                init, target = cfg[f"{key}_init"], cfg[f"{key}_target"]
                return CosineEpsilon(init=init, target=target, decay=(init - target) / est_steps)

            optimizer = PolyStepOptimizer(
                head,
                compile=False,
                seed=seed,
                epsilon=sched("epsilon"),
                step_radius=sched("step_radius"),
                probe_radius=sched("probe_radius"),
                num_probe=cfg["num_probe"],
                sinkhorn_max_iters=cfg["sinkhorn_max_iters"],
                use_momentum=cfg["use_momentum"],
                momentum_init=cfg["momentum_init"],
                momentum_final=cfg["momentum_final"],
            )
            buffers = dict(head.named_buffers())
            stream = _head_batches(splits["train"], HEADQUANT_BATCH, seed)
            evals = steps = 0
            next_probe = probe_every

            # ponytail: per_step measured from the first step; a full step whose cost
            # would cross the budget is not started, so PolyStep spends <= budget like
            # the Objective-capped baselines instead of overshooting by one step.
            per_step = None
            while evals < budget and (per_step is None or evals + per_step <= budget):
                before = evals
                x, y = next(stream)

                def closure(batched_params, _x=x, _y=y):
                    nonlocal evals
                    evals += next(iter(batched_params.values())).shape[0]
                    losses = vmap(lambda p: criterion(functional_call(head, {**p, **buffers}, (_x,)), _y))(
                        batched_params
                    )
                    signal.observe(losses.detach())
                    return losses

                optimizer.step(closure)
                steps += 1
                per_step = evals - before
                if evals >= next_probe:
                    next_probe = evals + probe_every
                    record(evals)
            extra = {**cfg, "gradient_based": False}

        else:  # mezo / eggroll, over the flat 1538-dim parameter space
            layout = ParamLayout.from_module(head)
            base = {e.key: p.detach().clone() for e, p in zip(layout.entries, head.parameters())}
            keys = list(base)
            shapes = [tuple(e.shape) for e in layout.entries]
            sizes = [math.prod(s) for s in shapes]
            stream = _head_batches(splits["train"], HEADQUANT_BATCH, seed)
            _assert_head_loss_matches(head, *splits["val"])

            def unflatten(flat: torch.Tensor):
                """``(N, total) -> {key: (N, *shape)}``, as an offset from the base weights."""
                out, off = {}, 0
                for k, s, n in zip(keys, shapes, sizes):
                    out[k] = base[k].unsqueeze(0) + flat[:, off : off + n].reshape(-1, *s)
                    off += n
                return out

            @torch.no_grad()
            def fn(flat: torch.Tensor) -> torch.Tensor:
                x, y = next(stream)
                p = unflatten(flat)
                losses = _head_loss_batch(head, p["weight"], p["bias"], x, y)
                signal.observe(losses)
                return losses

            def write(row: torch.Tensor) -> None:
                head.load_state_dict({k: v[0] for k, v in unflatten(row.unsqueeze(0)).items()}, strict=False)

            class _Tracked(Objective):
                def __call__(self, X):
                    out = super().__call__(X)
                    if self.evals >= getattr(self, "_next", probe_every) and self.best_x is not None:
                        self._next = self.evals + probe_every
                        # Score the method's own mean iterate as well as its best
                        # sampled candidate, and let validation selection keep the
                        # higher one -- the same rule run_baseline.probe applies. A
                        # sampled candidate sits ~one probe radius from the mean a
                        # population method actually maintains, so scoring best_x
                        # alone understates MeZO/EGGROLL.
                        it = getattr(self, "iterate", None)
                        if it is not None:
                            write(it)
                            record(self.evals)
                        write(self.best_x)
                        record(self.evals)
                    return out

            obj = _Tracked(fn, layout.total_params, budget, shapes=shapes)
            # "mezo_eps0.001" -> mezo at eps=0.001. The label carries the setting so
            # every probe radius lands in its own result file.
            eps_str = method.partition("_eps")[2]
            hyper = dict(HEADQUANT_MEZO if base_method == "mezo" else HEADQUANT_EGGROLL)
            # Tuned config first, then the explicit ``--zo-eps`` override: that sweep is
            # an eps -> 0 ablation and its whole point is to *pin* eps, at the lr the
            # validation sweep chose rather than at an untuned one.
            hyper = apply_point(hyper, point, probe_scale)
            if eps_str:
                hyper["eps" if base_method == "mezo" else "sigma"] = float(eps_str)
            result = METHODS[base_method](obj, x0=torch.zeros(layout.total_params, device=device), seed=seed, **hyper)
            # Final scoring mirrors the periodic probe: mean iterate and best
            # candidate both validate, selection keeps the higher.
            final_it = getattr(obj, "iterate", None)
            if final_it is not None:
                write(final_it)
                record(obj.evals)
            if obj.best_x is not None:
                write(obj.best_x)
                record(obj.evals)
            evals, steps = obj.evals, result.iters
            extra = {**hyper, "gradient_based": False, "best_train_batch_loss": result.best_loss}

    wall = time.time() - start
    if best["sd"] is None:
        record(evals or steps)
    head.load_state_dict(best["sd"])
    # Scored once, on the selected head -- and never during a sweep, where
    # ``splits["test"]`` is a tripwire that raises if anything unpacks it.
    test_acc = float("nan") if tune else _head_accuracy(head, splits["test"])

    benchmark = f"{HEADQUANT_BENCHMARK}_{variant}"
    path = save_result(
        benchmark=benchmark,
        method=method,
        seed=seed,
        metrics={
            "final_accuracy": test_acc,
            "best_accuracy": best["val"],
            "test_accuracy_at_selected": test_acc,
            "best_val_accuracy": best["val"],
            "wall_time_seconds": wall,
            "peak_gpu_memory_mb": mem["peak_gpu_memory_mb"],
            "function_evals": evals,
            "total_steps": steps,
            "eval_budget": None if method == "adam" else budget,
            "evals_used": None if method == "adam" else evals,
            **signal.as_metrics(),
        },
        hyperparameters={
            "variant": variant,
            "head": HEADQUANT_VARIANTS[variant][0]().__class__.__name__,
            "backprop_sees_the_weights": differentiable,
            # BinaryLinear quantizes the weight but not the bias, so 2 of 1,538
            # parameters stay live and Adam's gradient is nonzero here where int8's is
            # exactly 0. Read that row as a partial control, not as "no gradient
            # exists". See HEADQUANT_VARIANTS for why the bias is left alone.
            "backprop_sees_the_bias": bias_differentiable,
            "partially_differentiable_control": differentiable != bias_differentiable,
            "tuning": {
                "grid_point": point,
                "tuned": provenance is not None,
                "role": "sweep_trial" if tune else "headline",
                "sweep": provenance,
            },
            "trainable_params": n_params,
            "backbone_params": 124_439_808,
            "mode": "head_only_quantized",
            **_headquant_search_space(base_method, n_params),
            "eval_budget": None if method == "adam" else budget,
            "batch_size": HEADQUANT_BATCH,
            "max_train": MAX_TRAIN,
            **extra,
        },
        step_logs=trajectory,
        results_dir=results_dir,
    )
    print(
        f"    saved {path}  test@selected={test_acc * 100:.2f}%  val={best['val'] * 100:.2f}%  "
        f"evals={evals}  flat_pairs={signal.as_metrics()['flat_pair_fraction']}"
    )
    return {"test_accuracy_at_selected": test_acc, "val_accuracy": best["val"], "wall_time": wall}


def measure_memory(device: str = "cuda", batch_size: int = 8, max_seq_len: int = 128):
    """Measure peak VRAM for both polystep and Adam on GPT-2 124M.

    Creates the model, loads pretrained weights, runs 1 step of each method,
    and records peak memory allocation.

    Args:
        device: Device string.
        batch_size: Batch size for profiling.
        max_seq_len: Sequence length for profiling.

    Returns:
        dict: Memory measurements with keys 'polystep_peak_mb', 'adam_peak_mb',
            and 'theoretical' breakdown.
    """
    from torch.func import functional_call, vmap
    from polystep.optimizer import PolyStepOptimizer
    from polystep.adaptive_subspace import AdaptiveSubspace

    results = {}

    print("Measuring polystep memory...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    model = GPT2Small(**GPT2_FINETUNE_CONFIG).to(device)
    load_gpt2_weights(model)

    total_params = sum(p.numel() for p in model.parameters())

    subspace = AdaptiveSubspace.auto_from_params(
        model,
        compression_target=0.001,
        max_rank=POLYSTEP_CONFIG["subspace_dim"],
    )
    object.__setattr__(subspace, "rotation_mode", "random")

    optimizer = PolyStepOptimizer(
        model,
        seed=42,
        subspace=subspace,
        projection_type="sparse",
        step_radius=POLYSTEP_CONFIG["step_radius"],
        probe_radius=POLYSTEP_CONFIG["probe_radius"],
        epsilon=POLYSTEP_CONFIG["epsilon"],
        num_probe=POLYSTEP_CONFIG["num_probe"],
        chunk_size=POLYSTEP_CONFIG["chunk_size"],
        compile=False,
        sinkhorn_max_iters=POLYSTEP_CONFIG["sinkhorn_max_iters"],
    )

    criterion = nn.CrossEntropyLoss()
    buffers = dict(model.named_buffers())

    input_ids = torch.randint(0, 50257, (batch_size, max_seq_len), device=device)
    attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.long, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)

    def make_closure(_ids, _mask, _labels, _model=model, _buffers=buffers):
        def closure(batched_params):
            was_training = _model.training
            _model.eval()
            try:

                def single_forward(params):
                    full_dict = {**params, **_buffers}
                    logits = functional_call(_model, full_dict, (_ids, _mask))
                    return criterion(logits, _labels)

                losses = vmap(single_forward, in_dims=(0,))(batched_params)
            finally:
                if was_training:
                    _model.train()
            return losses

        return closure

    optimizer.step(make_closure(input_ids, attention_mask, labels))
    torch.cuda.synchronize()
    polystep_peak = torch.cuda.max_memory_allocated() / (1024**2)
    results["polystep_peak_mb"] = polystep_peak
    print(f"  polystep peak: {polystep_peak:.0f} MB")

    del optimizer, subspace, model
    gc.collect()
    torch.cuda.empty_cache()

    print("Measuring Adam memory...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    model = GPT2Small(**GPT2_FINETUNE_CONFIG).to(device)
    load_gpt2_weights(model)

    adam_opt = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    input_ids = torch.randint(0, 50257, (batch_size, max_seq_len), device=device)
    attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.long, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)

    model.train()
    adam_opt.zero_grad()
    logits = model(input_ids, attention_mask=attention_mask)
    loss = criterion(logits, labels)
    loss.backward()
    adam_opt.step()
    torch.cuda.synchronize()

    adam_peak = torch.cuda.max_memory_allocated() / (1024**2)
    results["adam_peak_mb"] = adam_peak
    print(f"  Adam peak: {adam_peak:.0f} MB")

    del adam_opt, model
    gc.collect()
    torch.cuda.empty_cache()

    # Theoretical breakdown
    param_bytes = total_params * 4  # FP32
    param_mb = param_bytes / (1024**2)

    results["theoretical"] = {
        "model_weights_mb": param_mb,
        "adam_gradients_mb": param_mb,
        "adam_states_mb": param_mb * 2,  # m + v
        "polystep_sparse_projection_mb": 11.0,  # Estimated
        "polystep_subspace_state_mb": 10.0,  # Estimated
        "polystep_no_gradients": True,
        "polystep_no_backward_activations": True,
    }

    print("\nTheoretical breakdown:")
    print(f"  Model weights: {param_mb:.0f} MB")
    print(f"  Adam gradients: {param_mb:.0f} MB")
    print(f"  Adam optimizer states (m+v): {param_mb * 2:.0f} MB")
    print("  polystep sparse projection: ~11 MB")
    print("  polystep subspace state: ~10 MB")

    return results


METHOD_RUNNERS = {
    "polystep": run_polystep,
    "adam": run_adam,
}

HEADONLY_METHOD_RUNNERS = {
    "polystep": run_headonly_polystep,
    "adam": run_headonly_adam,
}


#: Everything ``--tune`` sweeps here. Adam is gradient-based, so an evaluation budget
#: does not apply to it and it is not swept; it keeps its published lr.
#:
#: Cheapest first. At the same evaluation budget MeZO costs ~1000x the wall-clock of
#: the other two -- it scores 2 candidates per generation where EGGROLL scores 32 and
#: PolyStep 2,307, and ``Objective.__call__`` pays two host syncs per generation
#: regardless of population -- so it runs last and the selection file is rewritten
#: after each method rather than only at the end.
HEADQUANT_TUNABLE = ["polystep", "eggroll", "mezo"]
#: ``--tune`` budget as a fraction of ``--eval-budget``. PolyStep's cosine schedules are
#: derived from the budget, so a reduced-budget trial is a scale model of the full run:
#: same schedule shape, proportionally fewer steps.
#:
#: 5 and not more was measured, not guessed. PolyStep spends ~2,307 evaluations per step
#: on the 1,538-parameter head, so the reduction factor *is* the step count, and on the
#: int8 head its nine configurations produce this many distinct validation scores:
#:
#:     1/25 (20k evals,   8 steps)   1 of 9 -- every config ties at the majority class
#:     1/10 (50k evals,  21 steps)   5 of 9
#:     1/5  (100k evals, 43 steps)   6 of 9
#:     1/1  (500k evals, 216 steps)  9 of 9
#:
#: Below 1/5 the sweep does not choose anything: the tie-break returns the grid's first
#: entry, which is the untuned prior. 1/5 is the cheapest factor that still selects.
HEADQUANT_TUNE_DIVISOR = 5


def _tune_head_quant(args, splits):
    """Validation-only sweep over the hard-quantized head.

    Nine configurations per method, one seed, one reduced budget shared by all of
    them. The test split is a :class:`TestSplitTripwire` for every trial, so a leak
    raises instead of quietly selecting on test.
    """
    variants = args.head_quant or list(HEADQUANT_VARIANTS)
    methods = [m for m in (args.methods if args.methods != ["polystep", "adam"] else HEADQUANT_TUNABLE)]
    budget = max(1, args.eval_budget // HEADQUANT_TUNE_DIVISOR)
    trial_dir = os.path.join(os.path.dirname(args.selection), "headquant_trials")
    seed = args.seeds[0]

    n = sum(len(TUNING_GRID[m]) for m in methods if m in HEADQUANT_TUNABLE)
    print(f"GPT-2 head-quant tuning sweep (validation only) | {variants} x {methods} x seed {seed}")
    print(f"  {n} configs per variant, {budget} evals each (1/{HEADQUANT_TUNE_DIVISOR} of the headline budget)\n")

    trials, path = [], args.selection
    # Method-outer, so the cheap methods finish first and the selection file is usable
    # (and rewritten) after each one rather than only at the very end. MeZO is ~1000x
    # the wall-clock of the other two at the same evaluation budget; see the report.
    for method in methods:
        if method not in HEADQUANT_TUNABLE:
            print(f"  skip {method} (gradient-based; not budget-matched, not swept)")
            continue
        for variant in variants:
            print(f"=== {method} / {variant} ===")
            for i, point in enumerate(TUNING_GRID[method]):
                name = "_".join(f"{k}{v:g}" for k, v in sorted(point.items()))
                try:
                    out = run_headquant(
                        variant,
                        method,
                        seed,
                        args.device,
                        splits,
                        os.path.join(trial_dir, name),
                        budget,
                        adam_epochs=1,
                        probe_every=max(1, budget // 10),
                        point=point,
                        tune=True,
                    )
                except Exception as e:
                    print(f"  ERROR {variant}/{method}/{name}: {type(e).__name__}: {e}")
                    import traceback

                    traceback.print_exc()
                    continue
                trials.append(
                    {
                        "showcase": variant,
                        "method": method,
                        "index": i,
                        "name": name,
                        "point": point,
                        "val": out["val_accuracy"],
                    }
                )
                print(f"  {method:10s} {name:34s} val={out['val_accuracy'] * 100:.2f}%")
                # Checkpoint every trial. MeZO's nine configurations cost hours where
                # the other two cost minutes, so a file written only at the end is a
                # file you cannot use until the slowest method finishes -- and lose
                # entirely if the run is interrupted. ``trials_per_method`` in the
                # provenance says how much of the sweep the file actually represents.
                path = _write_headquant_selection(args, trials, budget, seed, methods, variants)

        print(f"  -> {method} done ({sum(t['method'] == method for t in trials)} trials)")

    print(f"\nwrote {len(trials)} trials and the selected configs to {path}")
    return path


def _write_headquant_selection(args, trials, budget, seed, methods, variants) -> str:
    """The selection file, with everything the paper has to cite about the sweep."""
    wanted = [m for m in methods if m in HEADQUANT_TUNABLE]
    done = {m: sum(t["method"] == m for t in trials) for m in wanted}
    full = {m: len(TUNING_GRID[m]) * len(variants) for m in wanted}
    return write_selection(
        "gpt2_headquant",
        trials,
        {
            "sweep": "experiments/runners/run_gpt2_finetune.py --head-quant --tune",
            "split": "validation only (test split replaced by TestSplitTripwire)",
            "selection_metric": "best_val_accuracy",
            "tie_break": "lowest TUNING_GRID index",
            "seeds": [seed],
            "budget_per_config": budget,
            "headline_budget": args.eval_budget,
            "budget_reduction_factor": HEADQUANT_TUNE_DIVISOR,
            "probe_scale": HEADQUANT_POLYSTEP_CONFIG["probe_radius_target"] / math.sqrt(1538),
            "batch_size": HEADQUANT_BATCH,
            "methods_requested": wanted,
            "trials_per_method": done,
            "trials_expected_per_method": full,
            "complete": done == full,
            "not_swept": ["adam (gradient-based: no evaluation budget applies)"],
        },
        {m: tuning_cost(m, budget, seeds=1) for m in wanted},
        args.selection,
    )


def _main_head_quant(args):
    """``--head-quant``: variants x methods x seeds over the cached features."""
    variants = args.head_quant or list(HEADQUANT_VARIANTS)
    if args.methods != ["polystep", "adam"]:
        methods = args.methods
    else:
        # The tuned rows first, then the eps ablation. Without the bare ``mezo`` /
        # ``eggroll`` entries the table would have no row at the probe radius the
        # validation sweep actually chose -- every ZO row would be eps-pinned by the
        # ablation, and the tuned sigma would never be used.
        zo = [f"{m}_eps{e:g}" for e in args.zo_eps for m in ("mezo", "eggroll")]
        methods = ["polystep", "mezo", "eggroll", *zo, "adam"]
    seeds, budget, adam_epochs = args.seeds, args.eval_budget, HEADQUANT_ADAM_EPOCHS
    results_dir = args.results_dir
    if args.smoke:
        seeds, budget, adam_epochs = seeds[:1], 20_000, 2
        results_dir = os.path.join(results_dir, "headquant_smoke")

    splits = load_headquant_features(args.device, args.feature_cache)
    if args.tune:
        return _tune_head_quant(args, splits)

    print(f"GPT-2 head-only, hard-quantized: {variants} x {methods} x {seeds}, budget={budget}\n")
    print(f"  train {len(splits['train'][1])} | val {len(splits['val'][1])} | test {len(splits['test'][1])}\n")

    for variant in variants:
        print(f"=== {variant} ===")
        for method in methods:
            for seed in seeds:
                out = os.path.join(results_dir, f"{HEADQUANT_BENCHMARK}_{variant}_{method}_{seed}.json")
                if os.path.exists(out):
                    print(f"  skip {method} seed={seed} (exists)")
                    continue
                print(f"  {method} seed={seed}")
                try:
                    run_headquant(
                        variant,
                        method,
                        seed,
                        args.device,
                        splits,
                        results_dir,
                        budget,
                        adam_epochs,
                        selection=args.selection,
                        untuned=args.untuned,
                    )
                except Exception as e:
                    print(f"    ERROR: {method} seed={seed}: {e}")
                    import traceback

                    traceback.print_exc()
    print(f"\nDone. Results in {results_dir}")


def main():
    parser = argparse.ArgumentParser(description="GPT-2 124M Fine-Tuning on SST-2: polystep vs Adam")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["polystep", "adam"],
        help="Methods to run (default: polystep adam)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Seeds to run (default: 42 123 456)",
    )
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument(
        "--steps",
        type=int,
        default=NUM_STEPS,
        help=f"Number of polystep optimizer steps (default: {NUM_STEPS})",
    )
    parser.add_argument(
        "--subspace-dim",
        type=int,
        default=POLYSTEP_CONFIG["subspace_dim"],
        help=f"Subspace dimensionality (default: {POLYSTEP_CONFIG['subspace_dim']})",
    )
    parser.add_argument(
        "--results-dir",
        default="experiments/results",
        help="Results directory (default: experiments/results)",
    )
    parser.add_argument(
        "--measure-memory",
        action="store_true",
        help="Run memory profiling only (no training)",
    )
    parser.add_argument(
        "--head-only",
        action="store_true",
        help="Train only classifier head (1,538 params) with frozen backbone",
    )
    parser.add_argument(
        "--head-quant",
        nargs="*",
        choices=list(HEADQUANT_VARIANTS),
        help=(
            "Hard-quantized head experiment over cached backbone features. Give the "
            "variants to run, or pass the flag alone for all three (int8 binary smooth). "
            "Methods: polystep mezo eggroll adam."
        ),
    )
    parser.add_argument(
        "--eval-budget",
        type=int,
        default=HEADQUANT_BUDGET,
        help=f"Candidate evaluations per gradient-free head-quant run (default: {HEADQUANT_BUDGET})",
    )
    parser.add_argument(
        "--zo-eps",
        nargs="+",
        type=float,
        default=ZO_EPS,
        help=f"Probe radii for MeZO/EGGROLL; one run each (default: {ZO_EPS})",
    )
    parser.add_argument(
        "--feature-cache",
        default="experiments/results/cache/gpt2_sst2_features.pt",
        help="Where the frozen-backbone features are cached",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny end-to-end check of the head-quant experiment: 1 seed, 20k evaluations",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help=(
            f"Validation-only hyperparameter sweep of the --head-quant experiment. Every "
            f"gradient-free method over its 9-config grid at 1/{HEADQUANT_TUNE_DIVISOR} of "
            f"--eval-budget, one seed; writes the picks to --selection. The test split is "
            f"unreadable for the whole sweep."
        ),
    )
    parser.add_argument("--selection", default=DEFAULT_SELECTION_PATH, help="Where --tune writes and a run reads")
    parser.add_argument(
        "--untuned",
        action="store_true",
        help="Ignore --selection and use the transplanted defaults (the pre-sweep numbers)",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Determine benchmark mode
    benchmark = HEADONLY_BENCHMARK if args.head_only else BENCHMARK
    mode_label = "Head-Only" if args.head_only else "Full"

    if args.head_quant is not None:
        return _main_head_quant(args)

    print(f"GPT-2 124M Fine-Tuning on SST-2 ({mode_label})")
    print(f"  Methods: {args.methods}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Device: {args.device}")
    if not args.head_only:
        print(f"  Steps (polystep): {args.steps}")
        print(f"  Subspace dim: {args.subspace_dim}")
    else:
        print("  Mode: head-only (classifier head, 1538 params)")
    print()

    if args.measure_memory:
        if args.device != "cuda":
            print("Memory profiling requires CUDA")
            return
        measure_memory(device=args.device)
        return

    # Load data once
    print("Loading SST-2 with GPT-2 tokenizer...")
    train_loader, val_loader = get_sst2_gpt2_loaders(
        max_seq_len=MAX_SEQ_LEN,
        batch_size=BATCH_SIZE,
        max_train=MAX_TRAIN,
    )
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print()

    for method in args.methods:
        for seed in args.seeds:
            output_file = os.path.join(args.results_dir, f"{benchmark}_{method}_{seed}.json")
            if os.path.exists(output_file):
                print(f"Skipping {method} seed={seed} (result exists: {output_file})")
                continue

            print(f"Running {method} seed={seed} ({mode_label})...")
            try:
                if args.head_only:
                    # Head-only dispatch
                    if method == "polystep":
                        run_headonly_polystep(
                            seed=seed,
                            device=args.device,
                            train_loader=train_loader,
                            test_loader=val_loader,
                            results_dir=args.results_dir,
                        )
                    elif method == "adam":
                        run_headonly_adam(
                            seed=seed,
                            device=args.device,
                            train_loader=train_loader,
                            test_loader=val_loader,
                            results_dir=args.results_dir,
                        )
                    else:
                        print(f"  Unknown method: {method}")
                else:
                    # Full fine-tuning dispatch
                    if method == "polystep":
                        run_polystep(
                            seed=seed,
                            device=args.device,
                            train_loader=train_loader,
                            test_loader=val_loader,
                            results_dir=args.results_dir,
                            num_steps=args.steps,
                            subspace_dim=args.subspace_dim,
                        )
                    elif method == "adam":
                        run_adam(
                            seed=seed,
                            device=args.device,
                            train_loader=train_loader,
                            test_loader=val_loader,
                            results_dir=args.results_dir,
                            num_epochs=ADAM_CONFIG["epochs"],
                        )
                    else:
                        print(f"  Unknown method: {method}")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  OOM ERROR: {method} seed={seed} ran out of GPU memory")
                    print(f"    Error: {e}")
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise
            except Exception as e:
                print(f"  ERROR: {method} seed={seed} failed: {e}")
                import traceback

                traceback.print_exc()

    print(f"\nDone. Results in experiments/results/{benchmark}_*.json")


if __name__ == "__main__":
    main()
