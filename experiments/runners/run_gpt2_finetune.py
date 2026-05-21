#!/usr/bin/env python
"""Run GPT-2 124M fine-tuning experiment on SST-2 with polystep.

This experiment fine-tunes pretrained GPT-2 small (124M parameters) on SST-2
binary sentiment classification using polystep's gradient-free optimizer with
SparseRandomProjection, compared against an Adam baseline.

Key technical components:
1. Weight mapping from HuggingFace Conv1D-based GPT-2 to custom VmapSafe model
   (Conv1D transpose + fused QKV split for VmapSafeMultiHeadAttention)
2. GPT2Small with attention_mask support for padded SST-2 sequences
3. SST-2 data loading with GPT-2 BPE tokenizer (50257 vocab)
4. polystep fine-tuning with SparseRandomProjection (128-dim subspace)
5. Adam baseline fine-tuning (lr=2e-5)
6. Memory profiling comparing peak VRAM for both methods

The weight loading correctness is verified by forward pass comparison against
HuggingFace model (must match within 1e-4 tolerance).

Usage:
    python experiments/runners/run_gpt2_finetune.py --methods polystep adam
    python experiments/runners/run_gpt2_finetune.py --methods polystep --steps 50
    python experiments/runners/run_gpt2_finetune.py --measure-memory
    python experiments/runners/run_gpt2_finetune.py --methods adam --seeds 42 123 456
    python experiments/runners/run_gpt2_finetune.py --head-only --methods polystep adam
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from typing import Optional

# Ensure repo root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from experiments.runners.common import (
    evaluate_accuracy,
    save_result,
    set_seed,
    track_gpu_memory,
)

# Import VmapSafeMultiHeadAttention for GPT-2 model
from polystep.layers import VmapSafeMultiHeadAttention


# ---------------------------------------------------------------------------
# GPT-2 Fine-Tuning Configuration
# ---------------------------------------------------------------------------

BENCHMARK = "gpt2_finetune"

GPT2_FINETUNE_CONFIG = {
    "vocab_size": 50257,      # BPE vocabulary
    "max_seq_len": 128,       # Reduced from 1024 for memory efficiency
    "embed_dim": 768,         # d_model
    "num_heads": 12,          # 64 per head
    "num_layers": 12,         # Transformer blocks
    "ff_dim": 3072,           # 4x expansion
    "dropout": 0.0,           # Disable for vmap compatibility
    "num_classes": 2,         # SST-2 binary sentiment classification
}

PSTORCH_CONFIG = {
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
MAX_TRAIN = 5000       # Limit training samples for feasibility
MAX_SEQ_LEN = 128

# Head-only fine-tuning configs (train classifier head only, freeze backbone)
# CosineEpsilon scheduling: broader exploration early -> refinement
# K=1 optimal (single probe, softmax solver auto-selected)
# Momentum smooths trajectory on 1538-param landscape
HEADONLY_PSTORCH_CONFIG = {
    "epsilon_init": 5.0,
    "epsilon_target": 0.5,
    "step_radius_init": 2.0,
    "step_radius_target": 0.5,
    "probe_radius_init": 2.0,
    "probe_radius_target": 0.5,
    "num_probe": 1,
    "chunk_size": None,  # Full-space, no chunking needed for 1538 params
    "sinkhorn_max_iters": 50,
    "use_momentum": True,
    "momentum_init": 0.5,
    "momentum_final": 0.9,
}
HEADONLY_ADAM_CONFIG = {
    "lr": 1e-3,
    "epochs": 3,
}
HEADONLY_EPOCHS = 50  # polystep epochs (not steps -- small param count allows epoch-based training)
HEADONLY_BENCHMARK = "gpt2_headonly"


# ---------------------------------------------------------------------------
# GPT-2 Small Model (modified from run_gpt2_feasibility.py for SST-2)
# ---------------------------------------------------------------------------

class GPT2TransformerBlock(nn.Module):
    """Single Transformer block with vmap-safe attention (GPT-2 style).

    Uses VmapSafeMultiHeadAttention for compatibility with polystep's
    vmap-based evaluation. Pre-norm style (GPT-2).
    """

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
            nn.GELU(approximate='tanh'),  # GPT-2 uses gelu_new (tanh approximation)
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

        # Causal attention mask (GPT-2 is a causal language model)
        # Upper triangular of -inf prevents attending to future tokens
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=x.device, dtype=x.dtype),
            diagonal=1,
        )

        # Self-attention with residual (pre-norm)
        normed = self.norm1(x)
        attn_out = self.attention(normed, normed, normed, attn_mask=causal_mask)
        x = x + self.dropout_layer(attn_out)

        # Feed-forward with residual (pre-norm)
        normed = self.norm2(x)
        ff_out = self.ff(normed)
        x = x + ff_out

        return x


class GPT2Small(nn.Module):
    """GPT-2 Small model (124M parameters) for classification.

    Architecture: 12 layers, 768 embed_dim, 12 heads, 3072 FF dim, 50257 vocab.
    Adapted for classification (masked mean pooling + linear head) instead of
    language modeling. Supports attention_mask for padded sequences.
    """

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

        self.layers = nn.ModuleList([
            GPT2TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

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
            input_ids = input_ids[:, :self.max_seq_len]
            seq_len = self.max_seq_len
            if attention_mask is not None:
                attention_mask = attention_mask[:, :self.max_seq_len]

        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)

        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)

        x = self.layer_norm(x)

        # Masked mean pooling (ignore padding tokens)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()  # [B, S, 1]
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)

        logits = self.classifier(x)
        return logits


# ---------------------------------------------------------------------------
# Weight Loading: HuggingFace GPT-2 -> Custom GPT2Small
# ---------------------------------------------------------------------------

def load_gpt2_weights(model: GPT2Small, hf_model_name: str = "gpt2") -> dict:
    """Load pretrained GPT-2 weights into custom GPT2Small model.

    Handles three key transformations:
    1. Conv1D weight transpose: HF stores [in, out], nn.Linear stores [out, in]
    2. Fused QKV split: c_attn [768, 2304] -> W_q/W_k/W_v [768, 768] each
    3. Position embedding truncation: [1024, 768] -> [max_seq_len, 768]

    Args:
        model: Custom GPT2Small model to load weights into.
        hf_model_name: HuggingFace model name (default: 'gpt2').

    Returns:
        dict: The weight mapping used for loading.
    """
    from transformers import GPT2Model as HF_GPT2

    hf = HF_GPT2.from_pretrained(hf_model_name)
    hf_sd = hf.state_dict()

    mapping = {}

    # Embeddings
    mapping["token_embedding.weight"] = hf_sd["wte.weight"]  # [50257, 768]
    mapping["position_embedding.weight"] = hf_sd["wpe.weight"][:model.max_seq_len]  # truncate

    # Final LayerNorm
    mapping["layer_norm.weight"] = hf_sd["ln_f.weight"]
    mapping["layer_norm.bias"] = hf_sd["ln_f.bias"]

    for i in range(12):
        pfx = f"h.{i}"
        lpfx = f"layers.{i}"

        # Fused QKV -> separate Q, K, V (split BEFORE transposing)
        c_attn_w = hf_sd[f"{pfx}.attn.c_attn.weight"]  # [768, 2304]
        c_attn_b = hf_sd[f"{pfx}.attn.c_attn.bias"]    # [2304]
        q_w, k_w, v_w = c_attn_w.split(768, dim=1)     # each [768, 768]
        q_b, k_b, v_b = c_attn_b.split(768, dim=0)     # each [768]

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
    assert set(missing) == {"classifier.weight", "classifier.bias"}, (
        f"Unexpected missing keys: {missing}"
    )
    assert len(unexpected) == 0, f"Unexpected keys: {unexpected}"

    # Free HF model memory
    del hf, hf_sd
    gc.collect()

    return mapping


# ---------------------------------------------------------------------------
# Forward Pass Verification
# ---------------------------------------------------------------------------

def verify_forward_pass(custom_model: GPT2Small, hf_model_name: str = "gpt2") -> float:
    """Verify weight loading by comparing hidden states with HuggingFace model.

    Compares at the layer_norm output (before pooling/classifier) to isolate
    the transformer stack comparison from classification-specific components.

    Args:
        custom_model: GPT2Small model with pretrained weights loaded.
        hf_model_name: HuggingFace model name for reference.

    Returns:
        float: Maximum absolute difference between hidden states.

    Raises:
        AssertionError: If max difference exceeds 1e-4.
    """
    from transformers import GPT2Model as HF_GPT2

    hf = HF_GPT2.from_pretrained(hf_model_name).eval()
    custom_model.eval()

    # Test input: "This movie is great" in GPT-2 BPE
    input_ids = torch.tensor([[1212, 3807, 318, 1049]])

    with torch.no_grad():
        # HF output: last_hidden_state [1, 4, 768]
        hf_out = hf(input_ids).last_hidden_state

        # Custom model: extract pre-pooling hidden states
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len).unsqueeze(0)
        x = custom_model.token_embedding(input_ids) + custom_model.position_embedding(positions)
        for layer in custom_model.layers:
            x = layer(x)
        custom_out = custom_model.layer_norm(x)  # [1, 4, 768]

    max_diff = (hf_out - custom_out).abs().max().item()
    print(f"Forward pass verification: max absolute difference = {max_diff:.8f}")

    assert max_diff < 1e-4, f"Forward pass mismatch: max diff = {max_diff}"

    del hf
    gc.collect()

    return max_diff


# ---------------------------------------------------------------------------
# SST-2 Data Loading with GPT-2 BPE Tokenizer
# ---------------------------------------------------------------------------

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

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token by default

    ds = load_dataset("glue", "sst2")
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
        padding='max_length',
        return_tensors='pt',
    )
    val_enc = tokenizer(
        val_texts,
        max_length=max_seq_len,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )

    train_ds = TensorDataset(
        train_enc['input_ids'],
        train_enc['attention_mask'],
        torch.tensor(train_labels),
    )
    val_ds = TensorDataset(
        val_enc['input_ids'],
        val_enc['attention_mask'],
        torch.tensor(val_labels),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# polystep Fine-Tuning Runner
# ---------------------------------------------------------------------------

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
        model, compression_target=0.001, max_rank=subspace_dim,
    )
    object.__setattr__(subspace, 'rotation_mode', 'random')

    optimizer = PolyStepOptimizer(
        model,
        seed=seed,
        subspace=subspace,
        projection_type='sparse',
        step_radius=PSTORCH_CONFIG["step_radius"],
        probe_radius=PSTORCH_CONFIG["probe_radius"],
        epsilon=PSTORCH_CONFIG["epsilon"],
        num_probe=PSTORCH_CONFIG["num_probe"],
        chunk_size=PSTORCH_CONFIG["chunk_size"],
        compile=False,
        sinkhorn_max_iters=PSTORCH_CONFIG["sinkhorn_max_iters"],
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
                step_logs.append({
                    "step": step,
                    "test_accuracy": step_test_acc,
                    "loss": loss,
                    "wall_time": time.time() - start_time,
                })

            # Periodic evaluation
            if step % 10 == 0 or step == num_steps:
                test_acc = evaluate_accuracy(model, test_loader, device=device)
                best_accuracy = max(best_accuracy, test_acc)

                epoch_logs.append({
                    "epoch": step,
                    "accuracy": test_acc,
                    "loss": loss,
                    "time": step_time,
                })
                print(f"    Step {step}/{num_steps} | acc={test_acc*100:.1f}% | loss={loss:.4f} | time={step_time:.1f}s")

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
            **PSTORCH_CONFIG,
        },
        epoch_logs=epoch_logs,
        step_logs=step_logs,
        results_dir=results_dir,
    )
    print(f"  Saved: {result_path}")
    print(f"  Final accuracy: {final_acc*100:.1f}%, Best: {best_accuracy*100:.1f}%")
    print(f"  Wall time: {wall_time:.1f}s, Peak GPU: {mem['peak_gpu_memory_mb']:.0f} MB")

    return {
        "final_accuracy": final_acc,
        "best_accuracy": best_accuracy,
        "wall_time": wall_time,
        "peak_memory_mb": mem["peak_gpu_memory_mb"],
    }


# ---------------------------------------------------------------------------
# Adam Baseline Runner
# ---------------------------------------------------------------------------

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

            epoch_logs.append({
                "epoch": epoch,
                "accuracy": test_acc,
                "loss": avg_loss,
                "time": epoch_time,
            })
            print(f"    Epoch {epoch}/{num_epochs} | acc={test_acc*100:.1f}% | loss={avg_loss:.4f} | time={epoch_time:.1f}s")

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
    )
    print(f"  Saved: {result_path}")
    print(f"  Final accuracy: {final_acc*100:.1f}%, Best: {best_accuracy*100:.1f}%")
    print(f"  Wall time: {wall_time:.1f}s, Peak GPU: {mem['peak_gpu_memory_mb']:.0f} MB")

    return {
        "final_accuracy": final_acc,
        "best_accuracy": best_accuracy,
        "wall_time": wall_time,
        "peak_memory_mb": mem["peak_gpu_memory_mb"],
    }


# ---------------------------------------------------------------------------
# Head-Only Fine-Tuning (Classifier Head Only, Frozen Backbone)
# ---------------------------------------------------------------------------

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
        input_ids = input_ids[:, :model.max_seq_len]
        seq_len = model.max_seq_len
        if attention_mask is not None:
            attention_mask = attention_mask[:, :model.max_seq_len]

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
    Features are pre-extracted from the frozen backbone for efficiency -- avoids
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
    cfg = HEADONLY_PSTORCH_CONFIG
    eps = CosineEpsilon(cfg["epsilon_init"], cfg["epsilon_target"]) if "epsilon_init" in cfg else cfg.get("epsilon", 2.0)
    sr = CosineEpsilon(cfg["step_radius_init"], cfg["step_radius_target"]) if "step_radius_init" in cfg else cfg.get("step_radius", 1.0)
    pr = CosineEpsilon(cfg["probe_radius_init"], cfg["probe_radius_target"]) if "probe_radius_init" in cfg else cfg.get("probe_radius", 1.0)

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
                    step_logs.append({
                        "step": total_steps,
                        "epoch": epoch,
                        "test_accuracy": step_test_acc,
                        "loss": batch_loss,
                        "wall_time": time.time() - start_time,
                    })

            avg_loss = epoch_loss / max(epoch_batches, 1)

            # Copy classifier weights back for evaluation
            model.classifier.weight.data.copy_(classifier_module[0].weight.data)
            model.classifier.bias.data.copy_(classifier_module[0].bias.data)

            test_acc = evaluate_accuracy(model, test_loader, device=device)
            best_accuracy = max(best_accuracy, test_acc)
            epoch_time = time.time() - epoch_start

            epoch_logs.append({
                "epoch": epoch,
                "accuracy": test_acc,
                "loss": avg_loss,
                "time": epoch_time,
            })

            if epoch % 5 == 0 or epoch == num_epochs:
                print(f"    Epoch {epoch}/{num_epochs} | acc={test_acc*100:.1f}% | loss={avg_loss:.4f} | time={epoch_time:.1f}s")

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
            **HEADONLY_PSTORCH_CONFIG,
        },
        epoch_logs=epoch_logs,
        step_logs=step_logs,
        results_dir=results_dir,
    )
    print(f"  Saved: {result_path}")
    print(f"  Final accuracy: {final_acc*100:.1f}%, Best: {best_accuracy*100:.1f}%")
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

            epoch_logs.append({
                "epoch": epoch,
                "accuracy": test_acc,
                "loss": avg_loss,
                "time": epoch_time,
            })
            print(f"    Epoch {epoch}/{num_epochs} | acc={test_acc*100:.1f}% | loss={avg_loss:.4f} | time={epoch_time:.1f}s")

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
    )
    print(f"  Saved: {result_path}")
    print(f"  Final accuracy: {final_acc*100:.1f}%, Best: {best_accuracy*100:.1f}%")
    print(f"  Wall time: {wall_time:.1f}s, Peak GPU: {mem['peak_gpu_memory_mb']:.0f} MB")

    return {
        "final_accuracy": final_acc,
        "best_accuracy": best_accuracy,
        "wall_time": wall_time,
        "peak_memory_mb": mem["peak_gpu_memory_mb"],
    }


# ---------------------------------------------------------------------------
# Memory Profiling
# ---------------------------------------------------------------------------

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

    # -- polystep memory --
    print("Measuring polystep memory...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    model = GPT2Small(**GPT2_FINETUNE_CONFIG).to(device)
    load_gpt2_weights(model)

    total_params = sum(p.numel() for p in model.parameters())

    subspace = AdaptiveSubspace.auto_from_params(
        model, compression_target=0.001, max_rank=PSTORCH_CONFIG["subspace_dim"],
    )
    object.__setattr__(subspace, 'rotation_mode', 'random')

    optimizer = PolyStepOptimizer(
        model,
        seed=42,
        subspace=subspace,
        projection_type='sparse',
        step_radius=PSTORCH_CONFIG["step_radius"],
        probe_radius=PSTORCH_CONFIG["probe_radius"],
        epsilon=PSTORCH_CONFIG["epsilon"],
        num_probe=PSTORCH_CONFIG["num_probe"],
        chunk_size=PSTORCH_CONFIG["chunk_size"],
        compile=False,
        sinkhorn_max_iters=PSTORCH_CONFIG["sinkhorn_max_iters"],
    )

    criterion = nn.CrossEntropyLoss()
    buffers = dict(model.named_buffers())

    input_ids = torch.randint(0, 50257, (batch_size, max_seq_len), device=device)
    attention_mask = torch.ones(batch_size, max_seq_len, dtype=torch.long, device=device)
    labels = torch.randint(0, 2, (batch_size,), device=device)

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
    torch.cuda.synchronize()
    polystep_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    results["polystep_peak_mb"] = polystep_peak
    print(f"  polystep peak: {polystep_peak:.0f} MB")

    del optimizer, subspace, model
    gc.collect()
    torch.cuda.empty_cache()

    # -- Adam memory --
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

    adam_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    results["adam_peak_mb"] = adam_peak
    print(f"  Adam peak: {adam_peak:.0f} MB")

    del adam_opt, model
    gc.collect()
    torch.cuda.empty_cache()

    # Theoretical breakdown
    param_bytes = total_params * 4  # FP32
    param_mb = param_bytes / (1024 ** 2)

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


# ---------------------------------------------------------------------------
# Method Dispatch
# ---------------------------------------------------------------------------

METHOD_RUNNERS = {
    "polystep": run_polystep,
    "adam": run_adam,
}

HEADONLY_METHOD_RUNNERS = {
    "polystep": run_headonly_polystep,
    "adam": run_headonly_adam,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 124M Fine-Tuning on SST-2: polystep vs Adam"
    )
    parser.add_argument(
        "--methods", nargs="+", default=["polystep", "adam"],
        help="Methods to run (default: polystep adam)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42, 123, 456],
        help="Seeds to run (default: 42 123 456)",
    )
    parser.add_argument("--device", default="cuda", help="Device (default: cuda)")
    parser.add_argument(
        "--steps", type=int, default=NUM_STEPS,
        help=f"Number of polystep optimizer steps (default: {NUM_STEPS})",
    )
    parser.add_argument(
        "--subspace-dim", type=int, default=PSTORCH_CONFIG["subspace_dim"],
        help=f"Subspace dimensionality (default: {PSTORCH_CONFIG['subspace_dim']})",
    )
    parser.add_argument(
        "--results-dir", default="experiments/results",
        help="Results directory (default: experiments/results)",
    )
    parser.add_argument(
        "--measure-memory", action="store_true",
        help="Run memory profiling only (no training)",
    )
    parser.add_argument(
        "--head-only", action="store_true",
        help="Train only classifier head (1,538 params) with frozen backbone",
    )
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Determine benchmark mode
    benchmark = HEADONLY_BENCHMARK if args.head_only else BENCHMARK
    mode_label = "Head-Only" if args.head_only else "Full"

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
            output_file = os.path.join(
                args.results_dir, f"{benchmark}_{method}_{seed}.json"
            )
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
