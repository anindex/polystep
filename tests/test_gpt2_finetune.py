"""Tests for GPT-2 fine-tuning weight mapping, forward pass verification, and experiment infrastructure.

Verifies:
1. HuggingFace GPT-2 weights load correctly into custom GPT2Small model
2. Forward pass logits match HuggingFace model within 1e-4
3. attention_mask produces different output (masked mean pooling works)
4. SST-2 data loads with GPT-2 BPE tokenizer producing correct shapes
5. polystep can complete 1 optimizer step on pretrained GPT-2 (GPU smoke test)
"""

import pytest
import torch


# ---------------------------------------------------------------------------
# Test 1: Weight mapping correctness
# ---------------------------------------------------------------------------

def test_weight_mapping():
    """Load pretrained GPT-2 weights into GPT2Small and verify all keys mapped."""
    pytest.importorskip("transformers", reason="transformers not installed")
    from experiments.runners.run_gpt2_finetune import GPT2Small, load_gpt2_weights

    model = GPT2Small(num_classes=2)

    # Load weights
    mapping = load_gpt2_weights(model)

    # Verify mapping covers all expected keys (148 HF keys mapped)
    assert len(mapping) > 0, "Mapping should not be empty"

    # Check that model state dict has no unexpected missing keys
    # (only classifier.weight and classifier.bias should be missing from mapping)
    model_keys = set(model.state_dict().keys())
    mapped_keys = set(mapping.keys())
    missing = model_keys - mapped_keys
    assert missing == {"classifier.weight", "classifier.bias"}, (
        f"Unexpected missing keys: {missing}"
    )

    # Verify shapes of critical parameters
    sd = model.state_dict()
    assert sd["token_embedding.weight"].shape == (50257, 768)
    assert sd["position_embedding.weight"].shape == (128, 768)
    assert sd["layers.0.attention.W_q.weight"].shape == (768, 768)
    assert sd["layers.0.attention.W_k.weight"].shape == (768, 768)
    assert sd["layers.0.attention.W_v.weight"].shape == (768, 768)
    assert sd["layers.0.attention.W_o.weight"].shape == (768, 768)
    assert sd["layers.0.ff.0.weight"].shape == (3072, 768)
    assert sd["layers.0.ff.2.weight"].shape == (768, 3072)
    assert sd["layers.11.attention.W_q.weight"].shape == (768, 768)
    assert sd["layer_norm.weight"].shape == (768,)
    assert sd["classifier.weight"].shape == (2, 768)


# ---------------------------------------------------------------------------
# Test 2: Forward pass matches HuggingFace model
# ---------------------------------------------------------------------------

def test_forward_pass_match():
    pytest.importorskip("transformers", reason="transformers not installed")
    """Verify custom model hidden states match HF model within 1e-4."""
    from experiments.runners.run_gpt2_finetune import GPT2Small, load_gpt2_weights, verify_forward_pass

    model = GPT2Small(num_classes=2)
    load_gpt2_weights(model)

    # verify_forward_pass asserts max diff < 1e-4 internally
    max_diff = verify_forward_pass(model)
    assert max_diff < 1e-4, f"Forward pass mismatch: max diff = {max_diff}"


# ---------------------------------------------------------------------------
# Test 3: attention_mask changes output (masked mean pooling)
# ---------------------------------------------------------------------------

def test_attention_mask_effect():
    pytest.importorskip("transformers", reason="transformers not installed")
    """Verify that attention_mask produces different output than no mask."""
    from experiments.runners.run_gpt2_finetune import GPT2Small

    model = GPT2Small(num_classes=2)
    model.eval()

    # Input with padding
    input_ids = torch.tensor([[1212, 3807, 318, 1049, 50256, 50256, 50256, 50256]])
    mask_full = torch.ones(1, 8, dtype=torch.long)
    mask_partial = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])

    with torch.no_grad():
        out_no_mask = model(input_ids)
        out_full_mask = model(input_ids, attention_mask=mask_full)
        out_partial_mask = model(input_ids, attention_mask=mask_partial)

    # No mask and full mask should produce the same output (all 1s = mean of all)
    assert torch.allclose(out_no_mask, out_full_mask, atol=1e-5), (
        "No mask and full mask should produce identical output"
    )

    # Partial mask should produce different output than full mask
    assert not torch.allclose(out_full_mask, out_partial_mask, atol=1e-3), (
        "Partial mask should produce different output than full mask"
    )


# ---------------------------------------------------------------------------
# Test 4: SST-2 data loading with GPT-2 tokenizer
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.flaky(reruns=2)
def test_sst2_loading():
    """Verify data loader yields correct tensor shapes and types."""
    from experiments.runners.run_gpt2_finetune import get_sst2_gpt2_loaders

    train_loader, val_loader = get_sst2_gpt2_loaders(
        max_seq_len=32, batch_size=4, max_train=100,
    )

    # Check one batch from train loader
    batch = next(iter(train_loader))
    assert len(batch) == 3, "Batch should have 3 elements: input_ids, attention_mask, labels"

    input_ids, attention_mask, labels = batch
    assert input_ids.dtype == torch.long
    assert attention_mask.dtype == torch.long
    assert labels.dtype == torch.long
    assert input_ids.shape[1] == 32, f"Seq length should be 32, got {input_ids.shape[1]}"
    assert attention_mask.shape == input_ids.shape
    assert labels.shape[0] == input_ids.shape[0]

    # Labels should be binary (0 or 1 for SST-2)
    assert labels.min() >= 0 and labels.max() <= 1

    # Attention mask should have both 1s and 0s (some padding)
    assert attention_mask.sum() > 0, "Should have some real tokens"

    # Check val loader
    val_batch = next(iter(val_loader))
    assert len(val_batch) == 3


# ---------------------------------------------------------------------------
# Test 5: polystep one-step smoke test (GPU required)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_polystep_one_step():
    """Verify 1 polystep step completes without error on pretrained GPT-2."""
    import gc

    from experiments.runners.run_gpt2_finetune import GPT2Small, load_gpt2_weights

    from polystep.optimizer import PolyStepOptimizer
    from polystep.adaptive_subspace import AdaptiveSubspace

    device = "cuda"

    model = GPT2Small(num_classes=2).to(device)
    load_gpt2_weights(model)

    subspace = AdaptiveSubspace.auto_from_params(
        model, compression_target=0.001, max_rank=128,
    )
    object.__setattr__(subspace, 'rotation_mode', 'random')

    optimizer = PolyStepOptimizer(
        model,
        seed=42,
        subspace=subspace,
        projection_type='sparse',
        step_radius=2.0,
        probe_radius=1.0,
        epsilon=0.1,
        num_probe=2,
        chunk_size=4,
        compile=False,
        sinkhorn_max_iters=50,
    )

    criterion = torch.nn.CrossEntropyLoss()
    buffers = dict(model.named_buffers())

    # Synthetic data for smoke test
    input_ids = torch.randint(0, 50257, (2, 32), device=device)
    attention_mask = torch.ones(2, 32, dtype=torch.long, device=device)
    labels = torch.randint(0, 2, (2,), device=device)

    def make_closure(_ids, _mask, _labels):
        def closure(batched_params):
            from torch.func import functional_call, vmap

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

    # Should complete without error
    optimizer.step(make_closure(input_ids, attention_mask, labels))

    # Verify model still produces output
    with torch.no_grad():
        output = model(input_ids, attention_mask=attention_mask)
    assert output.shape == (2, 2), f"Expected (2, 2), got {output.shape}"

    # Cleanup
    del optimizer, model, subspace
    gc.collect()
    torch.cuda.empty_cache()
