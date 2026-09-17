"""Original DVS Gesture preprocessing and exactly hard backward controls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import tarfile
import time

import numpy as np
import torch
from torch import nn

from experiments.runners.nondiff_models import LIFNeuron, SpikingMNISTNet, QuantizedMLP, QuantizedLinear


class SpikeSurrogate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage, slope):
        ctx.save_for_backward(voltage)
        ctx.slope = slope
        return (voltage >= 1).float()

    @staticmethod
    def backward(ctx, grad):
        (voltage,) = ctx.saved_tensors
        return grad * ctx.slope / (1 + ctx.slope * (voltage - 1).abs()).square(), None


class SurrogateLIF(LIFNeuron):
    def __init__(self, slope):
        super().__init__()
        self.slope = slope

    def forward(self, current, membrane):
        voltage = 0.95 * membrane + current
        spike = SpikeSurrogate.apply(voltage, self.slope)
        return spike, voltage * (1 - spike.detach())


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, scale):
        ctx.save_for_backward(weight)
        ctx.scale = scale
        return torch.clamp(torch.round(weight / scale), -128, 127) * scale

    @staticmethod
    def backward(ctx, grad):
        (weight,) = ctx.saved_tensors
        return grad * ((weight >= -128 * ctx.scale) & (weight <= 127 * ctx.scale)), None


class GestureNet(nn.Module):
    def __init__(self, slope=None):
        super().__init__()
        self.fc1 = nn.Linear(2048, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, 11, bias=False)
        self.lif1 = LIFNeuron() if slope is None else SurrogateLIF(slope)
        self.lif2 = LIFNeuron() if slope is None else SurrogateLIF(slope)

    def forward(self, x):
        mem1, mem2 = x.new_zeros(len(x), 128), x.new_zeros(len(x), 64)
        total = x.new_zeros(len(x), 11)
        for t in range(25):
            spike1, mem1 = self.lif1(self.fc1(x[:, t].reshape(len(x), 2048)), mem1)
            spike2, mem2 = self.lif2(self.fc2(spike1), mem2)
            total = total + self.fc3(spike2)
        return total / 25


def hard_model(task, slope=None, ste=False):
    if task == "dvs":
        return GestureNet(slope)
    if task == "snn":
        model = SpikingMNISTNet()
        if slope is not None:
            model.lif1, model.lif2 = SurrogateLIF(slope), SurrogateLIF(slope)
        return model
    if task == "int8":
        model = QuantizedMLP()
        if ste:
            for layer in model.modules():
                if isinstance(layer, QuantizedLinear):
                    layer.polystep_weight_transform = lambda weight, scale=layer.scale: RoundSTE.apply(weight, scale)
                    layer.polystep_bias_transform = layer.polystep_weight_transform
        return model
    raise ValueError(task)


def read_aedat(path):
    """AEDAT 3.1 polarity packets, with validity bits and timestamp overflow."""
    times, addresses = [], []
    with Path(path).open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line.startswith(b"#"):
                f.seek(pos)
                break
            if line.strip() == b"#!END-HEADER":
                break
        while header := f.read(28):
            if len(header) != 28:
                raise ValueError(f"truncated packet header: {path}")
            kind, source, size, ts_offset, overflow, capacity, number, valid = struct.unpack("<HHIIIIII", header)
            if size == 0 or size > 1024 or number > capacity:
                raise ValueError(f"invalid packet dimensions: {path}")
            payload = f.read(size * capacity)
            if len(payload) != size * capacity:
                raise ValueError(f"truncated packet: {path}")
            if kind != 1:
                continue
            if size % 4 or ts_offset % 4 or ts_offset + 4 > size:
                raise ValueError(f"invalid polarity event layout: {path}")
            data = np.frombuffer(payload, dtype="<u4").reshape(capacity, size // 4)[:number]
            address = data[:, 0]
            keep = address & 1 != 0
            if int(keep.sum()) != valid:
                raise ValueError(f"valid-event count mismatch: {path}")
            timestamp = data[:, ts_offset // 4].astype(np.int64) + (int(overflow) << 31)
            addresses.append(address[keep])
            times.append(timestamp[keep])
    if not times:
        return np.empty(0, np.int64), np.empty(0, np.uint32)
    t, address = np.concatenate(times), np.concatenate(addresses)
    if np.any(t[1:] < t[:-1]):
        order = np.argsort(t, kind="stable")
        t, address = t[order], address[order]
    return t, address


def bin_interval(t, address, start, stop):
    if stop <= start:
        raise ValueError("nonpositive labeled interval")
    lo, hi = np.searchsorted(t, [start, stop])
    times, addr = t[lo:hi], address[lo:hi]
    x, y, polarity = (addr >> 17) & 0x7FFF, (addr >> 2) & 0x7FFF, (addr >> 1) & 1
    if np.any(x >= 128) or np.any(y >= 128):
        raise ValueError("event coordinate outside DVS128 sensor")
    bins = (25 * (times - start) // (stop - start)).astype(np.int64)
    output = np.zeros((25, 2, 32, 32), dtype=np.uint8)
    output[bins, polarity, y // 4, x // 4] = 1
    return output


def preprocess(archive, destination):
    started = time.perf_counter()
    archive, destination = Path(archive), Path(destination)
    with archive.open("rb") as f:
        digest = hashlib.file_digest(f, "md5").hexdigest()
    if digest != "8a5c71fb11e24e5ca5b11866ca6c00a1":
        raise ValueError(f"original archive checksum mismatch: {digest}")
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as tar:
        tar.extractall(destination, filter="data")
    raw = destination / "DvsGesture"
    train_files = (raw / "trials_to_train.txt").read_text().split()
    test_files = (raw / "trials_to_test.txt").read_text().split()

    def subject(name):
        return int(re.match(r"user(\d+)_", name).group(1))

    train_subjects, test_subjects = sorted({subject(f) for f in train_files}), sorted({subject(f) for f in test_files})
    assert len(train_subjects) == 23 and len(test_subjects) == 6
    assert not set(train_subjects) & set(test_subjects)
    val_subjects = train_subjects[-5:]
    records = {name: [] for name in ("train", "validation", "test")}
    frames = {name: [] for name in records}
    for filename in train_files + test_files:
        user = subject(filename)
        split = "test" if user in test_subjects else "validation" if user in val_subjects else "train"
        t, address = read_aedat(raw / filename)
        labels = np.loadtxt(
            raw / filename.replace(".aedat", "_labels.csv"), delimiter=",", skiprows=1, dtype=np.int64, ndmin=2
        )
        for interval, (label, start, stop) in enumerate(labels):
            assert 1 <= label <= 11
            frame = bin_interval(t, address, int(start), int(stop))
            frames[split].append(frame)
            records[split].append(
                dict(
                    file=filename,
                    interval=interval,
                    subject=user,
                    label=int(label - 1),
                    start_us=int(start),
                    stop_us=int(stop),
                    occupied=int(frame.sum()),
                )
            )
        print(filename, len(labels), flush=True)
    for split in records:
        targets = np.array([r["label"] for r in records[split]], dtype=np.int64)
        assert set(targets.tolist()) == set(range(11))
        np.savez_compressed(destination / f"{split}.npz", x=np.stack(frames[split]), y=targets)
    metadata = dict(
        archive_md5=digest,
        train_subjects=train_subjects[:-5],
        validation_subjects=val_subjects,
        test_subjects=test_subjects,
        records=records,
        preprocessing_seconds=time.perf_counter() - started,
        source="https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794",
        protocol="all labeled intervals; [start,end); 25 equal-duration bins; binary polarity occupancy at 32x32",
    )
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print({split: len(records[split]) for split in records})


def self_check():
    for task in ("snn", "int8", "dvs"):
        torch.manual_seed(9)
        hard = hard_model(task)
        torch.manual_seed(9)
        backward = hard_model(task, slope=5 if task != "int8" else None, ste=task == "int8")
        x = torch.rand(2, 25, 2, 32, 32) if task == "dvs" else torch.randn(2, 784)
        assert all(torch.equal(v, backward.state_dict()[k]) for k, v in hard.state_dict().items())
        assert torch.equal(hard(x), backward(x))
        torch.nn.functional.cross_entropy(backward(x), torch.tensor([0, 1])).backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in backward.parameters())
    v = torch.tensor([0.5, 1.0, 1.5], requires_grad=True)
    SpikeSurrogate.apply(v, 5).sum().backward()
    assert torch.allclose(v.grad, 5 / (1 + 5 * (v.detach() - 1).abs()).square())
    t = np.array([100, 104, 199, 200])
    address = np.array([1, 1, (127 << 17) + (127 << 2) + 3, 1], np.uint32)
    frame = bin_interval(t, address, 100, 200)
    assert frame.sum() == 3 and frame[24, 1, 31, 31] == 1
    assert bin_interval(t, address, 300, 400).sum() == 0
    print("hard-forward, backward, and event-bin checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--archive")
    parser.add_argument("--destination", default="data/dvs_controlled")
    args = parser.parse_args()
    if args.archive:
        preprocess(args.archive, args.destination)
    else:
        self_check()
