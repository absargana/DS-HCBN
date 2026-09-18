from __future__ import annotations

import argparse, json, platform, statistics, time
from pathlib import Path
import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity


def load_model(encoder, checkpoint=None):
    from dshcbn.models import get_model_class
    model = get_model_class(encoder)()
    if checkpoint:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(ck.get("model_state", ck), strict=True)
    return model


def one(name, model, device, warmup, repeats, amp):
    model = model.to(device).eval(); xl = torch.zeros(1, 1, 64, 64, 64, device=device); xc = torch.zeros(1, 1, 96, 96, 96, device=device)
    ctx = lambda: torch.autocast("cuda", dtype=torch.float16, enabled=amp and device.type == "cuda")
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device.type == "cuda" else [])
    with torch.no_grad():
        for _ in range(warmup):
            with ctx(): model(xl, xc)
        with profile(activities=activities, with_flops=True) as prof:
            with ctx(): model(xl, xc)
            if device.type == "cuda": torch.cuda.synchronize()
        counted_flops = int(sum(int(e.flops or 0) for e in prof.key_averages()))
        if device.type == "cuda": torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            with ctx(): model(xl, xc)
            if device.type == "cuda": torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
    return {"architecture": name, "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "latency_ms_median": float(statistics.median(times)), "latency_ms_q1": float(np.quantile(times, .25)),
            "latency_ms_q3": float(np.quantile(times, .75)), "repetitions": repeats,
            "peak_gpu_memory_mb": float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else None,
            "profiler_counted_flops": counted_flops, "profiler_counted_gflops": float(counted_flops / 1e9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet_checkpoint"); ap.add_argument("--cnn_checkpoint")
    ap.add_argument("--out_dir", required=True); ap.add_argument("--warmup", type=int, default=10); ap.add_argument("--repeats", type=int, default=50)
    ap.add_argument("--cpu", action="store_true"); ap.add_argument("--no_amp", action="store_true"); a = ap.parse_args()
    device = torch.device("cpu" if a.cpu or not torch.cuda.is_available() else "cuda")
    rows = [one("Residual CNN", load_model("residual_cnn", a.cnn_checkpoint), device, a.warmup, a.repeats, not a.no_amp),
            one("U-Net encoder", load_model("unet", a.unet_checkpoint), device, a.warmup, a.repeats, not a.no_amp)]
    meta = {"input_shapes": [[1,1,64,64,64],[1,1,96,96,96]], "batch_size": 1, "weights": "frozen trained checkpoints",
            "device": str(device), "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None, "torch": torch.__version__,
            "cuda": torch.version.cuda, "python": platform.python_version(), "amp": bool(not a.no_amp and device.type == "cuda"), "models": rows,
            "flops_note": "PyTorch-profiler FLOPs for supported operators; reported as a reproducible lower bound because some 3D/attention operators may not expose FLOP metadata. Latency and peak memory are measured directly."}
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True); json.dump(meta, open(out / "complexity_benchmark.json", "w"), indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__": main()
