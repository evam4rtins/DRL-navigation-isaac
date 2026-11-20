"""
Cost of a frozen DINOv2 encoder on a batch of camera images.

For every model and batch size: forward time (ms), images/s and peak GPU memory, in fp32 and in bf16 autocast.
The batch size is the number of robots: every env step encodes one 224x224 image per robot.

"""

import argparse
import time
from pathlib import Path

import timm
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+",
                    default=["vit_small_patch14_dinov2.lvd142m", "vit_base_patch14_dinov2.lvd142m"])
parser.add_argument("--batches", nargs="+", type=int, default=[64, 128, 256])
parser.add_argument("--res", type=int, default=224)
parser.add_argument("--iters", type=int, default=20)
args = parser.parse_args()

device = "cuda"


def bench(model, batch, dtype):
    x = torch.rand(batch, 3, args.res, args.res, device=device)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dtype == "bf16"):
        for _ in range(3):
            tokens = model.forward_features(x)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(args.iters):
            tokens = model.forward_features(x)
        torch.cuda.synchronize()
    ms = (time.time() - t0) / args.iters * 1000
    return ms, torch.cuda.max_memory_allocated() / 2**30, tuple(tokens.shape)


def main():
    print(f"GPU: {torch.cuda.get_device_name()}  image {args.res}x{args.res}\n")
    print(f"{'model':42} {'batch':>5} {'dtype':>5} {'ms':>8} {'img/s':>8} {'peak GB':>8}  tokens")
    for name in args.models:
        model = timm.create_model(name, pretrained=True, num_classes=0, img_size=args.res,
                                  cache_dir=Path("~/.cache/torch/hub/timm").expanduser())
        model = model.to(device).eval().requires_grad_(False)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"{name} ({n_params:.0f} M parameters, {torch.cuda.memory_allocated() / 2**30:.2f} GB weights)")
        for batch in args.batches:
            for dtype in ("fp32", "bf16"):
                try:
                    ms, peak, shape = bench(model, batch, dtype)
                    print(f"{'':42} {batch:5d} {dtype:>5} {ms:8.1f} {batch / ms * 1000:8.0f} {peak:8.2f}  {shape}")
                except torch.cuda.OutOfMemoryError:
                    print(f"{'':42} {batch:5d} {dtype:>5}  out of memory")
                    torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()
    print("\ntokens = (batch, 1 cls [+ registers] + (res/14)^2 patches, embed dim): what the adapter's queries attend to")


if __name__ == "__main__":
    main()