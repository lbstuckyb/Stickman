r"""Is torch actually going to use this machine's GPU? Run this first on a new box.

    python check_gpu.py              (Windows: .venv\Scripts\python check_gpu.py)

Imports torch and nothing else - no rfdetr, no 156 MB checkpoint download - so a broken
install shows up in two seconds instead of after the model loads. Prints the device report,
then times a big matmul in FP32 and FP16 to prove the tensor cores are really being used.
Exits non-zero (with the fix) when CUDA was expected and is not there.
"""

from __future__ import annotations

import argparse
import sys
import time

from rfdetr_pose import device_report, format_device_report, no_cuda_explanation


def bench_matmul(torch, dtype, n: int = 4096, iters: int = 20) -> float:
    """Median ms for one n*n matmul on the GPU, kernels synchronized."""
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(3):                      # warm up: allocator, autotune, clocks
        a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        a @ b
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=4096, help="matmul side length")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="exit 0 even when no GPU is found")
    args = ap.parse_args()

    import torch

    report = device_report()
    print(f"device     : {format_device_report(report)}")
    print(f"torch      : {report['torch']}  CUDA build: {report['torch_cuda_build']}")
    print(f"available  : {report['cuda_available']}")

    if not report["cuda_available"]:
        print(f"\nFAIL: {no_cuda_explanation(report)}")
        return 0 if args.allow_cpu else 1

    n = args.size
    fp32 = bench_matmul(torch, torch.float32, n)
    fp16 = bench_matmul(torch, torch.float16, n)
    flop = 2 * n ** 3
    print(f"\nmatmul {n}x{n}:")
    print(f"  fp32     : {fp32:7.2f} ms  ({flop / fp32 / 1e9:6.1f} TFLOP/s)")
    print(f"  fp16     : {fp16:7.2f} ms  ({flop / fp16 / 1e9:6.1f} TFLOP/s)  "
          f"{fp32 / fp16:.1f}x faster")
    free, total = torch.cuda.mem_get_info()
    print(f"  memory   : {(total - free) / 1024**3:.1f} / {total / 1024**3:.1f} GB in use")

    if fp16 > fp32:
        print("\nWARNING: FP16 is not faster than FP32 here - tensor cores are not being "
              "used, so --half will not help either.")
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
