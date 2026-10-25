#!/usr/bin/env python3
"""Compare two scanner-geometry .tensor files (or any torch.save'd objects).

Prints IDENTICAL when the files are byte-identical, otherwise walks both objects
and prints the largest absolute difference over all tensors, and MISMATCH if
the structure differs or any difference exceeds --atol.

Usage: python compare_tensors.py NEW.tensor REFERENCE.tensor [--atol 1e-9]
"""
import argparse
import filecmp
import sys

import torch


def walk(a, b, path, out):
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)) or a.shape != b.shape:
            out["structure"].append(path)
            return
        if a.numel():
            out["max"] = max(out["max"], float((a.double() - b.double()).abs().max()))
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            out["structure"].append(path)
        for k in set(a) & set(b):
            walk(a[k], b[k], f"{path}/{k}", out)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out["structure"].append(path)
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", out)
    elif a != b:
        out["structure"].append(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("new")
    ap.add_argument("ref")
    # Coordinates are float32 millimetres up to about 325 mm, where one float32
    # step is 3e-5 mm; ARM and x86 round the same geometry differently at that
    # level (step 1 on Grace: largest difference 3.8e-6 mm).
    ap.add_argument("--atol", type=float, default=1e-4)
    a = ap.parse_args()
    if filecmp.cmp(a.new, a.ref, shallow=False):
        print("IDENTICAL (byte for byte)")
        return
    out = {"max": 0.0, "structure": []}
    walk(torch.load(a.new, map_location="cpu", weights_only=False),
         torch.load(a.ref, map_location="cpu", weights_only=False), "", out)
    ok = not out["structure"] and out["max"] <= a.atol
    print(f"{'MATCH' if ok else 'MISMATCH'}: largest difference {out['max']:.3e}"
          + (f"; structure differs at {out['structure'][:5]}" if out["structure"] else ""))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
