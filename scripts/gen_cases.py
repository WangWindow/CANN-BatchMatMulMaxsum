#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成本地自测用例：输入 bin、golden 与 cases.txt 清单。

用法（在 build/ 目录下执行）：
    python3 ../scripts/gen_cases.py

产物：
    input/<name>/x1.bin, x2.bin
    output/<name>/golden_y.bin   （仅 correctness 用例）
    cases.txt                    每行： name B M N K dtype tx1 tx2
                                  dtype: 1=fp16 2=bf16
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from BatchMatmulMaxSum import impl

try:
    from ml_dtypes import bfloat16
except ImportError:
    bfloat16 = None


def make_input(shape, dtype, rng, low=-1.0, high=1.0):
    val = rng.uniform(low, high, size=shape).astype(np.float32)
    if dtype == 1:
        return val.astype(np.float16)
    return val.astype(bfloat16)


def physical_shapes(b, m, n, k, tx1, tx2):
    x1 = (b, k, m) if tx1 else (b, m, k)
    x2 = (b, n, k) if tx2 else (b, k, n)
    return x1, x2


# (name, B, M, N, K, dtype, tx1, tx2, golden)
# golden=False 的为 perf 用例：不生成 golden，只测速 + 有限性检查。
CASES = [
    # ---- correctness：四种 transpose 组合 ----
    ("c00", 1, 1, 1, 32, 1, False, False, True),
    ("c01", 2, 64, 64, 128, 1, False, False, True),
    ("c02", 2, 64, 64, 128, 1, False, True, True),
    ("c03", 2, 64, 64, 128, 1, True, False, True),
    ("c04", 2, 64, 64, 128, 1, True, True, True),
    # ---- 非对齐尾块 ----
    ("c05", 1, 17, 13, 32, 1, False, False, True),
    ("c06", 4, 129, 257, 64, 2, True, False, True),
    # ---- B=64 ----
    ("c07", 64, 32, 64, 32, 1, False, False, True),
    # ---- 中等规模 ----
    ("c08", 1, 2048, 2048, 2048, 1, False, False, True),
    ("c09", 1, 100, 50, 1024, 2, False, True, True),
    ("c10", 8, 256, 128, 512, 1, True, True, True),
    # ---- 大 M/N、小 K（访存 bound） ----
    ("c11", 1, 4096, 4096, 32, 1, False, False, True),
    ("c12", 1, 4096, 64, 4096, 1, True, False, True),
    # ---- perf：只测速，无 golden ----
    ("p00", 1, 8192, 8192, 8192, 1, False, False, False),
    ("p01", 1, 8192, 8192, 8192, 2, True, True, False),
    ("p02", 64, 1024, 1024, 1024, 1, False, False, False),
    ("p03", 2, 4096, 4096, 8192, 1, False, False, False),
    ("p04", 1, 8192, 8192, 32, 1, True, False, False),
    ("p05", 4, 2048, 2048, 2048, 2, False, False, False),
]


def main():
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    lines = []
    for name, b, m, n, k, dtype, tx1, tx2, golden in CASES:
        if dtype == 2 and bfloat16 is None:
            print(f"ERROR: ml_dtypes not installed, skip bf16 case {name}")
            continue
        # 稳定的 seed：与进程无关（python hash() 每次进程随机化，不可用于复现）
        seed = 20261001 + sum(ord(ch) for ch in name)
        rng = np.random.default_rng(seed)
        x1_shape, x2_shape = physical_shapes(b, m, n, k, tx1, tx2)
        # 限制约束：B*M*K 与 B*N*K 均 <= 2^26
        assert b * m * k <= (1 << 26) and b * n * k <= (1 << 26), name

        case_dir = os.path.join("input", name)
        os.makedirs(case_dir, exist_ok=True)
        out_dir = os.path.join("output", name)
        os.makedirs(out_dir, exist_ok=True)
        x1 = make_input(x1_shape, dtype, rng)
        x2 = make_input(x2_shape, dtype, rng)
        x1.tofile(os.path.join(case_dir, "x1.bin"))
        x2.tofile(os.path.join(case_dir, "x2.bin"))

        if golden:
            y = impl(x1, x2, transposeX1=tx1, transposeX2=tx2)
            y.astype(np.float32).tofile(os.path.join(out_dir, "golden_y.bin"))
            print(f"case {name}: input={x1_shape}x{x2_shape} golden={y.shape} range=[{y.min():.6f},{y.max():.6f}]")
        else:
            print(f"case {name}: input={x1_shape}x{x2_shape} (perf only)")

        lines.append(f"{name} {b} {m} {n} {k} {dtype} {1 if tx1 else 0} {1 if tx2 else 0} {1 if golden else 0}")

    with open("cases.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Generated {len(lines)} cases -> cases.txt")


if __name__ == "__main__":
    main()
