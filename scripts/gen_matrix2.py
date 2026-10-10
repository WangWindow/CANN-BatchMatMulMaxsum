#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二轮全矩阵（扩展等价类）。在 build/ 目录下执行：

    python3 ../scripts/gen_matrix2.py

产物：input/mx2*/…, output/mx2*/golden_y.bin, matrix2_cases.txt
覆盖：B 边界族(接近/跨越 blockCount=20 的倍数)、大 K 奇数值、超大形状组合、
      B>64 通用性、tx/dtype 在奇 K 上的交叉。
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

CASES = []
_counter = [200]


def add(note, b, m, n, k, dtype=1, tx1=False, tx2=False, data="uniform"):
    name = f"mx2{_counter[0]:03d}"
    _counter[0] += 1
    CASES.append((name, note, b, m, n, k, dtype, tx1, tx2, data))


def build_H():
    # B 边界族：B 与 blockCount=20 的倍数关系
    for b in [39, 40, 41, 59, 60, 61, 63]:
        add(f"H B={b} multi-tile", b, 96, 192, 72)
    # B>64 通用性（守卫放宽后）
    for b in [65, 100, 128, 256]:
        add(f"H B={b} tiny", b, 4, 8, 8)
        add(f"H B={b} k33", b, 65, 33, 33)


def build_I():
    # 大 K 奇数值/非对齐
    for k in [511, 513, 1023, 1025, 2047, 4095, 8191]:
        add(f"I K={k} M=129 N=65", 2, 129, 65, k)
    # 大 K 的 tx/dtype 交叉
    for k in [513, 4095]:
        for dt in (1, 2):
            for tx1 in (False, True):
                for tx2 in (False, True):
                    add(f"I K={k} dt={dt} tx=({int(tx1)},{int(tx2)})", 1, 65, 129, k,
                        dtype=dt, tx1=tx1, tx2=tx2)


def build_J():
    # 超大形状组合（1 batch 多 tile 压力）
    add("J 8192x256x65", 1, 8192, 256, 65)
    add("J 256x8192x65", 1, 256, 8192, 65)
    add("J 4096x128x129", 2, 4096, 128, 129)
    add("J 1024^2x33", 4, 1024, 1024, 33)
    add("J 512^2x72 B=16", 16, 512, 512, 72)
    add("J 4096^2x33", 1, 4096, 4096, 33)
    add("J 256^2x65 B=32", 32, 256, 256, 65)
    add("J 512x64x129 B=64", 64, 512, 64, 129)
    # 核数边界：每 block tile 数极端多
    add("J many-tiles 8192x8192x33", 1, 8192, 8192, 33)
    # 极小 + 大 K
    add("J tiny 1x1x8191", 1, 1, 1, 8191)
    add("J tiny 3x5x513", 1, 3, 5, 513)


def build_K():
    # 奇 K 上的数据特性
    for mode in ["neg", "zero", "extreme"]:
        add(f"K data={mode} K=65", 2, 129, 129, 65, data=mode)


def golden_affordable(b, m, n, k):
    return b * m * n * k <= (1 << 30)


def make_input(shape, dtype, rng, mode):
    if mode == "uniform":
        base = rng.uniform(-1.0, 1.0, size=shape)
    elif mode == "neg":
        base = rng.uniform(0.1, 1.0, size=shape)
    elif mode == "zero":
        base = np.zeros(shape)
    elif mode == "extreme":
        base = rng.choice(np.array([-1.0, 1.0, 0.0]), size=shape)
    else:
        raise ValueError(mode)
    if dtype == 1:
        return base.astype(np.float16)
    return base.astype(bfloat16)


def physical_shapes(b, m, n, k, tx1, tx2):
    x1 = (b, k, m) if tx1 else (b, m, k)
    x2 = (b, n, k) if tx2 else (b, k, n)
    return x1, x2


def main():
    build_H()
    build_I()
    build_J()
    build_K()

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    lines, manifest = [], ["name\tnote\tB\tM\tN\tK\tdtype\ttx1\ttx2\tgolden"]
    for name, note, b, m, n, k, dtype, tx1, tx2, data in CASES:
        if dtype == 2 and bfloat16 is None:
            continue
        seed = 20261201 + sum(ord(ch) for ch in name)
        rng = np.random.default_rng(seed)
        x1_shape, x2_shape = physical_shapes(b, m, n, k, tx1, tx2)
        x1 = make_input(x1_shape, dtype, rng, data)
        if data == "neg":
            x2 = (-rng.uniform(0.1, 1.0, size=x2_shape)).astype(
                np.float16 if dtype == 1 else bfloat16)
        else:
            x2 = make_input(x2_shape, dtype, rng, data)

        case_dir = os.path.join("input", name)
        os.makedirs(case_dir, exist_ok=True)
        out_dir = os.path.join("output", name)
        os.makedirs(out_dir, exist_ok=True)
        x1.tofile(os.path.join(case_dir, "x1.bin"))
        x2.tofile(os.path.join(case_dir, "x2.bin"))

        gold = golden_affordable(b, m, n, k)
        if gold:
            y = impl(x1, x2, transposeX1=tx1, transposeX2=tx2)
            y.astype(np.float32).tofile(os.path.join(out_dir, "golden_y.bin"))
        lines.append(f"{name} {b} {m} {n} {k} {dtype} {1 if tx1 else 0} "
                     f"{1 if tx2 else 0} {1 if gold else 0}")
        manifest.append(f"{name}\t{note}\t{b}\t{m}\t{n}\t{k}\t{dtype}\t"
                        f"{1 if tx1 else 0}\t{1 if tx2 else 0}\t{1 if gold else 0}")
        print(f"case {name}: B={b} M={m} N={n} K={k} dt={dtype} golden={gold}")

    with open("matrix2_cases.txt", "w") as f:
        f.write("\n".join(lines) + "\n")
    with open("matrix2_manifest.tsv", "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"Generated {len(lines)} cases -> matrix2_cases.txt")


if __name__ == "__main__":
    main()
