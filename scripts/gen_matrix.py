#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成全矩阵测试用例（TDD 基线，与判题无关的通用覆盖）。

用法（在 build/ 目录下执行）：
    python3 ../scripts/gen_matrix.py          # 生成全部组
    python3 ../scripts/gen_matrix.py A        # 只生成 A 组（K 轴）
    python3 ../scripts/gen_matrix.py A B C    # 生成多组

产物：
    input/mx*/x1.bin, x2.bin
    output/mx*/golden_y.bin        （对可承受的形状）
    matrix_cases.txt               每行： name B M N K dtype tx1 tx2 golden
    matrix_manifest.tsv            name 组 说明（人工核对用）

设计原则：等价类覆盖（K 的 16 分形余类、M/N 尾块、B 相对核数、
dtype×transpose、数据特性、规模边界），而不是针对判题形状。
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

CASES = []  # (name, group, note, b, m, n, k, dtype, tx1, tx2, golden, data_mode)
_counter = [0]


def add(group, note, b, m, n, k, dtype=1, tx1=False, tx2=False, golden=True, data="uniform"):
    name = f"mx{_counter[0]:03d}"
    _counter[0] += 1
    CASES.append((name, group, note, b, m, n, k, dtype, tx1, tx2, golden, data))


# ---------------------------------------------------------------- A 组：K 轴
def build_A():
    # A1: 极小形状（对齐官方 case1 的量级：M=2,N=3）+ K 全扫描（含 k<32、k%16!=0）
    K1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 16, 17, 24, 31, 32, 33, 40, 48,
          56, 63, 64, 65, 72, 88, 104, 120, 128]
    for k in K1:
        add("A1", f"tiny M=2 N=3 K={k}", 1, 2, 3, k)
    # A2: 常规形状 + K 全余类
    K2 = [4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 88, 104, 120, 128, 136, 200]
    for k in K2:
        add("A2", f"M=64 N=64 K={k}", 2, 64, 64, k)
    # A3: 转置组合下的非对齐 K
    for k in [4, 16, 40, 72, 128]:
        add("A3", f"M=100 N=50 K={k} tx=(0,1)", 1, 100, 50, k, tx2=True)
        add("A4", f"M=65 N=129 K={k} tx=(1,0)", 4, 65, 129, k, tx1=True)
    # A5: bf16 非对齐 K
    for k in [4, 16, 40, 72, 128, 136]:
        add("A5", f"bf16 M=65 N=129 K={k} tx=(1,1)", 4, 65, 129, k, dtype=2, tx1=True, tx2=True)
    # A6: 多 chunk 尾块
    for k in [136, 200, 264, 328, 520, 776, 1032, 2056]:
        add("A6", f"M=96 N=160 K={k} (tail chunks)", 2, 96, 160, k)


# ---------------------------------------------------------- B/C/D 组：M/N/B 轴
def build_BCD():
    Mlist = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33, 48, 63, 64, 65,
             96, 100, 127, 128, 129, 192, 255, 256, 257]
    for m in Mlist:
        add("B", f"M={m} N=64 K=32", 2, m, 64, 32)
    Nlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 15, 16, 17, 31, 32, 33, 63, 64, 65,
             96, 127, 128, 129, 192, 255, 256, 257]
    for n in Nlist:
        add("C", f"M=64 N={n} K=32", 2, 64, n, 32)
    Blist = [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 19, 20, 21, 24, 32, 63, 64]
    for b in Blist:
        add("D", f"B={b} M=64 N=64 K=32", b, 64, 64, 32)


# ---------------------------------------------------------------- E 组：dtype×tx
def build_E():
    shapes = [
        ("tiny", 1, 8, 8, 4),
        ("small-odd", 2, 64, 64, 40),
        ("mid-odd", 4, 129, 257, 72),
        ("mid", 3, 256, 256, 128),
    ]
    for sname, b, m, n, k in shapes:
        for dtype in (1, 2):
            if dtype == 2 and bfloat16 is None:
                continue
            for tx1 in (False, True):
                for tx2 in (False, True):
                    add("E", f"{sname} dtype={dtype} tx=({int(tx1)},{int(tx2)})",
                        b, m, n, k, dtype=dtype, tx1=tx1, tx2=tx2)


# ---------------------------------------------------------------- F 组：数据特性
def build_F():
    shapes = [("F1", 2, 64, 64, 72), ("F2", 4, 129, 257, 64)]
    modes = ["uniform", "neg", "zero", "extreme", "small"]
    for sname, b, m, n, k in shapes:
        for mode in modes:
            add("F", f"{sname} data={mode}", b, m, n, k, data=mode)
    # 全负（对齐官方 Case 02 断言）
    add("F", "F3 neg M=300 N=300 K=300", 1, 300, 300, 300, data="neg")


# ---------------------------------------------------------------- G 组：规模边界
def build_G():
    # 边界：B*M*K 与 B*N*K 接近/到达 2^26
    add("G", "B*M*K near 2^26", 64, 8192, 128, 128)
    add("G", "B*N*K near 2^26", 64, 128, 8192, 128)
    add("G", "big B M=1024 N=128 K=512", 16, 1024, 128, 512)
    add("G", "wide N M=1 N=8192 K=128", 1, 1, 8192, 128)
    add("G", "tall M M=8192 N=1 K=128", 1, 8192, 1, 128)
    add("G", "deep K M=1 N=1 K=8192", 1, 1, 1, 8192)
    add("G", "big batch B=64 tiny", 64, 1, 1, 32)
    add("G", "M=129 N=129 K=129", 2, 129, 129, 129)
    add("G", "M=257 N=65 K=65", 2, 257, 65, 65)


# ---------------------------------------------------------------- 生成
def make_input(shape, dtype, rng, mode):
    if mode == "uniform":
        base = rng.uniform(-1.0, 1.0, size=shape)
    elif mode == "neg":
        # 让所有乘积为负：x1 全正、x2 全负 → y 全负（在 main() 中配对处理）
        base = rng.uniform(0.1, 1.0, size=shape)
    elif mode == "zero":
        base = np.zeros(shape)
    elif mode == "extreme":
        base = rng.choice(np.array([-1.0, 1.0, 0.0]), size=shape)
    elif mode == "small":
        base = rng.uniform(-0.01, 0.01, size=shape)
    else:
        raise ValueError(mode)
    if dtype == 1:
        return base.astype(np.float16)
    return base.astype(bfloat16)


def physical_shapes(b, m, n, k, tx1, tx2):
    x1 = (b, k, m) if tx1 else (b, m, k)
    x2 = (b, n, k) if tx2 else (b, k, n)
    return x1, x2


def golden_affordable(b, m, n, k):
    return b * m * n * k <= 1 << 30


def main():
    groups = sys.argv[1:] or ["A", "B", "C", "D", "E", "F", "G"]
    if "A" in groups or not sys.argv[1:]:
        build_A()
    if "B" in groups or not sys.argv[1:]:
        build_BCD()
    if "E" in groups or not sys.argv[1:]:
        build_E()
    if "F" in groups or not sys.argv[1:]:
        build_F()
    if "G" in groups or not sys.argv[1:]:
        build_G()

    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    lines = []
    manifest = ["name\tgroup\tnote\tB\tM\tN\tK\tdtype\ttx1\ttx2\tgolden"]
    for name, group, note, b, m, n, k, dtype, tx1, tx2, golden, data in CASES:
        if dtype == 2 and bfloat16 is None:
            continue
        seed = 20261101 + sum(ord(ch) for ch in name)
        rng = np.random.default_rng(seed)
        x1_shape, x2_shape = physical_shapes(b, m, n, k, tx1, tx2)
        x1 = make_input(x1_shape, dtype, rng, data)
        # neg 模式：x1 正、x2 负 → 全负输出
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

        gold = golden and golden_affordable(b, m, n, k)
        if gold:
            y = impl(x1, x2, transposeX1=tx1, transposeX2=tx2)
            y.astype(np.float32).tofile(os.path.join(out_dir, "golden_y.bin"))
        lines.append(f"{name} {b} {m} {n} {k} {dtype} {1 if tx1 else 0} "
                     f"{1 if tx2 else 0} {1 if gold else 0}")
        manifest.append(f"{name}\t{group}\t{note}\t{b}\t{m}\t{n}\t{k}\t{dtype}\t"
                        f"{1 if tx1 else 0}\t{1 if tx2 else 0}\t{1 if gold else 0}")

    tag = "".join(groups)
    cases_path = f"matrix_cases_{tag}.txt"
    with open(cases_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open("matrix_manifest.tsv", "w") as f:
        f.write("\n".join(manifest) + "\n")
    print(f"Generated {len(lines)} cases -> {cases_path}")


if __name__ == "__main__":
    main()
