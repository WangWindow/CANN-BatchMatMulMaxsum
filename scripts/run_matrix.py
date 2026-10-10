#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""矩阵用例批跑器：分批运行 + 超时隔离 + 金标校验 + 报告。

用法（在 build/ 目录下执行）：
    python3 ../scripts/run_matrix.py matrix_cases_X.txt [--batch 8] [--reset] [--timeout 120]

工作方式：
- 把用例按 batch 行分组，每组写入临时 cases 文件整体运行（快）。
- 若某组超时/非零退出，自动二分该组定位到具体挂起用例。
- 校验：有 golden 的逐元素比对（|d| > atol + rtol*|g| 计错），无 golden 检查有限性。
- 输出：matrix_report.txt + matrix_summary.tsv
"""

import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np

OP = "./batch_matmul_max_sum_custom"
RESET = "/tmp/opencode/npu_reset"
ENV_LD = "/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64"


def load_cases(path):
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            cases.append({
                "line": line, "name": p[0], "b": int(p[1]), "m": int(p[2]),
                "n": int(p[3]), "k": int(p[4]), "dtype": int(p[5]),
                "tx1": int(p[6]), "tx2": int(p[7]), "golden": int(p[8]),
            })
    return cases


def run_group(cases, timeout):
    """运行一组用例，返回 (rc, stdout)。"""
    tmp = "/tmp/opencode/rt_batch.txt"
    with open(tmp, "w") as f:
        for c in cases:
            f.write(c["line"] + "\n")
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = ENV_LD + ":" + env.get("LD_LIBRARY_PATH", "")
    t0 = time.time()
    try:
        r = subprocess.run([OP, tmp], capture_output=True, text=True,
                           timeout=timeout, env=env, cwd=os.getcwd())
        return r.returncode, r.stdout + r.stderr, time.time() - t0
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"")
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return 124, out, time.time() - t0


def reset_device():
    if os.path.exists(RESET):
        env = dict(os.environ)
        env["LD_LIBRARY_PATH"] = ENV_LD + ":" + env.get("LD_LIBRARY_PATH", "")
        subprocess.run([RESET], capture_output=True, env=env, timeout=60)


def check_case(c):
    """返回 (ok, detail)。"""
    y_path = os.path.join("output", c["name"], "y.bin")
    if not os.path.exists(y_path):
        return False, "y missing"
    y = np.fromfile(y_path, dtype=np.float32)
    if y.size != c["b"]:
        return False, f"y size {y.size}!={c['b']}"
    if not np.all(np.isfinite(y)):
        return False, "non-finite"
    if not c["golden"]:
        return True, "finite-only"
    g_path = os.path.join("output", c["name"], "golden_y.bin")
    if not os.path.exists(g_path):
        return False, "golden missing"
    g = np.fromfile(g_path, dtype=np.float32)
    if g.size != y.size:
        return False, "size mismatch"
    bad = int((np.abs(y - g) > (1e-4 + 1e-4 * np.abs(g))).sum())
    if bad:
        d = float(np.abs(y - g).max())
        return False, f"bad={bad}/{y.size} maxdiff={d:.4g}"
    return True, f"maxdiff={float(np.abs(y - g).max()):.3g}"


def find_hang(cases, timeout, results):
    """二分定位组内挂起用例。"""
    if len(cases) == 1:
        rc, _, dt = run_group(cases, timeout)
        results[cases[0]["name"]] = ("HANG" if rc == 124 else f"rc={rc}", dt)
        return
    mid = len(cases) // 2
    for half in (cases[:mid], cases[mid:]):
        rc, _, dt = run_group(half, timeout)
        if rc != 0:
            find_hang(half, timeout, results)
        else:
            for c in half:
                if c["name"] not in results:
                    results[c["name"]] = ("OK", dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases_file")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--reset", action="store_true", help="每组前复位设备")
    ap.add_argument("--timeout", type=float, default=150)
    args = ap.parse_args()

    cases = load_cases(args.cases_file)
    print(f"total cases: {len(cases)}")
    results = {}
    rc_map = {}

    for i in range(0, len(cases), args.batch):
        grp = cases[i:i + args.batch]
        if args.reset:
            reset_device()
        rc, out, dt = run_group(grp, args.timeout)
        if rc == 0:
            for c in grp:
                results[c["name"]] = ("OK", dt)
        else:
            print(f"group {i}..{i+len(grp)} rc={rc} ({dt:.0f}s) — bisecting")
            find_hang(grp, args.timeout, results)

    # 校验全部输出
    lines = []
    npass = nfail = nhang = 0
    for c in cases:
        status = results.get(c["name"], ("?", 0))[0]
        if status == "OK":
            ok, detail = check_case(c)
            if not ok:
                status = "FAIL"
        elif status == "HANG":
            ok, detail = False, "hang"
        else:
            ok, detail = False, status
        if status == "OK" and ok:
            npass += 1
            verdict = "PASS"
        elif status == "HANG":
            nhang += 1
            verdict = "HANG"
        else:
            nfail += 1
            verdict = "FAIL"
        lines.append(f"{verdict}  {c['name']}  B={c['b']} M={c['m']} N={c['n']} "
                     f"K={c['k']} dt={c['dtype']} tx=({c['tx1']},{c['tx2']})  {detail}")
    print("\n".join(lines))
    print(f"\n=== PASS={npass} FAIL={nfail} HANG={nhang} / {len(cases)} ===")
    with open("matrix_report.txt", "w") as f:
        f.write("\n".join(lines) + f"\nPASS={npass} FAIL={nfail} HANG={nhang}\n")
    with open("matrix_summary.tsv", "w") as f:
        for line in lines:
            f.write(line.replace("  ", "\t") + "\n")


if __name__ == "__main__":
    main()
