#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""校验本地自测结果。

用法（在 build/ 目录下执行）：
    python3 ../scripts/verify_all.py [cases.txt]

对每个用例：
- 存在 output/<name>/golden_y.bin：与 output/<name>/y.bin 比对
  （fp32，rtol=1e-4，atol=1e-4，零容忍错配）。
- 无 golden（perf 用例）：仅检查输出有限。
"""

import os
import sys

import numpy as np


def check_finite(path, count):
    data = np.fromfile(path, dtype=np.float32)
    if data.size != count:
        return False, f"size {data.size} != {count}"
    return bool(np.all(np.isfinite(data))), ""


def main():
    cases_path = sys.argv[1] if len(sys.argv) > 1 else "cases.txt"
    cases = []
    with open(cases_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cases.append((parts[0], int(parts[1]), int(parts[8])))

    all_pass = True
    for name, b, golden in cases:
        y_path = os.path.join("output", name, "y.bin")
        if not os.path.exists(y_path):
            print(f"FAIL  [{name}] y.bin missing")
            all_pass = False
            continue
        if not golden:
            ok, msg = check_finite(y_path, b)
            print(("PASS" if ok else "FAIL") + f"  [{name}] finite check{(' — ' + msg) if msg else ''}")
            all_pass = all_pass and ok
            continue

        golden_path = os.path.join("output", name, "golden_y.bin")
        if not os.path.exists(golden_path):
            print(f"FAIL  [{name}] golden_y.bin missing")
            all_pass = False
            continue
        out = np.fromfile(y_path, dtype=np.float32)
        ref = np.fromfile(golden_path, dtype=np.float32)
        if out.size != ref.size:
            print(f"FAIL  [{name}] size mismatch {out.size} vs {ref.size}")
            all_pass = False
            continue
        diff = np.abs(out - ref)
        close = np.isclose(out, ref, rtol=1e-4, atol=1e-4)
        errs = int(np.sum(~close))
        if errs == 0:
            print(f"PASS  [{name}] max diff={np.max(diff):.3e}")
        else:
            print(f"FAIL  [{name}] {errs}/{ref.size} mismatched, max diff={np.max(diff):.3e}")
            all_pass = False

    print("=== ALL PASS ===" if all_pass else "=== SOME FAILED ===")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
