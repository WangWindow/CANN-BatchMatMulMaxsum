import subprocess, sys, os, numpy as np
CASES = [
    ("c01", "c01 2 64 64 128 1 0 0 1"),
    ("c06", "c06 4 129 257 64 2 1 0 1"),
    ("c07", "c07 64 32 64 32 1 0 0 1"),
    ("c08", "c08 1 2048 2048 2048 1 0 0 1"),
    ("c11", "c11 1 4096 4096 32 1 0 0 1"),
    ("c14", "c14 4 1024 512 512 1 1 1 1"),
]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
ENV = dict(os.environ)
ENV["LD_LIBRARY_PATH"] = "/home/developer/Ascend/cann-9.0.0/aarch64-linux/lib64:" + ENV.get("LD_LIBRARY_PATH","")
for name, line in CASES:
    ok = wrong = hang = 0
    for i in range(N):
        open("/tmp/opencode/soak_one.txt","w").write(line + "\n")
        try:
            subprocess.run(["../scripts/npu_reset"], capture_output=True, env=ENV, timeout=60)
        except Exception: pass
        try:
            subprocess.run(["./batch_matmul_max_sum_custom", "/tmp/opencode/soak_one.txt"],
                           capture_output=True, env=ENV, timeout=180)
        except subprocess.TimeoutExpired:
            hang += 1; continue
        g = f"output/{name}/golden_y.bin"
        if not os.path.exists(g): ok += 1; continue
        try:
            yv = np.fromfile(f"output/{name}/y.bin", dtype=np.float32); gv = np.fromfile(g, dtype=np.float32)
            if (np.abs(yv-gv) > (1e-4+1e-4*np.abs(gv))).any(): wrong += 1
            else: ok += 1
        except Exception: wrong += 1
    print(f"{'STABLE' if wrong==0 and hang==0 else 'FLAKY'} {name}: ok={ok} wrong={wrong} hang={hang} / {N}", flush=True)
print("=== SOAK DONE ===")
