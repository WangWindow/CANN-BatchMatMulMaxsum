#!/usr/bin/env python3
# E1-NC 消融：跳过 Fixpipe（L0C->GM 环写），保留标志对；量 C 写路径占比
import sys
src = open(sys.argv[1]).read()
old = """                {
                    GlobalTensor<float> gmSlot; /* 8 槽环：吸收 AIV 归约滞后 */
                    gmSlot.SetGlobalBuffer(
                        reinterpret_cast<__gm__ float*>(cWorkspace) +
                            static_cast<int64_t>(blockIdx) * 32 * kMTask * kNTask +
                            static_cast<int64_t>(tileSeq & 31) * kMTask * kNTask,
                        kMTask * kNTask);
                    Fixpipe<float, float, CFG_ROW_MAJOR>(gmSlot, l0c, fixParams);
                }"""
assert src.count(old) == 1, src.count(old)
src = src.replace(old, "                (void)fixParams; /* [ABL-NC] 消融：跳过 Fixpipe */")
open(sys.argv[2], 'w').write(src)
print("ABL-NC written:", sys.argv[2])
