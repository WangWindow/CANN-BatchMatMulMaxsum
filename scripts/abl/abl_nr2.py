#!/usr/bin/env python3
# E1-NR2 消融：跳过 AIV 环读+归约（保留 notify/ack/计数），量 AIV 侧占比
# 用法: python3 abl_nr2.py <base.asc> <out.asc>
import sys
src = open(sys.argv[1]).read()
old1 = """                GlobalTensor<float> gmTile =
                    cGlobal[static_cast<int64_t>(blockIdx) * 32 * kMTask * kNTask +
                            static_cast<int64_t>(aivTile & 31) * kMTask * kNTask];
                LocalTensor<float> cLocal = cQueue.AllocTensor<float>();
                if (actualM == kMTask) {
                    DataCopy(cLocal, gmTile, kMTask * kNTask);
                } else {
                    /* [S8-ACTREAD] 只搬 actualM 行（归约本就只用 actualM 行）：
                     * 小 M 形状省下 (kMTask-actualM)*kNTask*4 字节/格的无效搬运。 */
                    DataCopyExtParams rdParams = {static_cast<uint16_t>(actualM), kNTask * sizeof(float), 0, 0, 0};
                    DataCopyPadExtParams<float> rdPad{false, 0, 0, 0};
                    DataCopyPad<float>(cLocal, gmTile, rdParams, rdPad);
                }
                cQueue.EnQue(cLocal);
                cLocal = cQueue.DeQue<float>();"""
assert src.count(old1) == 1, src.count(old1)
src = src.replace(old1, "                /* [ABL-NR2] 消融：跳过 AIV 环读 */")
old2 = """                /* C 块行距恒为 kNTask；若本分片只覆盖 actualN 列，需把无效列掩蔽为 -inf 再归约。 */
                if (actualN < kNTask) {
                    /* 掩蔽无效列：矢量 Duplicate 需 32B 对齐起点，尾部不足 8 元素用标量补齐。 */
                    const uint32_t nAligned = RoundUp(actualN, 8);
                    const uint32_t nTailEnd = MinU32(nAligned, kNTask);
                    for (uint32_t row = 0; row < actualM; ++row) {
                        for (uint32_t c = actualN; c < nTailEnd; ++c) {
                            cLocal.SetValue(row * kNTask + c, kNegInf);
                        }
                        if (nAligned < kNTask) {
                            Duplicate(cLocal[row * kNTask + nAligned], kNegInf, kNTask - nAligned);
                        }
                    }
                    PipeBarrier<PIPE_ALL>();
                }
                uint32_t reduceShape[2] = {actualM, kNTask};
                ReduceMax<float, Pattern::Reduce::AR, true>(blockMax, cLocal, reduceTmp, reduceShape, true);
                if (actualM < partialStride) {
                    for (uint32_t row = actualM; row < partialStride; ++row) {
                        blockMax.SetValue(row, kNegInf);
                    }
                }
                Max(rowMaxSlice, rowMaxSlice, blockMax, partialStride);
                PipeBarrier<PIPE_V>();
                cQueue.FreeTensor<float>(cLocal);"""
assert src.count(old2) == 1, src.count(old2)
src = src.replace(old2, "                /* [ABL-NR2] 消融：跳过掩蔽/归约/部分和 */")
open(sys.argv[2], 'w').write(src)
print("ABL-NR2 written:", sys.argv[2])
