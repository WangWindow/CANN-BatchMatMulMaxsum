# Contracts: 入口与内部启动契约（不变式清单）

**Feature**: 001-naxis-split-combine | **Date**: 2026-10-14

本特性的核心契约原则：**对外接口零变化**。以下列出必须保持不变的契约与新增的内部约定。

## 1. 判题契约入口 `run_kernel`（外部，MUST 不变）

```text
run_kernel(x1, info_x1, x2, info_x2, y, info_y, availableCoreNum, stream, transposeX1, transposeX2)
```

- **签名与语义**: 与基线（S9）逐字节一致；参数数量、类型、顺序、返回值语义均不得变化
- **输入域**: 仅接收域内形状（B≤64、M/N≤8192、32≤K≤8192 且 K 为 8 的倍数、x1/x2 同 dtype∈{fp16,bf16}、y=fp32）
- **守卫行为**: 域外输入保持既有"静默返回不启动"行为（不被本特性改变）

## 2. 内部内核启动契约（MUST 不变）

```text
batch_matmul_max_sum_fp16<<<blockCount, nullptr, stream>>>(x1, x2, y, cWorkspace, batch, m, n, k,
    mTiles, nShards, partialStride, blockCount, transposeX1, transposeX2)   # 14 参数
batch_matmul_max_sum_bf16 <<< ... 同上 ... >>>
```

- 参数个数固定 14（禁止新增；触发判定与部分结果位置均由既有参数在核内推导）
- 不新增 `__global__` 内核
- `blockCount` 语义扩展（本特性）: 允许 `blockCount > batch × mTiles`（表示 tile 级切分启用）；
  现有约束 `blockCount ≤ batch × mTiles` 的形状行为不变

## 3. 设备工作区契约（内部扩展，向后兼容）

```text
cWorkspace 布局:
[ 0 , blockCount × 32 × kMTask × kNTask )          # 既有 C 环（语义不变）
[ 同上, 同上 + rBytes )                             # 新增: 部分结果区 r（仅触发形状分配）
[ rBytes 结束, +4B )                                # 新增: 就绪计数 cnt（仅触发形状使用）
```

- `rBytes = blockCount × batch × mTiles × partialStride × 4`（触发条件下 ≤ 194KB）
- 非触发形状不分配/不访问以上尾部区域（行为与基线一致）
- 计数 `cnt` 在每次触发执行前由宿主清零（不带新参数）

## 4. 同步原语契约（只允许既有判题实证要素）

| 原语 | 用途（本特性） | 依据 |
|---|---|---|
| 组通知 / 回执（既有 flag 1/2, mode2） | tile 生产-消费与背压 | S3–S9 判题实证 |
| 片内硬事件（既有 ID 使用方式） | L1/L0/搬出流水 | 基线不变 |
| `SetAtomicAdd` 写语义 | 就绪计数 +1 | 既有 y 原子路径实证（本地） |
| GM 读取 + 标量判等循环 | 轮询就绪计数 | 本特性新增，本地 soak 验证 |

**禁止**: 新的核间同步机制、跨块全局屏障、新内核、新入口参数。

## 5. 版本与可追溯契约

- 源码 MUST 带可 grep 标记 `[S11-TSPLIT]`（含关键注释）
- 交付三件套: 文件 + md5 + 核对方法（grep 标记 / md5sum）
- 回退点: 上一判题实证版本（S9, commit `47dc041`）
