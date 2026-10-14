# Phase 1 Data Model: N 轴并行切分与最大值合成

**Feature**: 001-naxis-split-combine | **Date**: 2026-10-14

本特性不引入外部数据存储；数据模型描述设备侧工作区布局、实体关系与不变量（全部为既有内存内的结构化复用）。

## Entities

### 工作项 (Work Item)
- **含义**: 现有并行单元 `(batch, mTileId)`，共 `workItems = batch × mTiles` 个
- **约束**: 当 `workItems < blockCount` 时 M 轴无并行度 ⇒ 进入 tile 级切分（本特性）

### Tile
- **含义**: 计算与部分结果产出的最小单位，三元组 `(b, mTileId, nShardId)`
- **扁平索引**: `flat = (b × mTiles + mTileId) × nShards + nShardId`（≤ 2^19 量级，32 位安全）
- **生产关系**: 每个 tile 由 AIC 独立产出（k 分块→矩阵乘→搬出），保持既有 32 槽环形缓冲

### 部分结果 (Row-Max Partial) `r`
- **布局**: `r[blockIdx][b][mTileId][m]`，`m ∈ [0, partialStride)`；浮点（与输出同精度）
- **有效范围**: 只有 `m < actualM` 的行有效；其余行不得被合成读取
- **写入者**: 块的 AIV（双子核冗余写同值，幂等安全）
- **规模上界**: `blockCount × batch × mTiles × partialStride × 4B`；触发条件下 ≤ 194KB

### 就绪计数 (Ready Counter) `cnt`
- **含义**: 标识"部分结果已全部写入"的共享进展信号；位于工作区尾部
- **写入语义**: 原子加写（`SetAtomicAdd` 语义），双子核各计一次
- **目标值**: `2 × blockCount`（双子核各 1）——指向同一块的 AIV 对各自 +1
- **生命周期**: 每次执行前必须归零（宿主在触发时不带参数地清零 4 字节）

### 输出 `y`
- **含义**: 每 batch 的最终分数，`(batch,)` float32
- **本特性的写入方式（触发形状）**: 合成后**幂等直写**（相同值重复写无害；不使用累加语义、无需预清零）

## Relationships

```text
workItems ──(当不足以铺满核数时)──> tiles（增加 nShardId 维度）
tile ──每个 tile──> 行最大值累计到 r 的一个 (b, mTileId) 切片
r ──全部就绪(cnt=2×blockCount)──> 合成：y[b] = Σ_m max_blocks r[block][b][mTileId(m)][m]
```

## Invariants（正确性硬约束）

1. **语义顺序**: 先对同一 (b, m 行) 跨块取最大值，再对 m 行求和；顺序不可交换（FR-002）
2. **幂等**: 任何部分结果或 y 的重复写不得改变结果（FR-003）
3. **无预清零依赖**: 合成路径不得读取未初始化或上一轮残留数据（有效范围仅 `m < actualM`、
   合成仅遍历有效行）
4. **就绪先于合成**: 合成必须在 `cnt == 2 × blockCount` 之后；部分结果与其计数写同队列先后有序（FR-008）
5. **空范围记账**: 无 tile 分配给某块时，该块的计数贡献必须仍然精确（AIC/AIV 记账公式一致）
6. **非触发形状零差异**: 不满足触发条件的形状，执行路径与基座逐字节一致（FR-004）

## State Transitions（每块 AIV 视角）

```text
[空闲] → 生成部分结果（逐 tile 读取/归约/累计）
       → 写 r 切片（MTE3）
       → cnt += 1（同队列原子写）
       → 轮询 cnt 至目标值
       → 合成（读全部 r 的有效行 → 逐 (b,m) 取最大 → 逐 b 求和）
       → 幂等直写 y
       → [空闲]
```

## Validation Rules（映射到门禁）

| 规则 | 验证方式 |
|---|---|
| 值一致（1e-4） | seq15 + 244 矩阵 + 新造 ns1/ns2/ns3（FP64 金标） |
| 无跨启动残留 | soak：同用例 ≥20 次独立进程 + 同进程多轮 |
| 无挂死/死等 | soak 超时统计；核数 1/4/20/32 变体 |
| 非触发形状不变 | 既有 244 矩阵与基线逐例比对（含域外守卫拒绝行为） |
