---

description: "Task list template for feature implementation"
---

# Tasks: N 轴并行切分与最大值合成

**Input**: Design documents from `/specs/001-naxis-split-combine/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: 本项目未要求独立测试任务——"测试"以验证门禁形式内嵌于各阶段（快门禁/全量回归），任务描述中给出确切命令与期望。

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- 单文件内核工程：唯一提交物 `kernel.asc`（仓库根）；本地框架 `main.asc`；门禁资产 `scripts/`；
  本地构建与用例数据 `build/`；本特性文档 `specs/001-naxis-split-combine/`。

---

## Phase 1: Setup（基线与资产准备）

**Purpose**: 冻结基线、生成目标类用例与门禁资产，供后续 A/B 与回归使用。

- [ ] T001 记录基线：确认 `kernel.asc` md5 为 S9 的 `1b1afade...`、git 干净、回退点 commit `47dc041`；把基线与回退命令写入 `PLAN.md` 底部（新增"## S11 基线"小节）
- [ ] T002 [P] 生成目标类用例与金标：用 `scripts/BatchMatmulMaxSum.py` 的 `impl`（FP64）生成
  `ns1=(1,64,8192,512)`、`ns2=(1,8,2048,2048)`、`ns3=(2,128,512,1024)` 的输入与金标到
  `build/input/ns[123]/x[12].bin`、`build/output/ns[123]/golden_y.bin`，并写出 `build/ns_cases.txt`
- [ ] T003 [P] 重建快门禁清单：`awk 'NR%4==1' scripts/matrix_all.txt > scripts/smoke.txt`（预期 61 行；覆盖 K 类/尾块/转置/B 段/切分路径）
- [ ] T004 [P] 采集基线耗时：在 `build/` 用当前 S9 二进制运行 `build/ns_cases.txt`，记录每例中位耗时
  （作为 ≥2× 目标的对照基准，写入 `specs/001-naxis-split-combine/research.md` 末尾的"基线数据"小节）

---

## Phase 2: Foundational（US1 的阻断性前置）

**Purpose**: 宿主与内核两侧的共享使能（触发判定、工作区扩展、指针推导）。完成后 US1 才可开工。

- [ ] T005 宿主使能（[S11-TSPLIT]）：在 `kernel.asc` 的 `run_kernel` 中把切分时的
  `blockCount` 公式改为 `min(coreCount, workItemsAll * nShards)`；派生 `tileSplit = blockCount > workItemsAll`；
  `cWorkspaceBytes` 追加 `tileSplit ? blockCount*batch*mTiles*partialStride*4 + 4 : 0`；
  `tileSplit` 时对计数器 4 字节执行 `aclrtMemsetAsync` 清零（不带新参数）
- [ ] T006 内核侧共享指针与判定：在 `RunCube`/`RunVector` 内以既有参数推导
  `tileSplit`、`rGlobal`（= cWorkspace + blockCount*32*kMTask*kNTask）、`cntGlobal`（= rGlobal + rBytes）
  与 tile 级划分公式（`flat = (b*mTiles+mTileId)*nShards+nShardId`，两侧逐字一致）

**Checkpoint**: 仅新增"非触发形状零变化"的判定与指针；先用快门禁确认 244 子集/seq15 不回归再继续。

---

## Phase 3: User Story 1 - 目标形状获得 N 轴并行 (Priority: P1) 🎯 MVP

**Goal**: workItems<核数 的形状在多核上并行执行；输出 1e-4 一致且耗时不劣化至少 2×。

**Independent Test**: 运行 `build/ns_cases.txt` 三条用例：chk 全过 + 对比 T004 基线 ≥2× 下降。

- [ ] T007 [US1] AIC tileSplit 主循环：在 `RunCube` 中新增 tileSplit 分支——按 flat 区间解码
  `(b, mTileId, nShardId)`，复用既有单 tile 流水（k 分块→mmad→fixpipe）与既有 barrier/tileSeq 节奏
  （背压/分组通知/残批补发逻辑不变）；非触发形状走原路径
- [ ] T008 [US1] AIV tileSplit 循环与部分结果：在 `RunVector` 中新增 tileSplit 分支——逐 (b,mTile)
  在该块 N 切片上累计逐行最大值到 `rowMaxSlice`（有效范围仅 `m < actualM`），块内该 (b,mTile)
  结束时把切片写入 `r[blockIdx][b][mTileId][m]`（幂等冗余写，双子核同值）
- [ ] T009 [US1] 就绪计数与两阶段合成：每子核写完备部分结果后 `SetAtomicAdd` 对 `cnt` +1；
  轮询 `cnt`（GM 读取 + 标量判等，配 MTE2→S 事件）至 `2*blockCount`；随后各块冗余合成
  `y[b] = Σ_m max_blocks r[block][b][mTileId][m]` 并**幂等直写** y（无原子、无需清零）
- [ ] T010 [US1] 构建与目标形状验证：在沙箱（同源构建）编译后运行 `build/ns_cases.txt`，
  `scripts/chk.py` 全过且三例耗时对比 T004 基线 ≥2× 下降（记录于 research.md）
- [ ] T011 [US1] 稳定性验证：ns 用例 ×20 次独立进程重复 + `BMMS_CORE_OVERRIDE` ∈ {1,4,20,32}
  变体执行，全部逐值一致、无挂死（期望耗时随核数缩放）

**Checkpoint**: US1 独立可交付——目标形状获得并行且正确；其余形状尚走原路径（由 T006 保证）。

---

## Phase 4: User Story 2 - 既有全部形状零回归 (Priority: P2)

**Goal**: 全部既有形状逐值一致、零挂死、核数无关。

**Independent Test**: 与基线逐例比对 244 矩阵 + soak + 核数变体。

- [ ] T012 [US2] 全量矩阵回归：`build/` 下 `python3 scripts/run_matrix.py scripts/matrix_all.txt --batch 24 --timeout 240`；
  期望 PASS=162 / FAIL=82（全部为域外 K 守卫）/ HANG=0，且失败清单与基线逐例一致
- [ ] T013 [US2] soak 与核数变体：`python3 scripts/soak.py 20`（6 用例×20 全稳）；
  `BMMS_CORE_OVERRIDE` ∈ {1,4,20,32} 各跑 `scripts/seq15.txt` 期望 15/15

---

## Phase 5: User Story 3 - 可解释、可回退的交付 (Priority: P3)

**Goal**: 受控交付：标记 + md5 + 提交 + 判题 + 回退预案。

**Independent Test**: 核对三件套与回退命令有效。

- [ ] T014 [US3] 交付标记与归档：确认 `kernel.asc` 含 `[S11-TSPLIT]` 标记（grep 计数注明）；
  `build/ns_cases.txt`/`build/smoke.txt` 归档进 git；重启核对 md5 稳定
- [ ] T015 [US3] 快门禁终检与提交：seq15 15/15 + smoke 子集通过后提交
  （消息注明：机制、实测 A/B、回归结论、回退点）
- [ ] T016 [US3] 判题交付：输出三件套（文件+md5+核对方法）交用户提交；准备结果记录模板
- [ ] T017 [US3] 判题结果归档与回退预案：结果写入 `PLAN.md`（逐点用时/机制结论）；
  若未达预期按 `quickstart.md` §5 回退到 `47dc041`

---

## Phase 6: Polish & Cross-Cutting Concerns

- [ ] T018 阈值标定复核：依据 T010/T016 实测复核触发条件与并行度下限（research.md R5），必要时微调并重跑快门禁
- [ ] T019 文档收尾：更新 `PLAN.md` 与 `specs/001-naxis-split-combine/` 产物状态（completed 标记）

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: 无依赖，立即开始（T002–T004 可并行）
- **Foundational (Phase 2)**: 依赖 T001（基线确认）；T005→T006 顺序执行
- **US1 (Phase 3)**: 依赖 Phase 2；T007/T008 分属 AIC/AIV 两段可并行编辑，T009 依赖 T008；T010→T011 顺序
- **US2 (Phase 4)**: 依赖 US1 完成（T012 可与 T011 并行启动在已构建产物上）
- **US3 (Phase 5)**: 依赖 US1+US2 通过
- **Polish (Phase 6)**: 依赖 US3（判题结果）

### Parallel Opportunities

- T002 / T003 / T004（资产生成、清单重建、基线采集，互不依赖）
- T007 / T008（同一文件不同函数段，可分散提交但需合并构建）
- T012 与 T011（矩阵回归与稳定性测试可并行在各自构建产物上）

### MVP Scope

**仅 US1**（+ 其前置 Phase 1/2）：目标形状并行且正确即为最小可交付价值；
US2 保证把既得分数守住，US3 保证交付纪律——三者叠加才完整。

---

## Implementation Strategy

1. 先做 Phase 1/2（基线+使能），用快门禁确认非触发路径零变化；
2. 实现 US1（T007→T008→T009 一组改动）→ 构建 → T010/T011 验证；
3. US2 全量回归后台执行，同时准备 US3 交付；
4. 判题回来后按 Polish 收尾；若目标未达或出现回归，立即 T017 回退。
