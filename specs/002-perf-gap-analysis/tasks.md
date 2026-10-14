# Tasks: 性能差距诊断与改进框架（对标榜首 5~10×）

**Input**: Design documents from `/specs/002-perf-gap-analysis/`

**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/, quickstart.md

**Tests**: 沿用项目门禁内嵌形式（快门禁/全量回归），不单列测试任务。

## Format: `[ID] [P?] [Story] Description`

- **[P]**: 可并行（不同文件、无未完成依赖）；**[Story]**: US1/US2/US3；描述含确切文件路径

## Path Conventions

单文件内核工程：`kernel.asc`（唯一提交物）；门禁资产 `scripts/` 与 `build/`；本特性文档 `specs/002-perf-gap-analysis/`。

---

## Phase 1: Setup（证据与基线准备）

**Purpose**: 采集诊断所需的三口径原始数据。

- [X] T001 汇总判题逐点数据：把 S9/S11 两次判题的 15 点用时与榜首列整理为对比表，写入 `specs/002-perf-gap-analysis/research.md` 的"差距画像"小节（含口径声明：判题=设备侧、采样时间、版本 md5）
- [X] T002 [P] 采集本地利用率证据：在 `build/` 运行 p00（8192³）与 c08（2048³）用例，按 `2*B*M*N*K/耗时` 计算 TFLOPS 并记录（对照峰值得利用率）
- [X] T003 [P] 手算结构流量：对 P8–P12 代表的形状族，计算 C 往返流量 `B*M*N*8B` 与输入流量 `B*K*(M+N)*2B`，写入 research.md 证据表

---

## Phase 2: Foundational（US1 前置：口径与清单）

- [X] T004 建立"已否证清单"与口径约束进 `specs/002-perf-gap-analysis/data-model.md`（K 预重排、L1 半区 2→4 等，含否证证据摘要与"禁止重复投入"标注）
- [X] T005 校验三口径一致性规则：任一根因需 ≥2 项证据（判题数据/本地指标/手算），把校验结果记录为 US1 的验收材料

---

## Phase 3: User Story 1 - 差距根因诊断 (Priority: P1)

**Goal**: 15 点全覆盖、结构化 ≤6 类、每类有证据与量级。

**Independent Test**: 按 quickstart §1 验证：根因 A/B 的量级与实测同阶（±2× 内）、每条根因 ≥2 项证据。

- [X] T006 [US1] 按形状族完成差距画像与归因：在 `specs/002-perf-gap-analysis/research.md` 落地 R1/R2 的最终版（更新的量级、置信度、与历史结论差异标注）
- [X] T007 [US1] 输出诊断报告：按 `contracts/diagnosis-report.md` 的四段式（画像表/根因清单/口径声明/差异标注）成文，并同步摘要到 `PLAN.md`

---

## Phase 4: User Story 2 - 改进框架与优先级 (Priority: P2)

**Goal**: ≤5 条方向、每条一轮可判定、含回退点。

**Independent Test**: 任取一条按"验证方法"执行可在一迭代内得有效/无效结论。

- [X] T008 [US2] 产出改进框架表：按 `收益×置信度÷成本风险` 排序（① 触发解耦 / ② Cube 喂数加深 / ③ C 通路改造 / ④ 微型点融合），每条含机制假设/预期收益/置信度/成本/风险/验证方法/回退点/判题可验证性，写入 `specs/002-perf-gap-analysis/research.md` R5–R6
- [X] T009 [US2] 框架自检：逐条核对"验证方法可执行、回退点存在、不与已否证清单冲突"，不合格条目移出并记录

---

## Phase 5: User Story 3 - 首项行动落地（S12：触发解耦） (Priority: P3)

**Goal**: tileSplit 触发与经济性估时解耦，P14/P15 类进入并行且小点不劣化。

**Independent Test**: ns 用例正确性 + 小点 A/B 不劣化 + 判题 P14/P15 用时下降。

- [X] T010 [US3] 宿主触发改造：`kernel.asc` 中把 `tileEligible` 从 est 门解耦（`workItems<核数 && totalTiles≥核数`），叠加待标定的代价门槛；非触发路径行为不变
- [X] T011 [US3] 门槛标定：用 `build/ns_cases.txt` 与 c00–c07 小点做触发/不触发本地 A/B，按"小点不劣化"约束取最大胆门槛并记录数据到 research.md
- [X] T012 [US3] 快门禁验证：seq15 15/15 + `build/smoke.txt` 子集一致；构建产物来源校验（md5）
- [X] T013 [US3] 交付与判题：提交到 main（含 `[S12-TRIGGER]` 标记与回退点 47dc041），输出判题三件套；判题结果与 P14/P15 用时归档 PLAN.md

---

## Phase 6: Polish & Cross-Cutting Concerns

- [X] T014 判题后框架重排序：按 S12 判题结果更新方向①②③的优先级与量级，同步 PLAN.md 与 `specs/002-perf-gap-analysis/`

---

## Dependencies & Execution Order

- **Setup**：T001→T002/T003（T002、T003 可并行）
- **Foundational**：T004→T005；依赖 T001
- **US1**：T006（依赖 Setup+Foundational）→ T007
- **US2**：T008（依赖 T006）→ T009
- **US3**：T010（依赖 US2 排序确认）→ T011 → T012 → T013
- **Polish**：T014 依赖 T013 判题结果

### Parallel Opportunities

- T002 ∥ T003（本地测量与手算互不依赖）
- T007 ∥ T008（诊断成文与框架表可并行起草，T008 的量级引用 T006 结论）

### MVP Scope

**US3（首项行动 S12）**是唯一产生判题收益的增量；US1/US2 是其决策依据（已基本成形，主要成本在成文与自检）。

---

## Implementation Strategy

1. 完成 Setup/Foundational（证据+清单），US1/US2 以现稿数据快速成文（1 轮）；
2. US3 按 S12 设计实施（触发解耦+标定），快门禁通过即提交判题；
3. 判题回来做 Polish（T014）重排框架，决定下一投入方向（②/③）。

## Phase 7: 探索性调查（2026-10-14 追加，用户方向：更多探索而非修补）

- [ ] E1 流水段消融实验：逐段关闭 A 喂数(NA)/B 喂数(NB)/Fixpipe(NC)/AIV 归约(NR)，量出各段在 k2048 的耗时占比，为结构性返工选型提供硬数据
- [ ] E2 架构候选清单评估：BlockMmad 深流水（L1_BUFFER_NUM=max+preload）/ streamK 化 / 双子核 AIV 分工 / 微型点纯向量路径，各给出收益-成本-风险初判
- [ ] E3 工具链探索：ops-simulator / msprof 在本工程的可接入性评估

> 注：T001–T014 已完成归档（判题 27.21 达成 US3；全量回归符合宪法 v1.1.0 门槛）；E1–E3 为新一轮探索任务。
