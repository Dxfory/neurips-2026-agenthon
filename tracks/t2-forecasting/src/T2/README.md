# Track 2 — Forecasting

**2026-10-19 Final 候选改为 v2.7.7，已在本地冻结（验证完成）。2026-10-21 镜像已公开、Final 包已打好，待上传（见[发布记录](sub/PIPELINE.md#published-image-and-final-upload-package-2026-10-21)）。**

- **v2.7.6 在 v2.7.5 之上加了两步，v2.7.7 再改一处（均为预注册实验）：**
  - E37：FOMC 周的国债与短期国库券格子，中心按 κ × SD 平移（κ 只用 2005–18 年估计）。测试期利率任务 −0.041 [−0.073, −0.011]。
  - E40b：月度卡的面板里若已有目标月的稳定同伴序列（如 PCE 配当月 CPI），就用回归现测。测试期 −0.495 [−0.759, −0.268]。
  - E41（v2.7.7）：只改 E40b 的宽度（近期残差波动、回归预测误差、按目标类型的常数）。测试期 −0.036 [−0.066, −0.011]，90% 覆盖率 0.766 → 0.841。
- **未采用：** E38、E39；E40 守卫未过，换成修正后的 E40b。
- **2026-10-21 复核（E42、E43，v2.7.7 不变）：**
  - 国库券 draws 取整到 1 bp 只在零下限有效，离开零下限 +0.0009 [−0.0051, +0.0093]，未采用；
  - EM 两格卡重开 `pair_match`，训练期 +0.0646 [+0.0287, +0.1055]，未采用；
  - 2025–26 的 14 个 FOMC 周样本外（E43）：v2.7.7 每卡 1.094，M0 1.118。E24、E25、E29 成立；E37 反向 +0.009 [−0.012, +0.029]，因为这两年 FOMC 周利率没有下行。按事先标准不改版本，是否关掉 E37 由团队决定；
  - LLM 实验复查：E2、E6 的结论加了更正（练习卡注释是事后信息），遮蔽漏洞等问题记入[错题本](ERROR_BOOK.md)。
- **验证：** 149 项测试、fuzz 70 张、126/126 受限容器（含 22 张官方短期练习卡）、40 张 Final 形状卡全部通过；与实验 arm 和 v2.7.6 的一致性检查全部通过。
- **预估分数：**
  - Dev 约 2.46；
  - ranked Final 日频卡每张约 1.07（M0 1.16）。随资产组合不同在 1.05–1.14 之间，单周可能落在约 0.6–2.1；
  - 面板里有当月 CPI 的 PCE 卡，每张约 0.75（M0 1.50）。
- **详见：** [forecast-v2_7_7](forecast-v2_7_7/README.md)、[实验 E37–E42](exp/README.md)、[发布记录与情景分析](sub/PIPELINE.md)、[验证记录](sub/verification-v2_7_7/)。

以下 2026-10-18 的 v2.7.5 记录已被 v2.7.7 取代，保留原文。

**2026-10-18 Final 候选 v2.7.5 已在本地冻结（验证完成；镜像未公开，未打包，未上传）。**

- **官方 ranked Final 的现行格式：** as-of 为周一 2026-10-26；UST 与 ECB 定价汇率在 1–4 个工作日后；另有少量月度数据发布卡，预测 10 月 27–30 日首发的下一个数值。
- **v2.7.1–v2.7.5 依次加入：**
  - F4 宽度回到 1.1（E23）；
  - 数据发布卡与周频修复（v2.7.2）；
  - FOMC 周方差剖面，两格卡不再配对（E24、E25）；
  - 远离零下限时，UST 票据保留家族宽度（E29）；
  - 价格指数月度开关（E30，事后规则，由团队决定开启）；
  - 未季调月度目标的季节调整（E35、E36）：留出集测试期 −0.188，20 个序列中 19 个改善。
- **验证：** v2.7.5 在 104 张公开卡上与 v2.7.4 逐位一致；134 项测试、fuzz 70 张、104/104 受限容器、38 张 Final 形状卡全部通过。
- **预估分数：** Dev 约 2.46；ranked Final 每张卡约 1.05（M0 约 1.1），单周可能落在 0.65–1.65。
- **详见：** [forecast-v2_7_5](forecast-v2_7_5/README.md)、[实验 E23–E36](exp/README.md)、[发布记录](sub/PIPELINE.md)、[验证记录](sub/verification-v2_7_5/)。

2026-10-12 的 v2.7.0 当前合同验收与打包记录保持不变。

**2026-10-12 Final 打包适配与离线验证完成：** 已修正提交顺序并新增当前 common v2.6.0 / profile 打包入口。私有源码链接不能作为公开 provenance，因此新候选镜像补入完整记录后重新验收，108/108 通过、324 个产物哈希与旧镜像一致；56 项工具测试通过，离线 Final descriptor 与合成 key 的两文件 ZIP 校验通过。见[本轮结果](contracts/final-packaging-2026-10-12/RESULT.md)与[打包及发布门](contracts/final-packaging-2026-10-12/README.md)。[官方通知](https://github.com/Agenthon-2026/track2-forecasting-public/issues/4#issuecomment-6103652999)说明 ranked 卡片在提交关闭后生成，不是上传前置。镜像尚未公开、未生成真实团队包、未正式上传；截止资格与平台接受单独记录，旧源码与收据保留。

**2026-10-12 当前合同本地工程验收通过：** 同一 v2.7.0 数值源码在 common v2.6.0 / Track `a2c93e7` 下，104 张历史公开卡 + 4 个短期限/有序轴/单位边界案例 **108/108 通过**，0失败/回退/遗漏；7次独立目录重复交付字节一致，130个单元测试通过。旧配置与历史收据保留，数值源码未改。见[工程结果与证据](contracts/current-contract-2026-10-11/RESULT.md)。本轮为本地历史/合成工程验收；真实 ranked Final 的 panels/grid/generator、截止资格与平台接受仍待绑定，镜像未发布、未上传比赛。

**2026-10-11 确认可行性收口：** 本轮新增合格名册为0，当前状态 `insufficient_new_confirmation_support`；不实施/运行位置×形状诊断，不创建科学注册，不采用。固定 B_VAR 拒绝、四块已揭盲开发身份及生产v2.7.0不变；早期历史空档/其他资产和独立N仍未知。下一任务是当前合同兼容与完整 ranked Final 工程绑定，实际 panels/grid/generator、观测可用性和验收仍未建立。见[决定与重开条件](contracts/confirmation-feasibility-2026-10-11/DECISION.md)、[来源审计](contracts/confirmation-feasibility-2026-10-11/SOURCE_FEASIBILITY.md)、[覆盖矩阵](contracts/confirmation-feasibility-2026-10-11/COVERAGE_MATRIX.md)。这些审计的历史行号固定解析于 aligned base `dd272ebbda62428dd816e225829cf7ed796e45ef`，不随本次入口插入偏移。

**当前纯价格 B_VAR 四块稳健性实验已完成，拒绝固定配方的有界 F3 研究采用（`reject_fixed_B_VAR_for_scoped_research`）。** 61训练／36校准／4新结果块、8/8有效，0失败/遗漏/重试/回退，单科学进程、0网络/外部模型调用。full B_VAR/V27 均值 0.7518253187/0.7563976959，仅改善 **0.6044938%**、**1/4胜**，UST composite/variogram/80%覆盖受损，16门中5门失败。旧五窗6.374983%仅为发现背景，未池化或救门；固定配方停止，不证明所有VAR无用。操作块严格错开但统计独立未验证、有效独立N未知；current-vintage/non-PIT及既往raw词法暴露保留，日历中性评估仍继承policy-selected训练/校准。独立[结果复审](contracts/price-var-robustness-2026-10-11/reviews/result-review.md) accepted（0 Critical / 0 Important / 0 Minor）；[完整结果](contracts/price-var-robustness-2026-10-11/RESULT.md)、[summary](contracts/price-var-robustness-2026-10-11/results/run-v1/summary.json)、[归档映射](contracts/price-var-robustness-2026-10-11/archive_manifest.json)。`production_adoption=false`，生产仍v2.7.0。

**此前政策均值五起点实验已完成，固定候选拒绝继续（`reject_fixed_candidate_for_continuation`）。** 61训练／36校准／5评估、25/25有效槽位，0失败/遗漏/重试/回退，0网络/外部模型调用。C_VAR−B_VAR=+0.0162499048（+3.025383%），C_VAR−V27=−0.0203228496（−3.542468%）；纯价格B_VAR较V27均值改善6.374983%、较B_old改善1.914395%，说明直接胜v2.7不能替代匹配消融。新用途输入/标签隔离、注册及执行复审在科学运行前提交、唯一预测封存后评分均已执行；[完整结果](contracts/policy-mean-new-cohort-2026-10-11/RESULT.md)与[归档映射](contracts/policy-mean-new-cohort-2026-10-11/archive_manifest.json)绑定实际字节。五个current-vintage时间窗有已知词法暴露，独立N/独立采用确认未建立；结论仅针对这组固定policy-mean表征/模型，不否定所有政策、文本/RAG或T2研究。独立算术核验通过，独立[结果复审](contracts/policy-mean-new-cohort-2026-10-11/reviews/result-review.md) accepted（0 Critical / 0 Important / 0 Minor）；生产仍v2.7.0、`production_adoption=false`。

**旧政策传导五臂实验已完成，H1/H2均拒绝采用。** 固定61训练／36校准／46评估、230/230槽位成功，0失败/遗漏/重试，单科学进程、0网络/外部模型调用。H1 C_mean−B=−0.0468815003（−3.669109%），95%区间[−0.153120360,+0.021363344]跨0；H2 C_joint−C_mean=+0.0004009388，亦未过冻结规则。点估计改善不等于验证增益。生产仍v2.7.0，未直接比较或替换生产。完整[结果](contracts/policy-transmission-2026-10-11/RESULT.md)与注册绑定真实执行；独立[结果复审](contracts/policy-transmission-2026-10-11/reviews/result-review.md) accepted（0 Critical / 0 Important / 0 new Minor）。

**从 [T2 流水线](FORECAST_PIPELINE.md) 开始。** 这是唯一完整主文档，按八个模块组织题目、公开资料、通用架构、分类需求、候选方案、研究评估、执行与结果交付。

此前完成的研究是已封存/评分的 Final 短周期开发实验：57起点/33激活，C与探索性B均 **reject**，不采用、不重调本名册；数值生产仍v2.7。官方ranked Final为 **UST/G10 FX、2–12BD、cutoff2026-10-23、targets2026-10-27–11-10、F1–F4**；F4为前瞻风险/日程，不保证冲击，旧sealed check为unranked。四资产×三期限研究是core research projection，官方实际target panels/family generator仍未知。详见[模块1.0](FORECAST_PIPELINE.md#final-short-official)、[最近实验结果与决定](FORECAST_PIPELINE.md#final-short-results)、[冻结结果](contracts/final-short-2026-10-11/RESULT.md)。Task3及最终独立审阅已通过；Git交付以main远端引用和推送后readback收据为准。

旧RF1为历史低优先级支线，原阈值/合同不变；104卡与长周期/EM矩阵保留历史资料身份，不是ranked Final目标要求。RF1来源缺口仍见[历史6.5](FORECAST_PIPELINE.md#rf1-historical-feasibility)。

| 查阅目的 | 入口 |
| --- | --- |
| 完整方向、任务顺序与验收条件 | [T2 流水线](FORECAST_PIPELINE.md) |
| 当前四块纯价格稳健性结果与证据身份 | [结果](contracts/price-var-robustness-2026-10-11/RESULT.md)、[summary](contracts/price-var-robustness-2026-10-11/results/run-v1/summary.json)、[归档映射](contracts/price-var-robustness-2026-10-11/archive_manifest.json) |
| 此前五起点政策均值实际结果与原始证据身份 | [结果](contracts/policy-mean-new-cohort-2026-10-11/RESULT.md)、[summary](contracts/policy-mean-new-cohort-2026-10-11/results/run-v1/summary.json)、[归档映射](contracts/policy-mean-new-cohort-2026-10-11/archive_manifest.json)；[资格阶段快照](research/policy-mean-new-blocks-2026-10-11/QUALIFICATION.md) |
| 旧政策传导结果、证据、设计与名册 | [结果](contracts/policy-transmission-2026-10-11/RESULT.md)、[证据](research/policy-transmission-2026-10-11/EVIDENCE.md)、[设计](research/policy-transmission-2026-10-11/STUDY_DESIGN.md)、[登记](research/policy-transmission-2026-10-11/design_registration.json)、[名册](research/policy-transmission-2026-10-11/roster.json) |
| 当前版本、研究进度、开放问题与交付状态 | [模块 8](FORECAST_PIPELINE.md#module-8) |
| 变量、资产、期限、单位、输出与评分公式 | [任务与目标合同](TASK_CONTRACT.md) |
| 104 卡与逐目标格清单 | [合同清单](contracts/public-2026-10-10/task_contract_inventory.json)、[目标格 CSV](contracts/public-2026-10-10/target_cells.csv) |
| 最近完成的短周期实验合同与完整测量 | [协议](contracts/final-short-2026-10-11/protocol.json)、[登记](contracts/final-short-2026-10-11/registration.json)、[结果](contracts/final-short-2026-10-11/RESULT.md)、[分数](contracts/final-short-2026-10-11/scores.csv)、[诊断](contracts/final-short-2026-10-11/diagnostics.csv)、[审阅/验证收据](contracts/final-short-2026-10-11/validation_receipt.json) |
| RF1 历史原始输入有界查找与封存 | [冻结注册](contracts/rf1-original-input-probe-2026-10-11/probe_registration.json)、[实际结果](contracts/rf1-original-input-probe-2026-10-11/probe_result.json)、[当前封存/真实审阅门](contracts/rf1-original-input-probe-2026-10-11/validation_receipt.json)、[交付验证器](contracts/rf1-original-input-probe-2026-10-11/validate_delivery.py) |
| RF1 前阶段固定候选输入/谱系资格与交付 | [预I/O登记](contracts/rf1-candidate-input-lineage-2026-10-11/input_registration.json)、[输入审计](contracts/rf1-candidate-input-lineage-2026-10-11/candidate_inputs.json)、[post-acquisition派生manifest](contracts/rf1-candidate-input-lineage-2026-10-11/derivation_manifest.json)、[whole-parent资格](contracts/rf1-candidate-input-lineage-2026-10-11/candidate_qualification.json)、[当前封存](contracts/rf1-candidate-input-lineage-2026-10-11/validation_receipt.json)、[交付验证](contracts/rf1-candidate-input-lineage-2026-10-11/validate_delivery.py) |
| RF1 历史July可行性证据与封存 | [历史证据](contracts/rf1-feasibility-2026-10-11/historical_evidence.json)、[日历可行性](contracts/rf1-feasibility-2026-10-11/cohort_feasibility.json)、[新收据](contracts/rf1-feasibility-2026-10-11/validation_receipt.json)、[交付验证](contracts/rf1-feasibility-2026-10-11/validate_delivery.py) |
| RF1 cohort 历史名册、执行决定与封存 | [冻结名册](contracts/rf1-cohort-2026-10-10/roster.json)、[执行决定](contracts/rf1-cohort-2026-10-10/execution_decision.json)、[封存/审查绑定](contracts/rf1-cohort-2026-10-10/validation_receipt.json) |
| 实验方法、开发结果与历史记录 | [实验记录](exp/README.md)、[证据附件索引](FORECAST_PIPELINE.md#evidence-index) |
| 失分根因、已否定路线与重开条件 | [T2 错题本](ERROR_BOOK.md) |
| 构建、验证、镜像、打包、上传与发布历史 | [提交流水线](sub/PIPELINE.md) |
| 旧总体方案链接 | [MASTER_PLAN.md](MASTER_PLAN.md) 仅保留兼容入口 |

`forecast-v2_7/` 是当前 main 数值版本的构建上下文；历史版本及既有 F1 实验基线保留原身份。源码、模型采用、镜像公开、平台上传和官方反馈的状态分别查模块 8，README 不维护第二份状态表。

比赛原始规则：[官方总仓库](https://github.com/Agenthon-2026/Agenthon2026-public)、[Track 2 仓库](https://github.com/Agenthon-2026/track2-forecasting-public)。具体复核使用任务合同绑定的源版本，不能混用不同规则与运行版本。
