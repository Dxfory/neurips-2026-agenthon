# T2 任务与目标合同

**2026-10-10：主文档模块 1 的公开任务定义已完成。** 本文定义预测什么、用什么单位、预测到什么时候、交付什么以及如何评分。覆盖当前 103 张 practice 卡和 1 张 exemplar；逐卡源文件与提取结果均已固定。它不表示预测器已通过新版运行验收，也不表示已获得预测增益或完成 Final 提交。

所属主文档：[T2 流水线模块 1](FORECAST_PIPELINE.md#module-1)；本文是精确合同附件。证据：[逐卡 JSON](contracts/public-2026-10-10/task_contract_inventory.json)、[逐目标格 CSV](contracts/public-2026-10-10/target_cells.csv)、[复核收据](contracts/public-2026-10-10/validation.json)。

## 1. 任务和优化目标

**T2 Forecasting — Reasoning-augmented time series。** 使用时间序列面板与带时间戳文本语料，在合法信息截止点生成未来目标面板的联合预测分布。任务动词为 `forecast`；评分指标为 **CRPS composite，越低越好**。解释文字不参与预测质量评分。文本与推理是否带来 information uplift 是独立研究问题，不是额外分数，也不是每张卡强制使用文本的要求。[官方任务与研究说明][track-readme]

对单卡 c，令截止前允许的信息集为 I_c，目标资产有序集合为 A_c，期限键有序集合为 H_c。需要交付的是 `P(Y_c | I_c)` 的等权联合 draws，其中 `Y_c = (Y[a,h])`。每个 draw 同时覆盖全部资产与期限；预测对象是未来观测变量的现实分布。事件、政策路径、RAG、定价与状态模型均是形成分布的可选方法，不能代替最终目标。

“Gate: as-of cutoff + calibration”是任务总体要求。当前机器准入实现为 g0–g3，没有另一个数值阈值可以使单张预测自动获得“统计校准通过”。校准需在独立、合格的多实例结果上检验；information uplift 需通过固定信息集与模型的对照研究检验。

## 2. 来源、身份与覆盖范围

| 项目 | 本次固定身份 |
| --- | --- |
| 官方 Track2 源 | `30c8019d997f930eab9ba569d089d12298d86d8c`，本次从官方 main 获取 |
| 官方 common | `v2.6.0`，peeled commit `bd01548e34d21fd660d88fd06157078fb25ece4e` |
| Track2 canonical verifier | 源码声明版本 `3.1.0`；不是同名兼容 shim |
| 本地 main 整合 | 已整合当次远端 `416a1f29cf08449cd6156606dccf2b13bdfbe2ba`；本地合并 HEAD `a25798c8db0ca45fea006d8babacb31a635d259d` |
| 来源和哈希 | [合同源 manifest](contracts/public-2026-10-10/source_manifest.json)、[规则源身份](contracts/public-2026-10-10/contract_sources.json) |

具体排名运行以签署计划 C1 及其绑定的 roster、ordered grid、scorer、reference 和 scales 为准；本次未获取 sealed Final 数据。公开语义需合读实际 `card.toml`、`forecast_spec.json`、每卡说明及专项规则。实际可执行 gate/score 以对应版本源码为依据。不能用泛化导读覆盖逐资产单位，也不能遇到正式源间矛盾就自行补出唯一解释。

| 名册 | F1 | F2 | F3 | F4 | 总卡数 | 每个 draw 的目标格总数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| practice | 23 | 27 | 22 | 31 | 103 | 263 |
| practice + exemplar | 23 | 27 | 23 | 31 | 104 | 267 |

practice 有 87 张 level、16 张 log_return，99 张日频、4 张月频；5 张 transfer 属于日频，不能再加到 99 之外。工程上可划分为 94 张其他日频、5 张 transfer、4 张月频。practice split 为 32 public-dev + 71 validation。全部 104 卡有 26 个不同目标资产 ID、60 张单格卡；exemplar 是 4 格多资产卡。

103 张 practice 均有 spec；exemplar 没有 `forecast_spec.json`，其有序网格取自 card。当前统计是公开覆盖事实，不是 Final 固定范围或抽样比例。[完整统计](contracts/public-2026-10-10/inventory_summary.json)

## 3. 单卡 ForecastContract 必备定义

下表是内部合同的语义字段，不是要求给官方 Parquet 增加列。逐卡清单已经填入公开可确定值；未公开的字段保持缺失及来源边界。

| 字段组 | 定义与依据 | 缺失或冲突时 |
| --- | --- | --- |
| identity | task_id、family、split、源 commit/hash、exemplar 标记 | 不靠目录名称猜任务内容；身份不一致停止绑定 |
| information cutoff | asof、文本 cutoff、输入文件和许可来源；正式运行还需规则/资源身份 | 时间资格不明的额外信息不进入预测 |
| target axes | 原样保留 asset_ids 和 horizons 的顺序、目标格数量 | 不静默增删、排序或改名；ranked 绑定 signed grid |
| target variable | level 或累计 log_return、经济变量、源序列、原生单位和 FX 报价方向 | 不能用相似序列或通用单位替代 |
| time mapping | 目标频率、horizon key、显式目标日期/观测月、实际末观测月、采样步数 | 月频缺有效月份映射不可机械换算；日期矛盾显式保留 |
| vintage | 目标首发/修订口径、指定 vintage；输入 vintage 与目标 vintage 分列 | 不用今日值猜首发；F1 未固定的目标快照保持 unknown |
| input roles | 目标序列与 context 分开；允许 panel/text、transfer 缺口、发布滞后 | context 出现在输入中不代表应出现在输出中 |
| output | 三文件、samples、draw 范围、有序完整网格、元信息绑定 | 非法输出是失败，不能靠 rationale 弥补 |
| scoring | scorer/common 身份、分项/权重、M0 scale 来源、clip、失败/分母 | 自建研究 scale 不冒充官方 committed scale |

单卡还需绑定实际 runner 的时间、内存、网络/API 与随机流预算。这些由运行计划/runner 核实，本清单不按 family 臆造统一预算；`g2_cutoff_resource` 的名称也不意味着其函数实际计量资源。

## 4. 变量、资产和原生单位

| 目标集合 | 预测变量 | 输出单位与变换 |
| --- | --- | --- |
| UST_2Y、UST_5Y、UST_7Y、UST_10Y、UST_30Y | 美债 constant-maturity **par yield level** | 年化百分数；4.5 表示 4.5%，25 bp 的变化为 0.25 个百分点，最终交付未来水平 |
| EUR、GBP、AUD、NZD | 对应即期 FX 水平 | 每 1 单位外币值多少 USD，即 USD/外币 |
| JPY、CHF、CAD、SEK、NOK、DKK、CNY、INR、BRL | 对应即期 FX 水平 | 每 1 USD 值多少对应外币，即外币/USD |
| MKT、SMB、HML、MOM、QMJ | 从截止后到目标期限的累计因子 log return | 输入 r 是 decimal daily simple return；输出 `sum(log1p(r))`，从 0 累计，不乘 100、不年化 |
| CPI_ALL | CPIAUCSL 指数水平 | 1982–84 = 100 的指数，非同比通胀率 |
| UNRATE | U-3 失业率水平 | percent，非小数比率 |
| NFP | PAYEMS 非农就业人数水平 | thousand jobs，非当月新增就业人数 |

UST_20Y、BAB、宏观面板中的其他序列可作 context，但不是本批目标。MKT 对应 French **Mkt-RF**，不是股票指数总收益。任务 slug 含 `cnh` 的卡仍以明确目标 ID **CNY** 为准。混合资产卡内部可标准化，但写出前必须逐资产还原原生单位、报价方向和 level/累计收益定义。[面板来源][provenance]；每个资产的直接单位出处见逐卡 JSON 的 `unit_source`。

**输入与目标的 vintage 不可混用。** F1 两张月频卡目标注明 current vintage，公开源没有固定精确目标 vintage 快照；F4 两张注明首发值。对外构造研究标签时，前者仍须取得明确目标定义，不能自行统一成首发值。

## 5. 期限：提交键、观测期与采样步数分开

| Family | 当前公开 horizon keys |
| --- | --- |
| F1 | 126、127、128、129、140、145、160、165、189 |
| F2 | 63、64、65、126 |
| F3 | 21、63、126 |
| F4 | 7、21、63、126 |

日频保留 authored business-day lead key，不转为自然日，也不按每个资产的非缺失行数重编期限。作者指南的 BDay 是工作日偏移指导，未给出适用于所有市场的唯一节假日/缺测标签解析规则。

月频依实际目标序列的最后观测月和 spec 的 `observation_periods` 计算月份差；必须覆盖发布滞后造成的尚未观测月份。asof、观测月、发布日期、vintage、horizon key 是不同字段。[月频专项合同][monthly]

| 月频卡 | asof | 输入末观测月 | 原始 h → 目标观测月 | participant 月步 | 目标口径 |
| --- | --- | --- | --- | --- | --- |
| t2-F1-cpi-glidepath-2023 | 2023-07-12 | 2023-05 | 140 → 2024-01；160 → 2024-02 | 8；9 | current vintage；精确快照未公开 |
| t2-F1-sahm-watch-2024 | 2024-08-02 | 2024-06 | 145 → 2025-01；165 → 2025-02 | 7；8 | current vintage；精确快照未公开 |
| t2-F4-covid-nfp-2020 | 2020-03-31 | 2020-02 | 21 → 2020-04 | 2 | 首发就业水平，2020 年 5 月发布 |
| t2-F4-cpi-vintage-2022 | 2022-05-31 | 2022-04 | 21 → 2022-06 | 2 | 首发；指定 2022-07-13 vintage |

本次只读这 4 张输入面板的 asset/date 列核实末观测月，未解码 value 或未来标签；[日期证据](contracts/public-2026-10-10/monthly_anchor_evidence.json)绑定原文件 SHA256。**UNRATE 的 participant 月步是 7/8；官方 M0 文档独立指定该卡 normalizer 用 8/9。** 后者采用其另列的 M0 转换分支，两者不能共享一个默认步数或据此修改官方 normalizer。[M0 月步规范][m0]

exemplar 同时给出 `asof=2024-06-28`、`h=21`、显式 `target_date=2024-07-31`；标准 BDay(21) 为 7/29，配套说明又涉及美国假日并称日期 approximate。清单保留原键与显式日期，不宣称已解决唯一日历映射。最终目标解析要由正式合同/reference 绑定，不能自行改键或以推算日期替换作者日期。

5 张 transfer 卡保留早期目标历史与 asof 单点锚，中间有长期缺口。不能理解为“完全没有目标数据”，也不能跨缺口计算一日变化；早期样本数量不等于当前连续可估计窗口长度。

## 6. 输入和输出合同

运行接口为：

```text
forecast --panels /input/panels/ --text /input/text/ --asof YYYY-MM-DD --out /output/forecast.parquet
```

整个 staged unit 挂载在 `/input:ro`；card 位于 `/input/card.toml`，有 spec 时在 card 旁。目标网格来自实际任务合同；不能从面板中出现的所有资产推断输出目标。reference CLI 的月频路径额外读 spec，日频路径没有自动消费其中全部语义。[提交调用合同][submission]

### 6.1 三件套与样本网格

`/output` 根目录必须**恰好**有这三个普通文件：

1. `forecast.parquet`
2. `forecast_meta.json`
3. `forecast_rationale.md`（非空）

日志、诊断、额外目录、`.DS_Store`、第四文件均不得放入该目录。平台通用的文件数量上限不放宽 Track2 的三文件要求。[实际 g0–g3][scorer]

Parquet 只包含四列：`draw:int32, asset:string, horizon:int32, value:float64`。输出写入器遵守发布类型；当前 checker 还做整数/有限性检查，但不是严格的 Arrow 位宽检查。

对于 M draws、A 个资产、H 个期限，恰有 `M×A×H` 行，每个 `(draw,asset,horizon)` 唯一且完整。draw IDs 为 `0..M−1`。meta 的资产/期限顺序与合同一致；物理行顺序可以不同，canonical 展平顺序为 asset-major、horizon-minor。每个 draw 都表示一个完整联合未来；独立抽各格可能通过 schema，却不代表联合结构正确。

当前 active scorer 只接受 **samples**。硬范围是 **200–20,000 draws**；F3 ≥500、F4 ≥1000 是建议，不是额外硬门。当前 practice 卡均声明 min 200；exemplar 说明也为 200。若未来卡提高自身下限，应显式遵守并核对新版规则；当前 g1 实装只读全局 ParseLimits，不能声称它已经执行任意未来的卡级下限。

当前边界：meta ≤256 KiB、rationale ≤1 MiB、Parquet ≤64 MiB、Parquet 解压后 ≤2 GiB、≤64 row groups、≤500 万行；外层输出总量 ≤64 MiB。禁止软/硬链接及特殊文件。这些约束需同时满足。[解析限制][limits]

### 6.2 metadata

samples 的必要字段如下：

| 字段 | 要求 |
| --- | --- |
| unit_id | 与实际 task.id 一致 |
| asof | YYYY-MM-DD，与 trusted card 一致 |
| representation | `samples` |
| asset_ids | 合同给定的有序资产列表 |
| horizons | 合同给定的有序正整数键列表 |
| n_draws | 实际完整 draws 数，满足范围 |

可选 `target`（不是 `target_type`）若提供，必须匹配 card.target_type；可选 `units` 和 `rationale` 对象不取消原生单位义务或必需 rationale 文件。`seed` 不是该 sidecar 必需字段，模型披露的 `models[]` 属于 submission descriptor。随机流及拟合模型身份另按 runner/提交政策封存。[固定 common schema][schema]

common schema 仍列 `parametric`，但 active Track2 明确拒绝。schema 的 target enum 包含 `yield`；本批实际只有 level/log_return，不把当前样本范围误写为所有未来卡的全局枚举限制。

## 7. 准入关卡与时间资格

| 关卡 | 当前代码实际检查 | 不能由此得出的结论 |
| --- | --- | --- |
| g0_integrity | 恰好三文件、普通文件、大小、metadata 可解析 | 预测正确、来源真实 |
| g1_schema | 公共 metadata schema、samples、draw 范围、非空 rationale | 解释可信、统计校准通过 |
| g2_cutoff_resource | unit_id/asof/可选 target 绑定 trusted card；其目标日期晚于 asof | 此函数重扫了全部输入、独立认证出版时间、计量了资源 |
| g3_domain_semantics | ordered grid、四列、完整唯一 draws、有限值 | 单位算对了、相关结构正确、分布非退化或尾部覆盖达标 |

输入 panel/text 的 cutoff 扫描发生在 staging；通常读取面板日期与语料索引时间及覆盖情况，不能独立证明所有内容的首次可得时间。资源另由 runner/集成层约束。参赛者仍须遵守信息边界，不因 checker 没有检测某种泄漏而使用未来信息。[cutoff 实现][cutoff]

数据资格待下一阶段系统验收：因子卡承认输入使用有发布滞后的最终库值，部分文本再版表格含后续修订。提供输入符合公开 staging 合同与自建历史研究满足严格 point-in-time 是不同问题；本轮记录风险，不改写官方面板或声称全部输入已完成历史资格证明。

## 8. 评分数学合同

令 m 为 draws 数、d 为资产×期限目标格数；`x[r,i]` 为联合样本，`y[i]` 为已成熟官方标签。以下是当前全部公开卡的有效默认组合；具体 ranked 运行还须绑定签署计划。[Track2 canonical scorer][scorer]、[common 数学实现][common-score]

### 8.1 三项损失

**边际 M：fair ensemble CRPS 的目标格均值。**

```text
CRPS_i = mean_r |x[r,i] − y[i]|
         − sum_{r≠s}|x[r,i] − x[s,i]| / (2m(m−1))
M = mean_i CRPS_i
```

**联合 V：p=0.5 variogram。**

```text
V = sum_{i,j} ( |y[i]−y[j]|^0.5 − mean_r |x[r,i]−x[r,j]|^0.5 )^2
```

对全部有序 pair 求和，不是 pair 均值；对角项为零，跨资产与跨期限都包括。每个无序 pair 在此被计算两次。

**尾部 T：四个分位的 pinball loss。**

```text
taus = {0.01, 0.05, 0.95, 0.99}
q[tau,i] = linear sample quantile of x[:,i]
T = mean_{tau,i} (tau − 1{y[i] < q[tau,i]}) × (y[i] − q[tau,i])
```

当前 T2 的 tail.py 实现 pinball，T2 scorer 组装总分。不能直接以 common 的旧 `crps_composite` helper 替代：后者默认尾部仍是旧 coverage penalty。[Track2 tail][tail]

### 8.2 M0 标准化与权重

`a_M, a_V, a_T` 是 M0 在本卡截止前输入确定的理论分布下，各损失的**自期望误差**。不是用实际未来 y 算出的 M0 realized loss；参赛者也不能自行提供 ranked scale。[M0 规范][m0]、[scale 校验][normalization]

```text
d > 1: S_raw = 0.5 M/a_M + 0.3 V/a_V + 0.2 T/a_T
d = 1: S_raw = (5/7) M/a_M + (2/7) T/a_T
```

single cell 是单资产且单期限；单资产多期限仍属于多格联合任务。Family 不改变上述权重，F4 也不是只按尾部评分。

M0 为 text-blind joint Gaussian random walk，其公开方法与 reference CLI、占位 baselines 不等同。公开 M0 draw 规范使用 500 draws，normalizer 则由理论分布计算，不拿这 500 draws 对假标签打分。若 M0 理论协方差为 C、`sd_i=sqrt(C_ii)`：

```text
a_M = mean_i sd_i / sqrt(pi)
a_T = mean_tau phi(Phi_inverse(tau)) × mean_i sd_i
a_V = sum_{i≠j} Var(|Y_i−Y_j|^0.5),  Y ~ M0
```

归一化分数为 1 不保证与 M0 实际得分持平；**M0 在实际某张卡或完整名册的分数不保证等于 1**。三个 scale 是按卡计算的自期望误差，通常不为 1。M0 的生产 generator 和实际 sealed scales 未在本轮取得；研究复建需标明身份，不能宣称是官方 scale 字节。

### 8.3 裁剪、失败与名册聚合

```text
有效预测: v_c = min(8, max(0, S_raw,c))
participant failure: v_c = 8
总分: S_total = mean_{c in committed roster} v_c
```

先对每卡完整 composite 裁剪，再对所有 committed cards 等权求均值；不先裁剪三分项、不先求均值再裁剪、不把 F1–F4 的均值等权。失败卡保留在分母中。[Track2 聚合][aggregate]

无输出、非法预测或超时等属于 participant failure；reference 缺失、scale/commitment 错误、重复/缺失结果行、非有限 metric 或 plan 不一致等属于 **OrganizerFault，终止整批评价**，不能算成 8 或删卡后发布部分成绩。

公开卡没有官方 labels 时只能 gates-only/no_score，不等于 participant failure=8。当前官方入口每卡一个 result，没有强制多 seed 平均合同；多 seed 属于另行登记的研究重复设计，不能挑最好 seed。裁剪影响评分的严格 propriety，不能无条件声称 clipped composite 仍严格 proper；研究保留 raw/clip 两套诊断，但正式目标按该次签署规则计算。

## 9. 已记录的源分歧和未公开项

| 情况 | 本合同处理 |
| --- | --- |
| 泛化 FX 指南称全部 USD/外币，部分卡引用的单位表缺失 | 用同一固定源中逐资产显式报价，并保存直接出处；不统一倒数 |
| 普通导读概括日频/四种 panel、transfer 无目标行 | 实际 4 月频与 5 transfer 单列，以逐卡目标及输入声明为准 |
| exemplar h21/日期不构成唯一日历映射 | 原键与显式日期都保留，正式解析待绑定 |
| UNRATE participant 7/8 与 M0 8/9 | 分开两套有来源的步数，不修 normalizer |
| 旧 glossary/exemplar 声称 std>0、calibration flags 或 tail interval score | 以当前实际 g0–g3 与 pinball 实装为准，不新增虚构门 |
| common schema 比 Track2 接受范围宽 | 当前提交 samples；枚举与实际卡范围分开 |
| F1 current-vintage 目标快照未固定 | 不伪造真实研究标签；后续需明确 reference vintage |
| Final signed roster、scales、labels、部署 scorer 绑定未取得 | 本次完成公开合同定义，不宣称线上排名/当前部署验收 |

这些源语义边界与“2,433 项结构字段对照零冲突”不矛盾：后者只针对清单列出的可比结构字段，不是全部文字叙述无歧义。

## 10. 验收、复现和阶段出口

本阶段交付：104 个有源身份的单卡合同、267 个目标格、单位与期限映射、当前输出/评分/失败合同、公开源缺失项。检查包括独立重读全部 card/spec、完整格去重核对、逐文件 hash、固定源重复提取，以及单字段源突变拒绝；结果详见 [validation.json](contracts/public-2026-10-10/validation.json)。

从仓库根目录，准备官方固定 commit 的只读快照后，用 Python 3.11+ 标准库复现；`T2_CONTRACT_SOURCE` 指向该快照根目录：

```bash
python3 T2/contracts/extract_contract_inventory.py \
  --source "$T2_CONTRACT_SOURCE" \
  --monthly-evidence T2/contracts/public-2026-10-10/monthly_anchor_evidence.json \
  --expected-source-manifest T2/contracts/public-2026-10-10/source_manifest.json \
  --out /tmp/t2-contract-reproduced
```

提取器固定已审查 manifest 的 SHA256，核对完整卡路径集与源文件字节，再生成清单；不能将不同版本的 104 张卡贴上旧 commit 身份。`source_manifest_sha256` 在逐卡 JSON 中表示 canonical compact JSON `files` 映射的 hash；manifest 文件本身的字节 hash 另列在规则源/验收收据，不混为同一种 hash。

本次合同验收未运行预测、模型 API、真实标签评分或输入数值质量验收；当时的生产锚点 v2.6.1 源码保持原 hash。随后 main 的 v2.7.0 发布更新见 [主文档当前状态](FORECAST_PIPELINE.md#current-status)，不改写本合同验收收据的历史身份。后续独立阶段是数值面板与时间戳文本的来源/可得时间/vintage/缺口资格，只有该阶段通过后才能声称相应数据可用于训练或独立评价。本轮到此完成，不自动启动数据或建模阶段。

[track-readme]: https://github.com/Agenthon-2026/track2-forecasting-public/blob/30c8019d997f930eab9ba569d089d12298d86d8c/README.md
[submission]: https://github.com/Agenthon-2026/track2-forecasting-public/blob/30c8019d997f930eab9ba569d089d12298d86d8c/SUBMISSION_CLI.md
[provenance]: https://github.com/Agenthon-2026/track2-forecasting-public/blob/30c8019d997f930eab9ba569d089d12298d86d8c/data/PROVENANCE.md
[monthly]: https://github.com/Agenthon-2026/track2-forecasting-public/blob/30c8019d997f930eab9ba569d089d12298d86d8c/docs/MONTHLY-HORIZONS.md
[m0]: https://github.com/Agenthon-2026/track2-forecasting-public/blob/30c8019d997f930eab9ba569d089d12298d86d8c/docs/M0-BASELINE.md
[scorer]: https://github.com/Agenthon-2026/track2-forecasting-public/blob/30c8019d997f930eab9ba569d089d12298d86d8c/qfbench2_track_forecasting/scoring.py
[limits]: https://github.com/Agenthon-2026/track2-forecasting-public/blob/30c8019d997f930eab9ba569d089d12298d86d8c/qfbench2_track_forecasting/limits.py
[cutoff]: https://github.com/Agenthon-2026/track2-forecasting-public/blob/30c8019d997f930eab9ba569d089d12298d86d8c/qfbench2_track_forecasting/cutoff.py
[tail]: https://github.com/Agenthon-2026/track2-forecasting-public/blob/30c8019d997f930eab9ba569d089d12298d86d8c/qfbench2_track_forecasting/tail.py
[normalization]: https://github.com/Agenthon-2026/track2-forecasting-public/blob/30c8019d997f930eab9ba569d089d12298d86d8c/qfbench2_track_forecasting/normalization.py
[aggregate]: https://github.com/Agenthon-2026/track2-forecasting-public/blob/30c8019d997f930eab9ba569d089d12298d86d8c/qfbench2_track_forecasting/aggregate.py
[schema]: https://github.com/Agenthon-2026/Agenthon2026-public/blob/bd01548e34d21fd660d88fd06157078fb25ece4e/common/qfbench2_common/schemas/forecast.schema.json
[common-score]: https://github.com/Agenthon-2026/Agenthon2026-public/blob/bd01548e34d21fd660d88fd06157078fb25ece4e/common/qfbench2_common/scoring/crps.py
