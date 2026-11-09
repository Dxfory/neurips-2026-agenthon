# Track 1 — Coding

> **当前状态（2026-10-22）：** T1 的 Final 候选是本目录的 Python 解题 agent（`t1_phase1/thin.py` 的 thin 循环，profile `stack14-thin-v1`），开发分支 `claude/t1-final-py`。原生 C++ 线已于 2026-10-21 停用，最后状态保存在 tag `t1-native-cpp-line`。R2 起的新机制都挂在 `T1_THIN_FEATURES` 开关后，镜像默认关闭。本文下方的第一阶段介绍保留为历史基线。

最新实验状态见 [R2 X′ / Y′ 台账](../docs/experiments/t1-r2-status.md)（2026-10-23，北京时间）：X′ 已完成，16/86 本地通过；相对 E1b 新增通过 7 题、退步 5 题。Y′ 于 08:05:36 完成，13/86 本地通过；相对 X′ 新增通过 2 题、失去通过 5 题，共同通过 11 题，另有 1 题评分器错误未判定。本次结果不支持将 Y′ 提升为默认，但单次随机运行不足以证明因果。Y′ 仅增加 `repair_feedback` 开关，但该开关同时改变初始环境说明与失败后的修复提醒；R3 仍待单独决定。本次仅同步文档，不改变冻结代码、默认开关或运行配置。

当前研发流程见 [统一解题流水线](PIPELINE.md)：比赛准备、模型与控制器分工、单题流程，以及模型 token 与本地执行成本。
当前默认 profile 是 `stack14-thin-v1`。显式启用 `T1_THIN_DIRECT_STEP=1` 时，控制器确定动作和版本，模型只提交代码；程序在隔离环境中执行，再按实际反馈修复。该 direct 模式仍默认关闭。
direct 模式在当前版本已执行成功、无未完成代码时使用 `APPEND_STEP`：模型只提交新代码，控制器保留原文件并追加，随后清空输出、重跑完整程序。完整旧前缀可核对时，控制器提取新段；旧代码字节不变，新增导入不能覆盖旧名称。截断、修改/缺失前缀、无新增工作、重复有效追加段及未完成候选均原子拒收，并留下每次准入记录。提示附少量所引用 NumPy/SciPy API 的真实 Returns 文档，数值正确性仍需实际检查。仅新的规范 AST 版本实际执行成功才清零无进展计数；两次无此进展停止，原有单次强制替换恢复机会及预算边界保留。这是执行进展，不证明新增计算有用或金融答案正确；任务通过仍须原 checker。
思考按请求类型分开设置：`T1_THIN_CONTRACT_THINKING`（契约请求，镜像默认 `low_effort`）与 `T1_THIN_CODE_THINKING`（写码与续写请求，镜像默认 `off`），取值 `on`、`low_effort`、`off` 或 `budget:<1-2048>`，由 `solver_profile.profile_env` 写入镜像 ENV。某一类型未设置时，才回落到旧的全局 `T1_THIN_THINKING`/`T1_THIN_REASONING_BUDGET`（`on` 保留原生思考，数值 budget 仅能与 `on` 同用）；镜像两项都设置，所以旧开关在镜像里不起作用。
free 模式（镜像现行）：写码回复在 4,000 token 处截断且仍在代码块内时，控制器从最后一个完整行发续写请求（每条回复至多 3 次，每次计 1 个请求），拼接后按普通回复准入；写码回复里的代码由同一个提取器（`program_version.extract_code`）取出，头部路径与无头路径共用：容忍多余或重复的裸围栏、代码块前后的说明文字和多个代码块；只选一个块，从不拼接，优先能编译的 python 块（length 截断处的块排在已闭合的块之后，只有没有完整程序时才做截断打捞），其次点名全部期望输出文件的块，再取最长；更长的 python 块有真语法错时报它的错，不让更短的用法示例顶替；一个程序被拆在几个块里（含另一个块用所选块的名字并写文件）时拒收，取不出程序时反馈写明具体的格式问题（`no_code_block`、`ambiguous_code_blocks`、`split_code_blocks`、带行号的 `invalid_syntax`）。缺少 REPLACE/BUNDLE 头但取出的代码是可直接运行、无未定义名字的完整程序时，按 REPLACE 准入，否则反馈说明未准入的原因；连续 4 条回复未改变程序或待完成项才停止（每次停滞改写提示）；429/408/409/425/5xx 与传输失败按 2、4、8、15、25、40 秒退避重试（遵守 Retry-After），每次尝试都计请求，且须留出预留时间外至少 30 秒；后一次较差的执行不会覆盖已交付更多文件、或同等交付但正常退出的较早版本，收尾时恢复其输出与状态。
通用运行时的本轮实现见 [qfruntime 0.2](qfruntime/README.md) 和
[六项能力实施与验收](SIX_CAPABILITIES_PLAN.md)，`T1_RUNTIME` 仍默认关闭。
以下第一阶段介绍与对应证据保留为历史基线，不代表当前实验代码已发布或提交。

最新控制器默认每题最多25次请求（Final 上限，重试与续写都计入）。可用 `T1_REQUEST_CAP=1…25` 明确下调本次资源上限；仅接受规范ASCII整数，非法值在读取House凭据前拒绝。Docker本地模型运行会转发已验证的变量名，实际上限写入status及公开评测row的`request_cap`；显式配置另写replay资源记录。请求资源上限单列于评测协议，源码/策略identity不包含它。下调不改变每次输出、时间或检查标准，较低预算的结果须单独报告。

这是第一阶段的本地研究基线：每题至多发起一次模型请求，在独立容器中运行生成程序，再由独立 checker 容器评估产物。原生 AMD64 CI 已通过 59 个单元测试、9 场景、22 项验收、13 项隔离探针及 6 项外层超时检查；本机 QEMU 下的资源配置版本也通过容器验证。**GHCR 发布与远程 digest 核验已完成，匿名拉取仍要求认证**。见[当前验证记录](verification/native-amd64.md)及[首个本地历史快照](verification/summary.md)。真实 House 调用为 0，尚未正式打包或官方提交。

## 环境与源码

宿主需要 Python 3.13 和可用的 Docker。容器目标为 `linux/amd64`。示例显式使用 `colima` context；其他环境可替换 `--context`，不需要修改全局 Docker 默认 context。Apple Silicon 上的 AMD64 执行属于本地跨架构验证，性能不等同官方平台。

| 路径 | 用途 |
| --- | --- |
| `t1_phase1/` | 输入 staging、House 客户端、单次生成、Docker runner 与本地评测 |
| `scripts/solve` | 容器内 `solve --task-dir /input --out /app/output` 入口 |
| `scripts/integration_fixture.py` | 离线合成 HTTP 服务及九种正常/故障场景 |
| `scripts/timeout_fixture.py` | 固定睡眠程序的外层超时与容器清理验证 |
| `scripts/fetch_sources.py` | 固定 commit 的公开题目下载与哈希验证 |
| `selection/suite.json` | 固定的 12 题清单：9 dev、3 本地 holdout |
| `evidence/source_files.json` | 官方公开文件路径、大小与 Git blob SHA 索引 |

以下命令均在 `T1/` 中运行。宿主单元测试只需 Python 标准库；不会在宿主执行模型生成程序。

```sh
python3.13 -m unittest discover -s tests -v
```

## 构建与离线容器测试

```sh
python3.13 build/amd64/download_wheels.py
python3.13 build/amd64/verify_wheelhouse.py
python3.13 build/amd64/build_image.py \
  --context colima --image t1-phase1:b0-amd64 \
  --out-dir evidence/image-build/amd64-001

python3.13 scripts/integration_fixture.py \
  --context colima --platform linux/amd64 --profile minimal --image t1-phase1:b0-amd64 \
  --out-dir runs/integration-amd64 \
  --gate-receipt runs/gates/amd64.json

python3.13 scripts/timeout_fixture.py \
  --context colima --platform linux/amd64 --image t1-phase1:b0-amd64 \
  --participant runs/integration-amd64/participant \
  --out-dir runs/timeout-amd64
```

下载并校验47个固定版本的官方PyPI wheel（含与官方 finance-bench-sandbox 对齐的分析包）后，构建器以显式文件清单提供离线上下文，使用固定的AMD64基础镜像digest和多阶段Dockerfile。普通 `Dockerfile` 仅保留作旧版诊断入口，候选构建请使用上述锁定流程。

运行目录必须是新目录；重复执行时换一个名称，保留之前的证据。`--prepare-only` 只生成合成夹具，不更新 gate receipt。实际尝试运行即更新选定 gate，失败也会使原有通过记录失效。未指定 `--gate-receipt` 时仍写入 `evidence/integration/result.json`。

集成测试只向容器内 loopback 合成服务请求，不访问 House。场景覆盖正常返回、缺少配置、HTTP 401/403/503、截断代码、Python 语法错误、执行异常和执行超时，并检查正负 checker 结果。外层超时另由固定睡眠夹具验证。两个夹具均支持 `--cpus` 和 `--memory`：本机默认 4 CPU/8 GiB；当前 GitHub CI 显式使用 2 CPU/6 GiB，报告会记录实际 profile。容器默认无外网、只读根目录、UID 65534、4 CPU、8 GiB 内存、64 MiB `noexec` `/tmp` 和 256 PID 限制；这是本地 profile，不是官方资源等价证明。

## 获取与隔离公开任务

```sh
python3.13 scripts/fetch_sources.py
python3.13 -m t1_phase1 prepare --root . --participant runs/participant
```

下载来源是官方 `Agenthon-2026/track1-coding-public` 的固定 commit `1a60fc48024c4f0e9978416d84e8370dbb0236cd`，共 12 题、121 个文件。下载器核对 Git blob SHA 和大小，写入自己的 SHA256 manifest；重跑会复验已有文件。原始数据及 checks/reference 只供宿主评测侧使用，不进入提交源码。

Participant staging 仅保留允许的任务说明、card、数据和任务模板，删除评分材料及 canary 标记行，并拒绝路径穿越和符号链接。官方 Dockerfile 不会执行。与平台一致，agent 容器只看到 `/input`、`/app/output` 与 `/output`，没有 `/app` 数据挂载；静态 `COPY data` 映射只给 checker 提供只读 `/app` 路径，`solve` 则把同一映射（本地取自 staging manifest，官方取自 `/input/environment/Dockerfile`）写成提示词里的 `/app` → `/input` 路径对照。Agent 输出在交给 checker 前再次隔离，既有 reward/report 不作为评分证据。

这 3 道 holdout 仍来自官方 public-dev；它们只是本地流程隔离，不是官方隐藏测试，也不能声称研究者从未见过。冻结后不得依据 holdout 结果调整本轮方法。

## 后续 House 测量

官方评测时平台会注入 `MODEL_ENDPOINT`、`MODEL_NAME`、`MODEL_TOKEN`。本地真实测量须先得到组织方正式提供的开发接入，并完成相同镜像的 AMD64 集成验证。不要将凭据写入源码、镜像、命令参数或提交文件。配置缺失时不会发起模型请求；服务错误与模型/程序错误分别记录。

```sh
python3.13 -m t1_phase1 preflight \
  --context colima --platform linux/amd64 --image t1-phase1:b0-amd64

python3.13 -m t1_phase1 evaluate --root . --split dev \
  --context colima --platform linux/amd64 --image t1-phase1:b0-amd64 \
  --run-dir runs/dev-amd64 \
  --integration-evidence runs/gates/amd64.json
```

模型访问使用带网络的本地 profile；它不等同官方仅允许指定端点的网络隔离。这里汇总的是本地 checker 结果，不重写或冒充官方 `pass@k` scorer。未完成完整 roster 或基础设施失败时不会产生完整本地通过率；官方提交与排名须通过组织方指定流程验证。

## 手动 CI 验证与 GHCR 发布

`t1-candidate.yml` 只通过 `workflow_dispatch` 手动触发，push 不会启动长时间构建，`publish` 默认为 `false`。工作流在 Ubuntu 24.04 原生 AMD64 runner 上使用 Python 3.13，依次完成单元测试、固定 wheel 下载与校验、离线构建、九场景集成与正负 checker、外层超时清理，并核对同一镜像的验收收据。整个验证流程不调用真实 House。

在仓库根目录执行：

```sh
# 仅构建与测试，不发布。
gh workflow run t1-candidate.yml -f publish=false

# 所有原生 AMD64 检查通过后，才使用临时 GITHUB_TOKEN 登录并推送 GHCR。
gh workflow run t1-candidate.yml -f publish=true
```

发布目标为 `ghcr.io/yiping-yin/agenthon-t1:sha-<完整提交SHA>`。工作流在 push 成功后读取指定 repository 的 `RepoDigests`，并按 digest 查询远程 manifest 核验；本地 image ID 单独记录，不作为远程 digest。登录信息放在 runner 临时目录，发布后清理，不进入镜像或证据 artifact。

GHCR 包首次创建时可能仍是 private。发布成功后，工作流使用独立的空 Docker 配置另做匿名 pull，写入 `evidence/publish/anonymous-pull.json`；认证要求、网络失败或其他拉取错误都会保留明确状态。**发布成功不代表镜像已经可以公开匿名拉取。** 如需 public 镜像，应调整 GitHub 包可见性，并重新取得匿名拉取成功的证据。

每次运行上传经过路径与凭据清理的测试 JSON、构建日志和上下文哈希清单，artifact 名称为 `t1-amd64-evidence-<run_id>-<attempt>`，保留 14 天；不上传 wheel 二进制、下载分片、任务数据或凭据。
