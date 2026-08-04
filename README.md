# Correctability-Driven Environment–Student Co-Evolution

本目录保存“教师可纠正性驱动的环境—学生共进化”项目的研究设计、Swift 基线实现和可复现运行入口。\(\tau^2\)-Bench 固定为官方 `v1.0.0` release commit；它与 ms-swift 上游依赖均由 bootstrap 脚本拉取，不直接提交到本仓库。

项目的核心命题是：

> 环境持续寻找“当前学生不会、固定教师能够可靠纠正”的状态，学生再通过纯 On-Policy Distillation 消除这些失败。

当前真正属于本项目的实现位于 [`correctability_coevolution/`](correctability_coevolution/)。历史 turn-start LoRA 版本和当前 Teacher-selected multi-cutoff 版本都有真实一步 smoke 记录；正式全参、多轮训练仍需单独验收。

新的 slime 版本将放在同级独立目录 `/mnt/disk4/zhangboyao/OPD-ACL`，不覆盖这里的 Swift baseline。本 README 只描述当前 `OPD/` 目录。

## 1. 核心方法

系统包含三个模型和一个非模型环境：

```text
Trainable Buyer / User
        │ 生成用户动作
        ▼
  τ² environment ◄────► Trainable Student
        │                         │
        │                         │ 完整 Student turn
        │                         ▼
        └────────────── Fixed privileged Teacher
                                  │
                                  ▼
                            τ² verifier
```

- **Teacher**：固定模型，拥有 oracle resolution plan 等特权信息；负责选择 cutoff，并提供 token-level 蒸馏分布。
- **Student**：可训练模型；只接受 correctability-gated OPD，不使用任务 reward 做 RL。
- **Buyer**：可训练的闭环用户策略；观察 Student 的回答后继续生成用户动作，使用 GRPO 更新。
- **\(\tau^2\) environment / verifier**：执行工具、维护数据库状态并判断任务是否成功，不是第四个模型。

对一个 Student-induced cutoff state \(h\)，分别估计 Teacher 与 Student 从相同 prefix 继续后的成功率：

\[
q_T(h)=P(\text{Teacher succeeds}\mid h),\qquad
q_S(h)=P(\text{Student succeeds}\mid h).
\]

绝对可纠正性定义为：

\[
C(h)=q_T(h)\left(1-q_S(h)\right).
\]

它同时承担两个作用：

```text
Student loss = C(h) × token-level OPD loss
Buyer reward = validity × trajectory mean correctability
```

Student 和 Buyer 不在同一个 optimizer step 中同时更新。完整训练按 round 交替进行：

```text
收集当前 Student/Buyer 轨迹
  -> Teacher 选择 Student turn 内的 semantic cutoffs
  -> Teacher/Student 从相同 prefix 分叉到 terminal
  -> verifier 计算 qT、qS、C(h)
  -> gated OPD 更新 Student
  -> 在更新后的 Student 下在线 rollout
  -> masked GRPO 更新 Buyer
  -> 进入下一 round
```

完整研究定义、假设、消融和论文边界见 [`correctability_driven_environment_student_coevolution.md`](correctability_driven_environment_student_coevolution.md)。

## 2. 仓库结构

```text
OPD/
├── README.md
├── pyproject.toml
├── Dockerfile
├── compose.yaml
├── .env.example
├── correctability_driven_environment_student_coevolution.md
├── correctability_coevolution/
└── third_party/                 # bootstrap 生成，Git 忽略
    ├── tau2-bench/
    └── ms-swift/
```

| 路径 | 作用 | 是否属于核心实现 |
|---|---|---|
| `README.md` | 仓库入口、代码导航和当前状态 | 是 |
| `correctability_driven_environment_student_coevolution.md` | 研究方案的 source of truth：方法、公式、训练流程、实验和风险 | 是 |
| `correctability_coevolution/` | 独立的 Swift + \(\tau^2\) correctability co-evolution 实现 | 是 |
| `pyproject.toml` | 本项目 Python 包、测试和 Ruff 配置 | 是 |
| `Dockerfile` / `compose.yaml` | 无 sudo 的固定 Swift 运行环境和挂载入口 | 运行基础设施 |
| `third_party/tau2-bench/` | Sierra Research 的 \(\tau^2\)-Bench；由脚本 sparse checkout | 上游依赖 |
| `third_party/ms-swift/` | 可选的 ModelScope ms-swift 固定源码；使用 `--with-swift-source` 拉取 | 上游依赖 |

### 上游快照

| 仓库 | 当前 commit | 说明 |
|---|---|---|
| `third_party/tau2-bench/` | `17e07b1`（`v1.0.0`） | detached sparse checkout |
| `third_party/ms-swift/` | `c6875ef` | detached sparse checkout |

## 3. 推荐阅读顺序

第一次审查代码时建议按以下顺序阅读：

1. [`correctability_driven_environment_student_coevolution.md`](correctability_driven_environment_student_coevolution.md)：先理解为什么使用 absolute correctability。
2. [`correctability_coevolution/SETUP.md`](correctability_coevolution/SETUP.md)：搭建固定版本运行环境并执行 preflight。
3. [`correctability_coevolution/FULL_INFRA.md`](correctability_coevolution/FULL_INFRA.md)：查看当前完整数据流、GPU/端口布局和运行入口。
4. [`coevo/orchestration/collection.py`](correctability_coevolution/coevo/orchestration/collection.py)：查看一轮数据如何落成训练集。
5. [`coevo/rollout/collector.py`](correctability_coevolution/coevo/rollout/collector.py)：查看完整 trunk 如何收集和拆成 Student turns。
6. [`coevo/cutoff/`](correctability_coevolution/coevo/cutoff/) 与 [`coevo/rollout/prefix_branch.py`](correctability_coevolution/coevo/rollout/prefix_branch.py)：查看 Teacher 如何选 cutoff，以及相同 prefix 如何分叉。
7. [`coevo/rewards/`](correctability_coevolution/coevo/rewards/)：查看 \(q_T\)、\(q_S\)、\(C(h)\) 和 Buyer reward。
8. [`coevo/training/gated_gkd.py`](correctability_coevolution/coevo/training/gated_gkd.py)：查看 Student 的 gated OPD/GKD。
9. [`coevo/training/buyer_scheduler.py`](correctability_coevolution/coevo/training/buyer_scheduler.py)：查看 Buyer 多轮 rollout、reward 和 loss mask。
10. [`scripts/run_coevolution.py`](correctability_coevolution/scripts/run_coevolution.py)：查看 Student/Buyer 如何按 round 交替更新。
11. [`correctability_coevolution/INFRA_SMOKE.md`](correctability_coevolution/INFRA_SMOKE.md)：最后查看历史真实 smoke、指标和曾遇到的问题。

## 4. `correctability_coevolution` 文件说明

### 4.1 文档

| 文件 | 作用 |
|---|---|
| `FULL_INFRA.md` | 当前版本的主文档；描述 Teacher-selected multi-cutoff、全参 Student OPD、Buyer GRPO 和 round controller |
| `SETUP.md` | 固定上游、容器构建、preflight、测试和最小真实 smoke 操作手册 |
| `INFRA_SMOKE.md` | 2026-08-02 的历史 LoRA smoke 记录；其中 turn-start 单 cutoff 已被当前 multi-cutoff 设计替代 |

### 4.2 配置与模型构造

| 文件 | 作用 |
|---|---|
| `coevo/config.py` | 定义 Teacher、Student、Buyer endpoint，以及 domain、task split、task、cutoff、continuation、prior、seed 等实验参数；支持从环境变量读取 |
| `coevo/models/tau2_factory.py` | 将三个 endpoint 转成 \(\tau^2\) policy；Teacher 使用 `LLMGTAgent` 获取 oracle plan，Student 使用 `LLMAgent`，Buyer reference 使用 `UserSimulator` |
| `coevo/models/__init__.py` | 导出模型工厂 |

### 4.3 环境层

| 文件 | 作用 |
|---|---|
| `coevo/environment/tau2.py` | \(\tau^2\) adapter；负责初始历史、环境实例、工具/DB 状态、orchestrator、terminal continuation 和 verifier 调用，不包含训练逻辑 |
| `coevo/environment/__init__.py` | 导出 `Tau2Environment` 和消息序列化接口 |

关键原则：environment 只执行状态转移和验证，不计算优化器 loss，也不控制 Student/Buyer 的训练阶段。

### 4.4 Cutoff 层

| 文件 | 作用 |
|---|---|
| `coevo/cutoff/boundaries.py` | 在已经完成的 Student 文本 turn 内寻找可续写的句子边界；不会在任意字符位置截断 |
| `coevo/cutoff/teacher_selector.py` | 将完整 Student turn、历史、oracle plan 和候选边界交给固定 Teacher；Teacher 只能从合法 candidate ID 中选择 Top-K |
| `coevo/cutoff/__init__.py` | 导出 cutoff 数据结构和 selector |

Teacher 的选择理由只用于记录，不直接进入 reward。真正的 reward 来自分叉 continuation 的 verifier 结果。

### 4.5 Rollout 层

| 文件 | 作用 |
|---|---|
| `coevo/rollout/views.py` | 将同一条 \(\tau^2\) history 投影为 Student 视角或 Buyer 视角；同时决定 assistant/user/tool 的角色映射 |
| `coevo/rollout/prefix_branch.py` | 使用 vLLM `continue_final_message=true` 从未完成的 assistant prefix 继续生成，再把分支跑到 terminal |
| `coevo/rollout/cutoff_scorer.py` | 对一个完整 Student turn 生成候选、请求 Teacher 选择、逐 cutoff 估计 correctability，并对 cutoff 分数取 turn mean |
| `coevo/rollout/collector.py` | 收集完整 Buyer–Student trunk，定位 Student turns，调用 scorer，并生成 Student GKD 与 Buyer GRPO 数据行 |
| `coevo/rollout/pipeline.py` | 组装 selector、prefix branch runner、correctability estimator 和 turn scorer |
| `coevo/rollout/__init__.py` | 导出 rollout 公共接口 |

这里采用 retrospective cutoff：先看到完整 Student turn，再在 turn 内选择语义 cutoff。每个 cutoff branch 只继续到 terminal，不在 branch 中递归创建新 cutoff。

### 4.6 Reward 层

| 文件 | 作用 |
|---|---|
| `coevo/rewards/correctability.py` | 对同一 prefix 分别运行 Teacher/Student continuations，使用 Beta prior 估计 \(q_T\)、\(q_S\)，计算 \(C(h)=q_T(1-q_S)\) |
| `coevo/rewards/buyer.py` | 检查真实 tool error 形成 validity gate，并计算 `validity × mean(turn correctability)` |
| `coevo/rewards/__init__.py` | 导出 correctability 和 Buyer reward 接口 |

### 4.7 训练接入层

| 文件 | 作用 |
|---|---|
| `coevo/training/gates.py` | 对逐样本 loss 乘绝对 correctability gate，再取 batch mean；不做组内归一化 |
| `coevo/training/gated_gkd.py` | 自定义 `CorrectabilityGKDTrainer`；复用已收集的 Student response，查询 Teacher top-k logprobs，并对每个样本的 GKD loss 应用 gate |
| `coevo/training/buyer_scheduler.py` | Swift `MultiTurnScheduler` 实现；让可训练 Buyer 与冻结 Student/\(\tau^2\) 交互，在线计算 correctability，并只返回 Buyer token 的 loss mask |
| `coevo/training/swift_plugin.py` | 向 Swift 注册自定义 GKD trainer、Buyer scheduler 和 correctability reward |
| `coevo/training/__init__.py` | 训练模块入口 |

Student 与 Buyer 的 mask 边界不同：

- Student phase：只对已收集 Student turn 的 token-level distillation loss 施加 correctability gate。
- Buyer phase：Buyer 生成 token 的 `response_loss_mask=1`；Student、Teacher branch、tool observation 和环境文本不进入 Buyer loss。

### 4.8 数据编排层

| 文件 | 作用 |
|---|---|
| `coevo/orchestration/collection.py` | 遍历 task/trajectory，调用 collector，并写出 `trajectory.json`、`trajectories.jsonl`、`student_gkd.jsonl`、`buyer_grpo.jsonl` 和 `summary.json` |
| `coevo/orchestration/__init__.py` | 导出数据收集入口 |

### 4.9 启动与训练脚本

| 文件 | 作用 |
|---|---|
| `scripts/start_role.sh` | 按 role 启动 Teacher、Student、Buyer reference 或 Swift rollout server，并记录 PID |
| `scripts/start_servers.sh` | 一次启动四个服务角色 |
| `scripts/preflight.py` | 检查 Python、依赖、固定版本、模型、端口、服务和 GPU 布局 |
| `scripts/bootstrap_upstreams.sh` | sparse checkout 固定版本 τ²-Bench 和 ms-swift；可选安装 |
| `scripts/download_qwen3_4b.sh` | 通过 ModelScope 下载 Qwen3-4B，并按固定 Hugging Face revision 对应的权重、配置与 tokenizer SHA-256 校验 |
| `scripts/wait_for_servers.py` | 轮询 OpenAI-compatible `/v1/models`，直到服务 ready 或超时 |
| `scripts/stop_role.sh` | 只停止本项目 PID 文件记录的指定角色 |
| `scripts/stop_servers.sh` | 停止四个本项目服务 |
| `scripts/collect_smoke.py` | 通用数据收集 CLI；读取环境变量并输出数据集摘要 |
| `scripts/collect_round.py` | 完整 round 的收集入口，目前复用 `collect_smoke.py` 的参数和实现 |
| `scripts/train_student_smoke.sh` | 历史一步 LoRA GKD/OPD smoke |
| `scripts/train_buyer_smoke.sh` | 历史一步 LoRA masked-GRPO smoke |
| `scripts/train_student_full.sh` | Student 全参 GKD/OPD 训练入口 |
| `scripts/train_buyer_full.sh` | Buyer 全参 GRPO 训练入口，使用独立 Swift rollout server |
| `scripts/run_coevolution.py` | round controller：收集数据、更新 Student、刷新 Student 服务、更新 Buyer、刷新 Buyer 服务并写 manifest |
| `scripts/evaluate_student.sh` | 用固定、独立的 user simulator 和原生 τ² verifier 评测 Student；训练后的 Buyer 不参与最终评测 |

### 4.10 测试

| 文件 | 作用 |
|---|---|
| `tests/test_core.py` | 验证 correctability 公式、validity gate、semantic cutoff、Teacher-selected cutoff mean 和逐样本 GKD gate |
| `tests/test_buyer_scheduler.py` | 验证 Buyer 逐行 domain/split/task 语义、最终动作评分、loss mask、非法动作和截断处理 |
| `tests/test_tau2_integration.py` | 用真实 τ² v1 airline/retail/telecom 数据验证官方 split、角色信息边界、v1 消息序列化和 user tool 状态恢复 |

`__init__.py` 文件只负责导出各子包的公共接口；`__pycache__/`、`.pytest_cache/` 和 `.ruff_cache/` 都是生成缓存，不属于源码。

## 5. 数据产物

一次 `collect_dataset` 会写出：

| 文件 | 内容 |
|---|---|
| `trajectory.json` | 单轨迹运行时的完整可读记录 |
| `trajectories.jsonl` | 多轨迹 trunk、Student turns、cutoffs 和 correctability |
| `student_gkd.jsonl` | Student OPD/GKD 数据；含 `messages`、`teacher_prompt`、`correctability` 和 domain/split/task metadata |
| `buyer_grpo.jsonl` | Buyer GRPO 初始状态；后续多轮交互由 scheduler 在线执行 |
| `summary.json` | task 数、trajectory 数、turn/cutoff 数和 correctability 列表 |

训练 checkpoint、日志和以上数据默认写入 `correctability_coevolution/artifacts/` 与 `runtime/`。它们是运行产物，不是核心代码。

## 6. 当前 Swift baseline 的运行方式

### 6.1 首次搭建

```bash
cp .env.example .env
# 编辑 .env 中的 COEVO_TEACHER_MODEL_DIR 和 COEVO_STUDENT_MODEL_DIR
./correctability_coevolution/scripts/bootstrap_upstreams.sh
docker compose build coevo
docker compose up -d coevo
docker exec -it swift-grpo-dev bash
```

完整说明见 [`correctability_coevolution/SETUP.md`](correctability_coevolution/SETUP.md)。所有脚本都会从自身位置发现项目根目录，不再要求源码必须位于 `/workspace/OPD`。

### 6.2 启动服务

```bash
python scripts/preflight.py start
./scripts/start_servers.sh
```

默认布局：

| 角色 | GPU | 端口 |
|---|---:|---:|
| Teacher Qwen3-32B TP=2 | 0,1 | 8000 |
| Student Qwen3-4B | 2 | 8001 |
| 固定 Buyer reference Qwen3-4B | 3 | 8002 |
| Buyer Swift rollout | 4 | 8003 |
| 当前 phase Trainer | 5 | - |

### 6.3 只收集数据

```bash
python scripts/collect_round.py \
  --output-dir artifacts/example_round \
  --task-ids 1 3
```

### 6.4 运行一个交替 round

```bash
python scripts/run_coevolution.py \
  --output-dir artifacts/full_run \
  --rounds 1 \
  --student-steps 2 \
  --buyer-steps 2 \
  --task-ids 1
```

如果服务尚未启动，可增加 `--start-services`。

### 6.5 单元测试

```bash
pytest -q
```

### 6.6 使用独立 User 评测 Student

```bash
export OPENAI_API_KEY=...
COEVO_EVAL_SAVE_TO=student_gpt41_airline \
./scripts/evaluate_student.sh 2 6
```

训练和数据收集默认只读取 τ² v1 官方 `train` split；最终 benchmark 默认只读取
`test` split。airline / retail / telecom 的 train/test 数量分别为 30/20、74/40、
74/40，三组 domain 内均不重叠。训练中的 Buyer 是可学习 Qwen3-4B；最终 benchmark
使用 v1 官方示例里的固定 `gpt-4.1` user simulator。环境不是模型，而是 τ² 的
数据库、工具、policy、task 初始状态和 verifier。

### 6.7 停止服务

```bash
./scripts/stop_servers.sh
```

## 7. 主要配置

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `COEVO_TEACHER_URL` | `http://127.0.0.1:8000` | Teacher endpoint |
| `COEVO_STUDENT_URL` | `http://127.0.0.1:8001` | Student endpoint |
| `COEVO_BUYER_URL` | `http://127.0.0.1:8002` | Buyer reference endpoint |
| `COEVO_DOMAIN` | `airline` | \(\tau^2\) domain |
| `COEVO_TASK_SPLIT` | `train` | 共进化数据使用的官方 task split |
| `COEVO_TASK_ID` | `1` | 默认 task ID |
| `COEVO_CUTOFFS_PER_TURN` | `2` | 每个 Student turn 由 Teacher 选择的 cutoff 数 |
| `COEVO_MAX_CUTOFF_TURNS` | `3` | 每条 trunk 最多评分的 Student turn 数 |
| `COEVO_CONTINUATIONS` | `1` | 每个 cutoff、每个 policy 的 continuation 数 |
| `COEVO_CORRECTABILITY_PRIOR` | `1.0` | Beta smoothing prior |
| `COEVO_BRANCH_MAX_STEPS` | `24` | continuation 最大环境步数 |
| `COEVO_BRANCH_MAX_TOKENS` | `256` | prefix completion 最大 token 数 |
| `COEVO_SEED` | `42` | rollout 与 selector seed |
| `COEVO_EVAL_TASK_SPLIT` | `test` | 最终独立评测使用的官方 task split |
| `COEVO_EVAL_USER_MODEL` | `gpt-4.1` | 最终独立评测使用的固定 user simulator |

模型路径和 GPU 可以通过 `COEVO_*_PATH`、`COEVO_*_BASE_MODEL` 和 `COEVO_*_GPU(S)` 环境变量覆盖；完整列表直接查看 `scripts/start_role.sh`、`train_student_full.sh` 和 `train_buyer_full.sh`。

## 8. 当前验证状态

需要区分历史 smoke 与当前代码状态。

### 历史 LoRA smoke

`INFRA_SMOKE.md` 记录的 2026-08-02 真实链路已经跑通：

- \(\tau^2\) trunk rollout；
- Teacher/Student continuation 与 verifier；
- Student 一步 gated GKD/OPD；
- Buyer 一步 masked GRPO；
- 两个 LoRA checkpoint 均成功保存。

该 smoke 使用的是早期 turn-start 单 cutoff，不代表当前 Teacher-selected multi-cutoff 全量训练已经完成。

### 当前 v1 multi-cutoff 代码

`FULL_INFRA.md` 记录：

- Ruff 通过；
- 14 个单元与 τ² v1 集成测试通过；
- Swift plugin import 和 full-tuner CLI preflight 通过；
- Teacher、Student、Buyer reference 和 Buyer rollout 四个真实服务通过；
- 官方 v1 `airline/train` task 1 的真实采集通过，覆盖 v1 `MultiToolMessage`；
- 新的 Teacher-selected multi-cutoff Student gated-GKD 与 Buyer masked-GRPO 均完成
  1-step LoRA 并保存 checkpoint；产物位于
  `correctability_coevolution/artifacts/v1_infra_smoke/` 和
  `correctability_coevolution/artifacts/v1_buyer_reward_smoke/`（默认被 Git 忽略）；
- v1 retail 的 NL-assertion reward basis 已接入固定 Teacher judge；服务按独立 process
  group 启停，实测可完整清除 vLLM 子进程而不影响其他角色。

Student 最小样本为 `loss=0.0193512`、`grad_norm=0.0718474`；Retail Buyer 单轮
验收为 `reward=0.125`、`reward_std=0.1944`、`loss=0.01389`、
`grad_norm=0.1174`，4 个 generation 的截断率为 25%。这些结果证明真实接口、reward
和训练链路可执行，不证明学习效果。正式全参训练与独立 `test` benchmark 尚未运行，
不能把它描述为已完成全量共进化实验。

## 9. 哪些目录不是核心源码

以下内容不应放进核心代码压缩包：

- `correctability_coevolution/artifacts/`：数据、checkpoint 和 smoke 产物；
- `correctability_coevolution/runtime/`：PID、日志和模型缓存；
- 所有 `.pytest_cache/`、`.ruff_cache/`、`__pycache__/`；
- `tau2-bench/.venv/`、`tau2-bench/data/`、`tau2-bench/web/`；
- `MAD-OPD/outputs/` 和 `MAD-OPD/.cache/`；
- `ms-swift/`、`tau2-bench/`、`MAD-OPD/` 的 `.git/`。

模型权重不在本仓库中；Compose 默认把宿主机模型分别只读挂载为
`/models/teacher` 和 `/models/student`。

## 10. 开发约束

本项目后续修改遵循以下原则：

1. 代码保持干净、简洁，只实现真实主流程，不提前加入没有必要的安全检查。
2. 只有真实运行遇到具体问题后，才增加对应分支或兜底。
3. Smoke 使用真实 API、模型服务和环境；不通过 mock 掩盖链路问题。
4. API、模型服务或环境调用失败时直接抛出异常，保留现场用于排查。
5. environment、rollout、cutoff、reward、training 和 orchestration 保持模块化，不把全部逻辑塞进单个脚本。
6. 不修改上游 `ms-swift` 和 `tau2-bench`；框架接入通过本项目 plugin/adapter 完成。
