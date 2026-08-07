# Reproducible Runtime Setup

这套运行方式不需要宿主机 `sudo`。推荐使用当前机器已有的 Swift 4.1.3 / vLLM
基础镜像，在镜像构建阶段安装 τ²-Bench v1.0.0 和本项目代码。

## 1. 准备配置

在仓库根目录执行：

```bash
cp .env.example .env
```

设置 Student/Teacher 共享的唯一 policy checkpoint 目录：

```text
COEVO_POLICY_MODEL_DIR=/宿主机上的/Qwen3-4B
```

Compose 将它只读挂载到 `/models/policy`。Student 与 Teacher 都调用这一模型；
Teacher 的额外能力来自 `COEVO_TEACHER_HINT_*` 配置的闭源 hinter。Buyer 默认也从
该 checkpoint 初始化；若不使用容器，可覆盖 `.env` 中的 `COEVO_POLICY_PATH` 和
`COEVO_BUYER_PATH`。

如果本机缺少 Qwen3-4B，可在构建运行镜像后从宿主机下载。脚本会继承已有代理
变量、不需要 sudo，并校验权重、配置与 tokenizer 的 SHA-256 是否与固定
Hugging Face revision 完全一致：

```bash
./correctability_coevolution/scripts/download_qwen3_4b.sh \
  /mnt/disk2/models/Qwen3-4B
```

## 2. 获取固定版本上游源码

```bash
./correctability_coevolution/scripts/bootstrap_upstreams.sh
```

脚本使用 sparse checkout，只拉取运行所需的 τ² 文件。容器基础镜像固定 Swift
4.1.3；如需同时审查或在宿主环境 editable-install 对应 Swift 源码，增加
`--with-swift-source` 或 `--install`。固定版本为：

```text
tau2-bench: 17e07b1da2bbc0cadfddeea36412686e0604127b (v1.0.0)
ms-swift:   c6875ef6a962e83f01138bb239b5fb4e5e55b37f
```

上游源码写入被 Git 忽略的 `third_party/`，不会进入本项目提交。

τ² v1 的项目元数据要求 Python `>=3.12,<3.14`。现有 Swift 4.1.3 / vLLM 基础
镜像固定为 Python 3.11，不能脱离同一解释器复用已安装的 CUDA/PyTorch 扩展；因此
Dockerfile 对未经修改的 v1 源码使用明确的 `--ignore-requires-python` 兼容模式。
当前 API、真实数据和全部测试均在该镜像验证，preflight 会把它显示为 `WARN`，而非
冒充官方支持。v1 的文本入口会导入 voice 模块，因此镜像也安装其官方 voice extras
和 PortAudio headers；这些安装发生在容器内，不需要宿主机 sudo。

## 3. 构建并进入容器

```bash
docker compose build coevo
docker compose up -d coevo
docker exec -it swift-grpo-dev bash
```

容器内工作目录已经是：

```text
/workspace/OPD/correctability_coevolution
```

如果使用其他基础镜像，在 `.env` 中覆盖 `COEVO_BASE_IMAGE`。基础镜像必须提供
CUDA、PyTorch、vLLM、TRL 和 ms-swift 4.1.3；构建过程会检查 Swift 版本。

## 4. 运行预检

只检查 Python、依赖和固定版本上游：

```bash
python scripts/preflight.py python
```

启动服务前检查模型、端口、GPU 布局和显存占用：

```bash
python scripts/preflight.py start
```

服务启动后检查三个 endpoint 和 Trainer GPU：

```bash
python scripts/preflight.py services
```

预检默认拒绝在已有进程占用超过 2 GiB 显存的目标 GPU 上启动。只有在明确确认
显存可以安全共存时，才设置 `COEVO_ALLOW_BUSY_GPUS=1`。

## 5. 测试与最小真实 smoke

系统只使用完整自然决策边界：

```text
COEVO_BUYER_PLAN_MODE=structured
```

它在完整 Student action（文本或一组协议允许的并行 tool calls）后建立 Student / 单次
Teacher takeover 分支，两个分支立即交回同一 Student，并用相同 seed 的冻结 continuation
user 计算 category-balanced soft completion 差值。Buyer 只生成 private JSON plan；只有
冻结 Renderer 的 public action 会进入 Student、Teacher 和 verifier history。运行、采集、
Student 数据构造和 Buyer reward 中都不存在句内 token/字符 cutoff 分支。

`COEVO_CONTINUATIONS` 控制 paired continuation 数量；
`COEVO_MAX_INTERVENTION_DECISIONS=0` 表示评估轨迹中全部自然 Student actions。正式跑
Buyer-GRPO 前应先完成 structured-plan SFT 和 reward calibration gate；不要直接把旧
free-form Buyer checkpoint 当成 structured Planner。

```bash
ruff check .
pytest -q
./scripts/start_servers.sh
python scripts/run_coevolution.py \
  --output-dir artifacts/full_smoke \
  --rounds 1 \
  --student-steps 1 \
  --buyer-steps 1 \
  --task-ids 1
```

`start_servers.sh` 会在任一角色启动失败时停止已经启动的本项目服务。
`run_coevolution.py` 会持续写 `manifest.json`；失败时记录 phase 和异常，并在模型
服务刷新失败时尝试恢复上一版 checkpoint。

本机 2026-08-03 的最小真实验收已通过：旧架构的四个服务启动、官方 v1
`airline/train` task 1 采集、Student gated-GKD 1 step，以及官方
`retail/train` task 0 的 Buyer masked-GRPO 1 step。产物位于
`artifacts/v1_infra_smoke/` 和 `artifacts/v1_buyer_reward_smoke/`。Buyer 验收得到
`reward=0.125`、`reward_std=0.1944` 和非零梯度。这两个训练 smoke 使用 LoRA 节省
验收时间；正式入口仍是 `train_student_full.sh` / `train_buyer_full.sh` 的全参训练。

当前固定 vLLM 0.19.1 在这台 A100 节点上使用 2-way TP custom all-reduce 会在 KV
cache profiling 阶段触发 CUDA `invalid argument`，所以 Policy 启动脚本显式传入
`--disable-custom-all-reduce` 并使用 NCCL。Swift 对本地 Qwen3 目录也显式设置
`--model_type qwen3 --template qwen3_nothinking`，不依赖自动推断。
Buyer completion 默认上限为 512 token；此前 128 token 的真实 smoke 会让全部样本以
`finish_reason=length` 结束，只能验证截断保护分支，无法验收在线 τ² scorer。可用
`COEVO_BUYER_MAX_COMPLETION_LENGTH` 调整，但不应把截断样本当成有效 reward。

v1 retail/telecom 的部分任务将 NL assertion 放入 reward basis。judge 与 Teacher
hint 是两条独立路径，并用 `COEVO_NL_JUDGE_MODEL`、`COEVO_NL_JUDGE_URL` 和
`COEVO_NL_JUDGE_MAX_TOKENS` 配置。最终 benchmark 仍使用 τ² 原生 evaluator 的固定
外部 judge。每个服务以独立 Unix process group 启动，`stop_role.sh` 会同时停止父服务
和 `VLLM::EngineCore` 子进程。

## 6. 独立 benchmark user 评测

训练时的 Buyer 是可学习 Qwen3-4B，intervention continuation 使用独立冻结的 Buyer
reference。最终评测不要使用训练后的 Buyer；在 Student 服务已经启动后，用固定
user simulator 运行原生 τ² evaluator：

```bash
export OPENAI_API_KEY=...
COEVO_EVAL_SAVE_TO=student_gpt41_airline \
./scripts/evaluate_student.sh 2 6
```

位置参数是可选 task ID；不传时默认跑该 domain 的整个 `test` split。训练/收集默认
使用 `train` split，可分别用 `COEVO_TASK_SPLIT` 和 `COEVO_EVAL_TASK_SPLIT` 覆盖。
v1.0.0 官方示例使用 `gpt-4.1` 作为 user simulator，模型可由
`COEVO_EVAL_USER_MODEL` 覆盖。
环境本身不是 LLM：它始终由 τ² 的数据库、工具、policy、task 初始状态和 verifier
组成。

## 7. 可配置资源

默认布局占用四张 GPU；Policy 与 Buyer Trainer 按 phase 顺序运行：

| 角色 | 默认 GPU | 默认端口 |
|---|---:|---:|
| Student/Teacher 共享 Policy | 0 | 8000 |
| Buyer reference | 1 | 8002 |
| Buyer rollout | 2 | 8003 |
| 当前 phase Trainer | 3 | - |

GPU、端口、模型路径、Buyer 最大动作数都可以通过 `.env.example` 中对应的
`COEVO_*` 变量覆盖。Student 和 Buyer Trainer 按 phase 顺序运行，可以使用同一张
Trainer GPU；服务 GPU 与 Trainer GPU 不能重叠。
