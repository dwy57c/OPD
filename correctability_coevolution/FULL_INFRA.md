# Shared-Policy, Closed-Hint Multi-Cutoff Co-Evolution Infra

更新日期：2026-08-05

## 1. 主体入口

完整训练入口：

```text
scripts/run_coevolution.py
    -> coevo/orchestration/collection.py
    -> coevo/rollout/collector.py
    -> coevo/cutoff/*
    -> coevo/rollout/prefix_branch.py
    -> coevo/rewards/*
    -> scripts/train_student_full.sh
    -> scripts/train_buyer_full.sh
```

环境、rollout、cutoff、reward 和训练接入保持独立，不修改 `ms-swift`、`tau2-bench` 或 `slime`。

## 2. 每个 round 的数据流

```text
冻结 Buyer_k + Policy_k
  -> 收集完整多轮 trajectory
  -> 对每个完整 Student 文本 turn 提取合法句子边界
  -> 闭源模型根据当前上下文生成私有 Teacher hint
  -> 同一个 Policy_k 在 hint 条件下选择 Top-K
  -> vLLM continue_final_message 从同一 assistant prefix 续写
  -> Teacher / Student 分别 rollout 到 terminal
  -> verifier 得到 qT、qS 和 C(h)
  -> cutoff mean 得到 turn score C_t
  -> C_t 门控完整 Student turn 的 token-level OPD
  -> 得到 Policy_{k+1} 并刷新唯一 Policy 服务
  -> 在新 Policy 的 Student view 下在线采 Buyer trajectory
  -> turn mean 形成 trajectory reward
  -> 仅 Buyer token 参与 GRPO
  -> 得到 Buyer_{k+1} 并刷新 Buyer rollout 服务
```

## 3. Cutoff

Cutoff 位于已经完成的 Student turn 内部，而不是 turn 起点：

```text
h_t,c = (H_t, Buyer_t, Student_t,<c)
```

`coevo/cutoff/boundaries.py` 只产生可续写的完整句子边界。闭源 hinter
看到当前上下文、policy、工具 schema 和 oracle actions，输出结构化私有 hint；
`TeacherCutoffSelector` 让共享 Policy 在该 hint 条件下查看完整 Student turn，并且只能
返回候选 ID，不能输出任意字符位置。hint 和选择理由都不进入公开 trajectory。

`PrefixBranchRunner` 使用 vLLM 原生：

```text
continue_final_message = true
add_generation_prompt = false
```

因此 Teacher 与 Student 续写的是同一个未完成 assistant message，而不是开启一个新的对话 turn。

## 4. Student 全参 OPD

`CorrectabilityGKDTrainer` 直接使用收集阶段保存的完整 Student response，不在训练时重新生成另一条 response。每个样本先独立计算 OPSD loss：

```text
loss = mean_i(C_i * OPD_loss_i)
```

batch 内不对 `C_i` 做归一化。Teacher top-20 logits 由端口 8000 的冻结 Policy
snapshot 提供；Teacher prompt 等于原 query 加收集时保存的私有 hint。Student 与
Teacher 的模型和权重完全相同，只有输入信息不同。正式脚本使用 `--tuner_type full`。

## 5. Buyer 全参 GRPO

`Tau2BuyerScheduler` 在更新后的冻结 Student 下进行在线多轮 rollout。scheduler 运行在独立的 `swift rollout` server 内；每个 Buyer 输出（包括最后一个）都会先进入对应 task 的 τ² 环境，再决定终止和计算 reward。每个已完成 Student turn 都调用同一套 Teacher-selected cutoff scorer，累计：

```text
R_B = trajectory_validity * mean_t(C_t)
```

Swift loss mask 只覆盖 Buyer response token；Student、工具和 Teacher branch token 都不进入 Buyer loss。scheduler 从每条数据的 `domain` / `task_split` / `task_id` 选择 DB、oracle plan 和 verifier，不再把多 task batch 固定到 task 1。端口 8002 是 phase 内固定的 τ² Buyer reference；GRPO rollout 使用独立的 `swift rollout` server mode，完整权重由 Trainer 同步到端口 8003。

训练后的 Buyer 只用于下一轮生成更有区分度的训练环境，不作为最终 benchmark
user。最终结果使用 `scripts/evaluate_student.sh`，让 Student checkpoint 面对固定的
独立 user simulator（`v1.0.0` 官方示例为 `gpt-4.1`）并交给原生 τ² verifier。
τ² environment 不是模型；它负责数据库、工具执行、业务 policy 和状态验证。

训练/收集默认使用 v1 官方 `train` split，最终评测默认使用官方 `test` split；split
名称和 task ID 一起写入每条训练数据，防止 Buyer scheduler 跨 split 复用错误环境。

## 6. 运行

首次运行先按 [`SETUP.md`](SETUP.md) 拉取固定上游、构建容器并执行 preflight。进入容器：

```bash
docker exec -it swift-grpo-dev bash
```

启动唯一 Policy、冻结 Buyer reference 和 Buyer rollout 服务：

```bash
./scripts/start_servers.sh
```

只收集数据：

```bash
COEVO_CUTOFFS_PER_TURN=2 COEVO_MAX_CUTOFF_TURNS=3 \
python scripts/collect_round.py \
  --output-dir artifacts/example_round \
  --task-ids 1 3
```

运行一个两步 Student + 两步 Buyer 的完整 round：

```bash
python scripts/run_coevolution.py \
  --output-dir artifacts/full_run \
  --rounds 1 \
  --student-steps 2 \
  --buyer-steps 2 \
  --task-ids 1
```

如果服务尚未启动，可增加 `--start-services`。

独立评测 Student（位置参数为可选 task ID）：

```bash
OPENAI_API_KEY=... COEVO_EVAL_SAVE_TO=student_gpt41_airline \
./scripts/evaluate_student.sh 2 6
```

Teacher 使用 Gemini 3.1 Pro 根据最新 history、query、工具结果、policy 和 oracle
actions 在每个 turn 生成私有 hint。Student 和 Teacher 都由同一个 Policy model 负责
工具调用与用户可见回复；Teacher 不直接看到 oracle actions，hint 也不进入公开轨迹。
使用同任务、同 seed 比较无 hint 与闭源 hint：

```bash
COEVO_TEACHER_HINT_URL=... \
COEVO_TEACHER_HINT_API_KEY=... \
python scripts/benchmark_teacher_hint.py \
  --output-dir artifacts/teacher_hint_benchmark \
  --task-split test --task-ids 2 6
```

## 7. GPU profile

默认服务布局：

| Role | GPU | Port |
|---|---:|---:|
| Student/Teacher 共享 Policy | 0 | 8000 |
| 固定 Buyer reference | 1 | 8002 |
| Buyer `swift rollout` | 2 | 8003 |
| 当前 phase 的全参 Trainer | 3 | - |

GPU 可以通过 `COEVO_POLICY_GPUS`、`COEVO_BUYER_GPU`、
`COEVO_BUYER_ROLLOUT_GPU`、`COEVO_POLICY_TRAIN_GPUS` 和
`COEVO_BUYER_TRAIN_GPUS` 修改。

多卡 TP Policy 显式使用 `--disable-custom-all-reduce`，回退到 NCCL。原因是
这台 A100 节点上的 vLLM 0.19.1 custom all-reduce 在 KV cache profiling 阶段返回
CUDA `invalid argument`；NCCL 路径已通过真实启动验证。

## 8. 当前验证

以下是 2026-08-03 旧的“独立 32B Teacher + 4B Student”架构验收记录，仅作为历史
证据；它不代表本次 shared-policy refactor 已进行 GPU 验收：

```text
ruff: all checks passed
pytest: 14 passed
Swift plugin import: passed
Swift 4.1.3 full tuner / GKD / GRPO / server-mode CLI preflight: passed
4 services: Teacher / Student / Buyer reference / Buyer rollout ready
tau2 v1 train split: airline task 1 collection passed
Student LoRA gated-GKD: 1 step, loss 0.0193512, grad_norm 0.0718474
Buyer LoRA masked-GRPO: 1 step, reward 0.125, loss 0.01389, grad_norm 0.1174
```

本轮产物位于 `artifacts/v1_infra_smoke/`。采集限制为 1 个 trajectory、1 个
Student turn、1 个 Teacher-selected cutoff 和每个 policy 1 次 continuation；得到
`qT=1/3`、`qS=1/3`、`C=2/9`。分数包含 Beta(1,1) prior，只用于证明 v1 task、
MultiToolMessage、模型生成、prefix branch 和 verifier 的端到端连通性，不是性能结论。

额外的 Buyer 验收位于 `artifacts/v1_buyer_reward_smoke/`，使用官方
`retail/train` task 0 和单轮 scheduler。4 个 generation 中 1 个被 512-token 上限
截断，其余样本真实执行 retail 工具、Student turn、Teacher/Student terminal branch、
v1 NL-assertion judge、reward plugin 和 Buyer loss mask；最终 `reward_std=0.1944`，产生
非零梯度并保存 checkpoint。NL assertion 默认由独立配置的固定 judge 评判，可通过
`COEVO_NL_JUDGE_*` 覆盖；并发评分用进程内锁隔离 v1 的全局 judge 配置。

所有服务都在独立 Unix process group 中运行，停止 rollout 的实测同时清除了其
`VLLM::EngineCore` 并释放 GPU 4，没有影响其他三个服务。以上均是 LoRA 的最小链路
验收；正式的全参多轮训练和独立 `test` split benchmark 尚未运行，不能据此宣称
实验收敛。
