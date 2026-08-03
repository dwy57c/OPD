# Teacher-Selected Multi-Cutoff Co-Evolution Infra

更新日期：2026-08-03

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
冻结 Buyer_k + Student_k
  -> 收集完整多轮 trajectory
  -> 对每个完整 Student 文本 turn 提取合法句子边界
  -> 固定 Teacher 从编号候选中选择 Top-K
  -> vLLM continue_final_message 从同一 assistant prefix 续写
  -> Teacher / Student 分别 rollout 到 terminal
  -> verifier 得到 qT、qS 和 C(h)
  -> cutoff mean 得到 turn score C_t
  -> C_t 门控完整 Student turn 的 token-level OPD
  -> 得到 Student_{k+1} 并刷新 Student 服务
  -> 在新 Student 下在线采 Buyer trajectory
  -> turn mean 形成 trajectory reward
  -> 仅 Buyer token 参与 GRPO
  -> 得到 Buyer_{k+1} 并刷新 Buyer rollout 服务
```

## 3. Cutoff

Cutoff 位于已经完成的 Student turn 内部，而不是 turn 起点：

```text
h_t,c = (H_t, Buyer_t, Student_t,<c)
```

`coevo/cutoff/boundaries.py` 只产生可续写的完整句子边界。`TeacherCutoffSelector` 看到当前完整 Student turn、此前历史和 oracle resolution plan，只能返回候选 ID，不能输出任意字符位置。Teacher 的选择理由不进入 reward。

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

batch 内不对 `C_i` 做归一化。Teacher top-20 logits 由端口 8000 的固定 Teacher server 提供。正式脚本使用 `--tuner_type full`。

## 5. Buyer 全参 GRPO

`Tau2BuyerScheduler` 在更新后的冻结 Student 下进行在线多轮 rollout。每个已完成 Student turn 都调用同一套 Teacher-selected cutoff scorer，累计：

```text
R_B = trajectory_validity * mean_t(C_t)
```

Swift loss mask 只覆盖 Buyer response token；Student、工具和 Teacher branch token 都不进入 Buyer loss。端口 8002 是 phase 内固定的 τ² Buyer reference；GRPO rollout 使用独立的 `swift rollout` server mode，完整权重由 Trainer 同步到端口 8003。

## 6. 运行

进入容器：

```bash
docker exec -it swift-grpo-dev bash
cd /workspace/OPD/correctability_coevolution
```

启动 Teacher、Student、冻结 Buyer reference 和 Buyer rollout 服务：

```bash
./scripts/start_servers.sh
```

只收集数据：

```bash
COEVO_CUTOFFS_PER_TURN=2 COEVO_MAX_CUTOFF_TURNS=3 \
PYTHONPATH=/workspace/OPD/correctability_coevolution:/workspace/OPD/tau2-bench/src \
python scripts/collect_round.py \
  --output-dir artifacts/example_round \
  --task-ids 1 2
```

运行一个两步 Student + 两步 Buyer 的完整 round：

```bash
PYTHONPATH=/workspace/OPD/correctability_coevolution:/workspace/OPD/tau2-bench/src \
python scripts/run_coevolution.py \
  --output-dir artifacts/full_run \
  --rounds 1 \
  --student-steps 2 \
  --buyer-steps 2 \
  --task-ids 1
```

如果服务尚未启动，可增加 `--start-services`。

## 7. GPU profile

默认服务布局：

| Role | GPU | Port |
|---|---:|---:|
| Teacher Qwen3-32B TP=2 | 0,1 | 8000 |
| Student Qwen3-4B | 2 | 8001 |
| 固定 Buyer reference | 3 | 8002 |
| Buyer `swift rollout` | 4 | 8003 |
| 当前 phase 的全参 Trainer | 5 | - |

GPU 可以通过 `COEVO_TEACHER_GPUS`、`COEVO_STUDENT_GPU`、`COEVO_BUYER_GPU`、`COEVO_BUYER_ROLLOUT_GPU`、`COEVO_STUDENT_TRAIN_GPUS` 和 `COEVO_BUYER_TRAIN_GPUS` 修改。

## 8. 当前验证

已通过：

```text
ruff: all checks passed
pytest: 5 passed
Swift plugin import: passed
Swift 4.1.3 full tuner / GKD / GRPO / server-mode CLI preflight: passed
```

真实模型 smoke 的产物与指标在完成后写入 `artifacts/full_smoke/`。

2026-08-03 本轮代码完成时，8 张 A100 正被既有 `task37-*` Megatron/SGLang
训练容器占用；本项目没有抢占或停止这些进程，因此尚未生成新的真实模型指标。
