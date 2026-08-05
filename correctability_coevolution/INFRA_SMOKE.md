# Correctability Co-Evolution Infra 与 Smoke 记录

> 本文件记录 2026-08-02 的 turn-start 单 cutoff LoRA smoke，属于历史验证。
> 当前 Teacher-selected Student-turn 内多 cutoff、全参训练和 round controller
> 以 `FULL_INFRA.md` 为准。

更新日期：2026-08-02

## 1. 结论

独立 infra 已搭建在：

```text
/mnt/disk4/zhangboyao/OPD/correctability_coevolution
```

它不依赖 MAD-OPD，也没有修改 `ms-swift`、`tau2-bench` 或 `slime`。本次真实 smoke 已跑通四段链路：

1. τ²-Bench Buyer–Student 主干 rollout；
2. 相同 cutoff 上的 privileged Teacher / Student terminal continuation 与 verifier；
3. Swift 4.1.3 的 correctability-gated Student GKD/OPD 一步训练；
4. Swift 4.1.3 的 Buyer 多轮 masked-GRPO 一步训练。

## 2. 三模型与职责

```text
Buyer Qwen3-4B (trainable)
    │ 生成用户动作；这些 token 的 GRPO loss mask = 1
    ▼
τ² environment ─────► Student Qwen3-4B (trainable)
    │ 工具、DB、状态       │ 只做 gated OPD，不吃任务 RL reward
    │                     ▼
    └──── cutoff ──► Teacher Qwen3-32B (fixed, oracle actions)
                         │
                         ▼
                    τ² verifier
```

- Teacher：`/models/Qwen3-32B`，固定，使用 τ² `LLMGTAgent` 的 oracle resolution steps。
- Student：`/models/Qwen3-4B`，独立 LoRA，训练目标只有 gated GKD/OPD。
- Buyer：`/models/Qwen3-4B` 的另一套独立 LoRA；训练时它处在 Swift 的 assistant 视角，但语义上是 τ² User。
- Environment / verifier：τ²-Bench 的 policy、tools、DB replay 和 evaluator，不是第四个模型。

## 3. 目录边界

```text
coevo/
├── config.py                  # 模型 endpoint 与实验配置
├── models/tau2_factory.py     # Teacher / Student / Buyer policy 构造
├── environment/tau2.py        # τ² 状态、工具、恢复、terminal verifier
├── rollout/
│   ├── collector.py           # trunk 与 cutoff 数据收集
│   └── views.py               # Student / Buyer 两种角色视图
├── rewards/
│   ├── correctability.py      # qT、qS、C(h)
│   └── buyer.py               # validity gate 与 Buyer reward
└── training/
    ├── gated_gkd.py           # Student 的绝对 gate
    ├── buyer_scheduler.py     # Buyer 多轮 rollout + mask
    └── swift_plugin.py        # 向 Swift 注册 trainer/scheduler/reward

scripts/
├── start_servers.sh
├── stop_servers.sh
├── collect_smoke.py
├── train_student_smoke.sh
└── train_buyer_smoke.sh
```

边界与 `slime/sales_r1` 一致：policy 只出招，environment 只执行和恢复状态，rollout 管循环，reward 独立，Swift plugin 只负责训练框架接入。

## 4. Cutoff 与 reward

当前 smoke 使用最小的语义 cutoff：

```text
Buyer 发出一个完整 user message
            │
            └── cutoff：Student 尚未开始当前 turn
```

在完全相同的 τ² 历史和 DB 状态上：

- privileged Teacher rollout 到 episode terminal；
- Student rollout 到 episode terminal；
- 两边使用同一个固定 Buyer reference；
- branch 内不再递归产生 cutoff；
- τ² verifier 输出成功/失败。

本次每侧采样数 `M=1`，Beta prior 为 `β=1`：

```text
qT = (teacher_successes + β) / (M + 2β)
qS = (student_successes + β) / (M + 2β)
C(h) = qT × (1 - qS)
```

Buyer reward 为：

```text
R_buyer = G_valid × C(h)
```

`G_valid=0` 只在本次 smoke 实际发现的情况触发：Buyer 诱导 Student 调用工具后，τ² 返回 tool error。没有加入未被 smoke 需要的其他惩罚或防御分支。

## 5. Loss mask

Buyer 训练使用 Swift `MultiTurnScheduler`：

- Buyer 的每轮真实采样 `response_token_ids` 返回给 Trainer；
- 对应 `response_loss_mask` 全为 1；
- Buyer 默认启用 Qwen3 thinking；reasoning token 保留在采样序列中参与训练；
- 进入 τ² history 前只提取最终可见 user utterance，内部 reasoning 不会传给 Student、Teacher continuation 或 scorer；
- 如果后端把 reasoning 序列化为 `<think>...</think>`，scheduler 会剥离该区段；未闭合的 think 区段视为没有产生有效用户动作；
- Student 回复作为 Buyer 视角的 `role=user` 插入历史，不进入 response token IDs；
- τ² tool observation 使用 `role=tool`，不进入 Buyer loss；
- Teacher continuation 只用于计算 reward，从不进入 Buyer 训练序列。

因此实际优化的是 Buyer/User policy，不是 Student、Teacher 或工具文本。

Student 训练使用 Swift `GKDTrainer`：

- Student 对当前 cutoff 做 on-policy generation；
- Teacher server 返回 top-20 token log-probabilities；
- privileged oracle steps 通过 `teacher_prompt` 只提供给 Teacher；
- GKD loss 乘以未做组内归一化的绝对 `C(h)`；
- smoke 强制 per-device batch size 为 1，所以 gate 对样本精确生效。

## 6. 运行方式

### 6.1 进入容器

源码在宿主机上可直接用 VSCode 查看和修改；容器只是运行环境，目录通过 bind mount 映射：

```bash
docker exec -it swift-grpo-dev bash
cd /workspace/OPD/correctability_coevolution
```

### 6.2 启动三模型服务

```bash
./scripts/start_servers.sh
```

GPU/端口分配：

| 角色 | GPU | 端口 | 上下文 |
|---|---:|---:|---:|
| Teacher 32B | 0,1 | 8000 | 16384 |
| Student 4B | 2 | 8001 | 16384 |
| Buyer reference 4B | 3 | 8002 | 16384 |

Teacher 服务开启 `max_logprobs=20`，供 Swift GKD 读取 top-k logits。

### 6.3 收集一条 τ² 数据

```bash
PYTHONPATH=/workspace/OPD/correctability_coevolution:/workspace/OPD/tau2-bench/src \
python scripts/collect_smoke.py --output-dir artifacts/smoke
```

默认任务是 `airline:1`。可插拔切换方式：

```bash
COEVO_DOMAIN=airline COEVO_TASK_ID=3 \
PYTHONPATH=/workspace/OPD/correctability_coevolution:/workspace/OPD/tau2-bench/src \
python scripts/collect_smoke.py --output-dir artifacts/airline_task3
```

### 6.4 Student 一步训练

```bash
./scripts/train_student_smoke.sh
```

### 6.5 Buyer 一步训练

```bash
./scripts/train_buyer_smoke.sh
```

### 6.6 停止本项目服务

```bash
./scripts/stop_servers.sh
```

脚本只读取本项目 `runtime/pids` 中的三个 PID。

## 7. 本次真实 smoke 结果

### 7.1 单元与接口

```text
2 passed
```

验证了 correctability 公式和 Buyer validity gate。τ² `airline:1`、Buyer 角色视图、Swift trainer/scheduler/reward 注册均可装载。

### 7.2 τ² rollout

```text
domain: airline
task_id: 1
trunk_messages: 18
cutoffs: 1
teacher_successes: 0/1
student_successes: 1/1
qT: 0.333333
qS: 0.666667
C(h): 0.111111
```

这条样本不是 Teacher headroom 样本：Teacher 失败而 Student 成功，所以 gate 只有 Beta prior 留下的小权重。它证明了信号链能区分低教学价值状态，但不能作为方法效果结论。

产物：

- `artifacts/smoke/trajectory.json`
- `artifacts/smoke/student_gkd.jsonl`
- `artifacts/smoke/buyer_grpo.jsonl`
- `artifacts/smoke/summary.json`

### 7.3 Student gated-GKD

```text
Swift: 4.1.3
trainable LoRA params: 16.5151M
steps: 1/1
loss: 0.01427
grad_norm: 0.03932
memory: 10.16 GiB
runtime: 11.22 s
```

Checkpoint：

```text
artifacts/smoke/student_adapter/v6-20260802-171839/checkpoint-1
```

### 7.4 Buyer masked-GRPO

第一次 2-sample smoke 的两条 reward 相同，Swift 正确得到零 advantage。接入 `G_valid` 后使用 4 samples 重跑：

```text
steps: 1/1
num_generations: 4
num_turns: 2
reward_mean: 0.09722
reward_std: 0.1173
frac_reward_zero_std: 0
loss: -0.01389
grad_norm: 0.3734
memory: 43.89 GiB
runtime: 147.8 s
```

Checkpoint：

```text
artifacts/smoke/buyer_adapter/v3-20260802-172751/checkpoint-1
```

## 8. Smoke 中实际遇到并修掉的问题

只处理了真实运行触发的问题：

1. τ² policy/oracle prompt 超过 4K/8K：外部服务上下文改为 16K。
2. Qwen3 thinking 吃掉 Buyer 输出预算：τ² 与 Buyer rollout 关闭 thinking。
3. `swift` shebang 与 τ² venv 依赖错位：训练脚本统一用当前 `python -m swift.cli.rlhf`。
4. 镜像缓存默认指向不可写 `/mnt/workspace`：HF/ModelScope cache 改到项目 `runtime/`。
5. Swift 默认删除自定义 gate 字段：保留 dataset columns，模型 forward 前删除元数据。
6. OPSD 双 prompt 长度不同与 `use_logits_to_keep` 冲突：Student smoke 关闭该显存优化。
7. Buyer 编造 reservation ID 导致真实 tool error：按既定 `G_valid` 将该 turn 的 reward 置零。

## 9. 当前边界与下一步

当前是可运行的最小 infra，不是完整论文实验：

- 当前 cutoff 是 Student turn 起点，不是 Student 输出内部的多句子 cutoff。
- 当前 `M=1`，估计方差大；正式实验应提高到每侧 2–4 continuations。
- 当前 branch 顺序执行；4 条 Buyer rollout 的一步训练耗时约 2.5 分钟，可在结果一致后再并发化。
- 当前 Student gate 的精确实现要求 per-device batch size 1；正式大 batch 需要把 GKD reduction 改成逐样本 gate。
- Buyer reference 在一个 Buyer phase 内固定；交替训练时应在 phase 结束后用最新 Buyer adapter 刷新 reference 服务。
- 尚未实现句子/action 内多 cutoff、turn 内 mean 聚合、退化惩罚和整组 absolute-low skip；这些应在相应实验开始时作为独立插件加入，而不是塞进 environment。

下一层扩展应保持现有边界：新增 cutoff selector 放在 `rollout/`，新增 reward 项放在 `rewards/`，训练调度留在 `training/`，不要修改 τ² environment adapter。
