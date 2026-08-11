# Infrastructure Smoke Record

> **Archived evidence only.** The 2026-08-10 run below predates the current
> positive-LP-v2 contract: it used the old extra reward terms and
> `--scale_rewards none`. It must not be cited as validation of the current
> implementation. Per user instruction, no replacement model/training smoke was
> started while other GPU work was active; only static and CPU-level tests are
> permitted for this revision.

本文件只记录基础设施验证，不把训练 reward 写成 held-out 能力提升。当前主路径以
[`TODO(2).md`](../TODO(2).md) 和 [`FULL_INFRA.md`](FULL_INFRA.md) 为准：Student
学习一次生成并缓存的 skill-contrast-sharpened Teacher target；Buyer 只学习相邻
Student checkpoint 在同一 target 上的 stage learning progress。主路径没有 Shadow
OPD、utility critic 或 intervention-advantage reward。

## 2026-08-10：TODO(2) 本地端到端 smoke

运行条件：

```text
model:              /mnt/disk4/zhangboyao/models/modelscope/Qwen3-4B
runtime:            existing local ms-swift 4.1.3 / TRL 0.29.1 / vLLM container
model downloads:    disabled (HF/Transformers/ModelScope offline)
experiment logger:  Weights & Biases 0.25.0, offline mode
SwanLab reporter:   disabled
domain/split:       airline/train
```

### Student one-step

Smoke fixture 使用 schema v3，并先从当前 policy 的 hinted/unhinted prompt log-probs
构造一次 `TeacherTargetRecord`。Student 训练只读取这个缓存 target，没有在 optimizer
step 内重新生成 Teacher target，也没有乘 learning progress、Teacher quality 或
verifier 权重。

```text
global step:                 1/1
loss:                        3.84868813
gradient norm:               5.86214256
skill contrast mean:         3.73160148
skill gate mean:             0.64843863
sharpening temperature mean: 0.80546838
target support mass:         0.99999964
```

LoRA checkpoint 已保存并合并成可服务的 S1 checkpoint：

- `artifacts/todo2_infra_smoke_20260810/student_adapter/v2-20260810-191213/checkpoint-1`
- `artifacts/todo2_infra_smoke_20260810/student_merged_s1`

### 相邻 checkpoint 三视角打分

S0 使用原始本地 Qwen3-4B，S1 使用上面的 one-step Student。固定同一个 sharpened
Teacher target，分别计算 S0 与 S1 的 unhinted gap。合成样本故意产生了负 LP：

```text
previous gap (S0):       3.74203731
current gap (S1):        3.78589516
learning progress:      -0.04385785
positive LP:             0
residual gate:           1
decision reward:         0
```

这验证了回归样本不会获得正 reward；完整 target hash、熵和逐 token gate 记录在
`artifacts/todo2_infra_smoke_20260810/stage_gap.json`。

真实 τ² task-3 单轨迹还覆盖了正 LP 路径。三个成功打分的 decision 中，LP 分别为
`0.00424956`、`0.00796962`、`-0.00003639`；其中第二个 decision 的
`teacher_quality=0.5`，得到局部 reward `0.00398481`。由于该诊断 rollout 在 4 个
Buyer turns 后仍未显式 stop，`trajectory_validity=0`，最终 trajectory reward 被硬门控
为 0。这是预期的 fail-closed 行为，不是 reward fallback。

### Buyer one-step GRPO

Buyer smoke 使用两个真实 τ² train prompts（task 1、3），每个 prompt 采样两个 Buyer
completion，完成一个 GRPO optimizer step。最终代码版本的结果：

```text
global step:                       1/1
BuyerStageProgressReward mean:     0.0000191659
reward std:                        0.0000271047
loss:                              0.0000191659
gradient norm:                     0.0000879256
mean Buyer turns:                  4
checkpoint:                        checkpoint-1
```

非零 reward、非零 reward variance 和非零 gradient 共同确认：真实 Buyer rollout、
one-sided Teacher validation、缓存 target、S0/S1 三视角 gap、stage-progress reward、
absolute group handling 与 GRPO 反向传播已经接通。checkpoint 位于：

`artifacts/todo2_infra_smoke_20260810/buyer_adapter_final/v0-20260810-194446/checkpoint-1`

W&B offline run：

`artifacts/todo2_infra_smoke_20260810/wandb/wandb/offline-run-20260810_194451-pbxm7nt9`

### Smoke 过程中修复的真实兼容性问题

- Transformers v5 `apply_chat_template` 返回 `BatchEncoding`，现在显式提取
  `input_ids`。
- 文本 target 去掉 Qwen turn terminator，与 Swift `add_eos=False` labels 对齐。
- 纯 tool-call assistant action 不再被 `continue_final_message=True` 静默截成空 target；
  完整 macro-action tokens 会保留，只移除 turn terminator。
- Teacher tool macro-action 会先在重建后的 τ² environment 中执行，再把控制权交还
  Student；不再把 dangling tool call 交给 state replay。
- `COEVO_BRANCH_MAX_TOKENS` 现在真正传入 Student/Teacher endpoints。
- 极小 GRPO fixture 至少生成满足 generation batch 的两个 prompt，避免训练循环以
  `global_step=0` 静默结束。

## 自动化验证

隔离 Docker 全量测试：

```text
51 passed, 22 warnings
```

warnings 来自已安装依赖的 deprecation/experimental import；Trainer 的实际
`report_to` 为 `['wandb']`，W&B 明确处于 offline mode。

## 结论边界

该 smoke 证明当前 infra 能以本地模型完整执行 Student target 构造与更新、相邻
checkpoint 打分、Buyer stage-progress reward 和一步 GRPO。它不证明 Buyer 数据比
基线更好，不证明 official test 能力提升，也不构成多轮共进化实验结果；这些仍需按
正式等预算、held-out 协议运行。
