# Stage-Conditioned OPD：完整 Infra Contract

本文描述当前唯一主路径。核心只有两个训练信号：

- Student：学习同一 checkpoint 在 private skill 下构造的 sharpened Teacher target；
- Buyer：学习连续两个 Student checkpoint 相对同一 target 的 gap decrease。

两者严格分离。Buyer reward、任务 verifier、Teacher terminal quality 都不进入
Student token loss。

## 1. Round 语义

Round `k` 开始时持有 `S_k, B_k`：

```text
B_k + S_k 收集 D_k
        ↓
Student distillation: S_k -> S_(k+1)
        ↓
同时服务 previous=S_k, current=S_(k+1)
        ↓
Buyer GRPO 在新 rollout 上计算 stage learning progress
        ↓
B_k -> B_(k+1)
        ↓
下一轮由 S_(k+1), B_(k+1) 改变数据分布
```

新 rollout 只是探测“最近在哪些区域学会了且仍有 gap”，不被解释为造成这次
checkpoint 改进的样本。

`run_coevolution.py` 必须：

1. 在 Student update 前记录 `student_checkpoint_before`；
2. 在 update 后解析 `student_checkpoint_after`；
3. 以前者启动 `policy_previous`，以后者启动 `policy`；
4. 两端健康后才启动 Buyer training；
5. 拒绝缺失或相同 checkpoint；
6. Buyer 与 manifest commit 完成后才停止 previous service；
7. 失败时恢复最后一次已提交的 current Student 与 Buyer。

## 2. 自然 decision 与冻结 Teacher target

每个例子为 `x=(s,a_T)`：

- `s` 是 Student 产生 action 前完整可见状态；
- `a_T` 是 private skill 下生成的一次完整 Teacher macro-action；
- action 可为文本、单工具调用或协议允许的并行工具调用。

`DecisionState` 只在完整 assistant action 边界切分。主训练路径只生成一次 `a_T`，
随后冻结 Student-visible state、Teacher target、target tokens 与 loss mask；不会为了
reward 执行 Teacher takeover continuation。带 terminal continuation 的 Teacher
validation 仅保留为显式 debugging/analysis 工具。

## 3. Canonical TeacherTargetRecord

`TeacherTargetRecord` schema v2 是 Student 与 Buyer 共用的 target contract，包含：

```text
state/action/hint/checkpoint hashes
raw_teacher_target_hash
teacher_target_hash
student_visible_messages
hinted_teacher_messages
teacher_action
target_token_ids / target_loss_mask
raw hinted sparse distribution + support mass
same-checkpoint unhinted sparse distribution + support mass
skill contrast / gate / temperature
sharpened sparse distribution + support mass
raw/sharpened entropy
```

两个 prompt view 必须以完全相同的 Teacher action 结束；private hint 只能存在于
`hinted_teacher_messages`。所有 scoring view 使用同一 tokenizer、chat template、
action serialization 和 target token sequence。

Target cache key 为：

```text
(teacher checkpoint, unhinted reference checkpoint,
 state hash, action hash, hint hash, tokenizer hash, gate-config hash)
```

cache hit 不会重新生成 action、重新取 hinted logits 或重新 sharpen。

## 4. Skill-contrast target construction

当前 checkpoint `S_k` 在相同 token prefix 上给出：

```text
q_h = p(S_k | s, private skill)
p_0 = p(S_k | s)
c_j = KL(q_h,j || p_0,j)
g_j = clip((c_j-low)/(high-low), 0, 1)
T_j = 1 - g_j(1-T_min)
q_tilde_h,j = softmax(z_h,j/T_j)
```

实现使用 raw Teacher top-k 加实际 target token 作为固定显式 support，并保留一个
aggregate tail bucket。Teacher support mass 低于阈值时 fail closed。

必须满足：

- identical hinted/unhinted：`c=0, g=0, T=1, q_tilde_h=q_h`；
- `T_min <= T <= 1`；
- sharpening 不改变显式 support 的排序和 argmax；
- sharpening 对非均匀分布不增加 entropy；
- gate 只改变 target temperature，不再乘一次 Student loss；
- prompt/environment tokens 没有 gate 或 loss。

## 5. Student objective

collection 已将 `q_tilde_h` 缓存在 dataset row 中。训练器验证 schema、action hash、
target hash 与本地 tokenizer 后，对普通 Student-visible prompt 做 forward：

```text
L_student = mean_j KL(q_tilde_h,j || p_student,j)
```

每个 token 在固定 Teacher support 加 tail bucket 上计算 forward KL。所有 target
tensor detached。以下字段不会参与 loss：

```text
learning progress
Buyer reward
terminal verifier score
```

## 6. Three-view stage scorer

Buyer rollout 中每个新 `x` 使用三种 view：

1. current hinted：产生 raw `q_h`；
2. current unhinted：既是 `p_0`，也是 `d_current` 的 Student；
3. previous unhinted：用于 `d_previous`。

先用 1 与 2 构造一次 `q_tilde_h`，再复用：

```text
d_previous = mean KL(q_tilde_h || p_previous)
d_current  = mean KL(q_tilde_h || p_current)
LP         = d_previous - d_current
```

所有距离按 active target-token 数平均。previous/current 对调应翻转 raw LP 符号；
两端 logits 完全相同时 LP 应为数值零。

## 7. Buyer reward 与 GRPO

单 decision 与 trajectory reward：

```text
r(x) = max(LP(x), 0)
R_i  = trajectory_validity * mean_x r(x)
```

一条 Buyer trajectory 的 reward 是所有成功 score 的自然 decision 正 learning
progress 算术平均；没有可 score decision 或 trajectory invalid 时为零。没有
Teacher-quality multiplier、residual gate 或 exploration bonus。

Scheduler 输出：

```text
buyer_reward
reward_source=stage_learning_progress
trajectory_validity / decision_count
previous_gaps / current_gaps
learning_progresses / positive_learning_progresses
decision_rewards
checkpoint_previous / checkpoint_current
raw_teacher_target_hashes / teacher_target_hashes
skill_contrast_scores / skill_gate_values / sharpening_temperatures
raw_teacher_entropies / sharpened_teacher_entropies
scoring_errors
```

同一个 task/prompt 采样 `G` 条 Buyer trajectory。GRPO 只注册
`tau2_stage_learning_progress`，并启用 `--scale_rewards group`。整个 group 为零时保持
全零；非零 group 在训练 advantage 中做组内标准化，raw reward 仍原样记录。不使用
置信下界或其他 reward source 替换。

## 8. Validity 与 fail-closed

以下情况 total reward 为零：

- malformed structured Buyer plan；
- Buyer 发出非法 user tool transition；
- rollout 截断；
- 没有自然 Student action；
- previous/current endpoint 失败；
- target token 不对齐；
- 实际 target token 不在 scoring support；
- Teacher support coverage 不足。

Student 自身答错、工具选错或任务失败不是 Buyer invalidity，因为这些是环境要探测
的能力区域。Scoring error 会保留结构化错误记录，但没有旧 proxy fallback。

## 9. Artifact contract

Dataset schema v4、target schema v2。每个 JSONL、summary 和 manifest 记录：

- tokenizer ID/hash；
- target construction/version；
- reward name/formula version；
- current/previous Student 与 Buyer checkpoint/revision；
- ordered checkpoint tuple；
- deterministic dataset fingerprint。

merge、resume 或 Buyer phase 遇到 schema、tokenizer、target 或 checkpoint ordering
不一致时拒绝继续。

## 10. Reporting 与评测

Trainer 只使用 Weights & Biases，`COEVO_REPORT_TO=wandb`；隔离 smoke 使用
`WANDB_MODE=offline`。任何凭据都不进入仓库。

最终 Student 评测使用官方 held-out split、固定独立 user simulator 和固定 verifier。
learned Buyer 不作为 test user。Buyer reward 只能作为训练诊断，不能称为 held-out
Student improvement。
