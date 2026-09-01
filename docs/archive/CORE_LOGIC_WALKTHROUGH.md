# Correctability Co-Evolution 核心逻辑完整导读

> 本文只解释项目的算法与代码闭环，不展开 Docker、GPU、服务部署和历史实现。
> 读完后，你应该能够从一条 Buyer–Student 对话开始，一直追踪到 Student loss、
> Buyer reward，以及下一轮共进化数据分布的变化。

## 1. 一句话理解整个项目

项目训练一个闭环 Buyer，持续寻找：

> **当前 Student 无法完成，但固定特权 Teacher 能够从同一个 Student prefix 接管并完成的状态。**

这些状态的价值用绝对可纠正性表示：

\[
C(h)=q_T(h)\left(1-q_S(h)\right).
\]

- 对 Student：`C(h)` 是 OPD/GKD loss 的绝对 gate。
- 对 Buyer：`C(h)` 是环境策略的训练 reward。
- 当 Student 学会某类状态后，`qS` 上升、`C` 下降，Buyer 必须寻找新的弱点。

这就是完整闭环：

```text
Buyer 发现可纠正失败
  → Teacher 提供 on-policy 蒸馏监督
  → Student 消除失败
  → 原状态 reward 下降
  → Buyer 转向新的能力缺口
```

## 2. 四个角色与信息边界

| 角色 | 是否训练 | 能看到什么 | 负责什么 |
|---|---|---|---|
| Student | 是 | 普通对话历史、用户输入、工具结果 | 与任务环境交互；只通过 gated OPD/GKD 学习 |
| Buyer | 是 | 用户 scenario、Student 已经给出的回答 | 生成下一轮用户动作；通过 GRPO 寻找高价值状态 |
| Teacher | 否 | Student prefix、完整历史、oracle resolution plan | 选择 cutoff；从相同 prefix 接管；提供 token 分布 |
| τ² environment/verifier | 否 | 任务、数据库、工具调用和最终状态 | 执行状态转移；判断任务是否成功 |

角色构造在 [`coevo/models/tau2_factory.py`](correctability_coevolution/coevo/models/tau2_factory.py)：

- Student 使用 `LLMAgent`，不接收 task oracle actions。
- Teacher 使用 `LLMGTAgent(task=task)`，因此拥有 privileged resolution plan。
- Buyer reference 使用 `UserSimulator`，其 system prompt 包含 `task.user_scenario`。

环境封装在 [`coevo/environment/tau2.py`](correctability_coevolution/coevo/environment/tau2.py)。
这里不计算训练 loss，只负责：

1. 创建或恢复任务状态；
2. 执行工具调用；
3. 推进 Buyer–Student 对话；
4. 从任意 history 继续到 terminal；
5. 调用 τ² verifier 判断成功与否。

### 2.1 最重要的信息隔离

必须始终保持以下约束：

```text
oracle plan ──► Teacher
oracle plan ──X Student
oracle plan ──X 部署时的 Student
```

Verifier reward 也不直接训练 Student：

```text
Verifier：决定“这个状态值不值得学”
Teacher logits：决定“Student 在这里具体学什么”
```

## 3. 数学对象如何对应到代码

假设第 `t` 轮之前的历史为 `H_t`：

```text
H_t
  → Buyer 生成用户动作 u_t
  → Student 生成完整回答 a_t
```

在 Student 回答内部选择一个 cutoff `c`：

\[
h_{t,c}=(H_t,u_t,a_{t,<c}).
\]

`h` 不是一个新的任务，它是原任务中由 Student 自己产生的中间状态。

从同一个 `h` 出发：

```text
h ──► privileged Teacher continuation ──► terminal ──► verifier
│
└──► current Student continuation ──────► terminal ──► verifier
```

采样 `M` 次后，当前代码使用 Beta prior 平滑：

\[
\hat q_T=\frac{n_T+\beta}{M+2\beta},
\qquad
\hat q_S=\frac{n_S+\beta}{M+2\beta}.
\]

然后计算：

\[
C(h)=\hat q_T\left(1-\hat q_S\right).
\]

代码位于
[`coevo/rewards/correctability.py`](correctability_coevolution/coevo/rewards/correctability.py)。

### 3.1 四种状态的含义

| Teacher | Student | `C(h)` | 含义 |
|---|---|---:|---|
| 成功率高 | 成功率低 | 高 | 最有教学价值 |
| 成功率高 | 成功率高 | 低 | Student 已经掌握 |
| 成功率低 | 成功率低 | 低 | Teacher 也无法可靠纠正 |
| 成功率低 | 成功率高 | 很低 | Teacher 不是可靠标尺 |

因此 Buyer 追求的不是最大 Student failure，也不是最大 Teacher success，而是二者的交集：

```text
Teacher succeeds AND Student fails
```

### 3.2 一个最小数值例子

假设 `M=1`、`beta=1`，Teacher 成功、Student 失败：

\[
q_T=\frac{1+1}{1+2}=\frac{2}{3},
\qquad
q_S=\frac{0+1}{1+2}=\frac{1}{3}.
\]

所以：

\[
C=\frac{2}{3}\left(1-\frac{1}{3}\right)=\frac{4}{9}.
\]

Beta prior 防止一次随机成功或失败直接得到 0/1，但也意味着 `M=1` 的分数会明显受到 prior 影响。

## 4. 一轮共进化的完整调用链

总 controller 是
[`scripts/run_coevolution.py`](correctability_coevolution/scripts/run_coevolution.py)：

```text
run_coevolution.py
│
├── A. collection
│   └── collect_round.py
│       └── orchestration/collection.py
│           └── rollout/collector.py
│               ├── τ² Buyer–Student trunk
│               ├── Teacher-selected cutoffs
│               ├── Teacher/Student terminal branches
│               └── student_gkd.jsonl + buyer_grpo.jsonl
│
├── B. student_training
│   └── train_student_full.sh
│       └── CorrectabilityGKDTrainer
│
├── C. student_refresh
│   └── 用 Student 新 checkpoint 刷新 Student service
│
├── D. buyer_training
│   └── train_buyer_full.sh
│       └── Tau2BuyerScheduler 在线生成 Buyer trajectory
│
├── E. buyer_refresh
│   └── 用 Buyer 新 checkpoint 刷新 Buyer reference/rollout service
│
└── F. 下一 round
```

一轮中的模型版本关系是：

| 阶段 | Student | Buyer | Teacher |
|---|---|---|---|
| 收集 Student 数据 | `Student_k` 固定 | `Buyer_k` 固定 | 固定 |
| Student 更新 | 训练得到 `Student_{k+1}` | 不变 | 固定 |
| Buyer 在线 GRPO | `Student_{k+1}` 固定 | 训练得到 `Buyer_{k+1}` | 固定 |
| 下一轮收集 | `Student_{k+1}` 固定 | `Buyer_{k+1}` 固定 | 固定 |

Student 和 Buyer 不在同一个 optimizer step 中同时快速变化。

## 5. 数据收集：如何得到一个可训练的 Student turn

主入口是
[`coevo/orchestration/collection.py`](correctability_coevolution/coevo/orchestration/collection.py)
中的 `collect_dataset()`。

它按 task 和 seed 遍历，每完成一条 trajectory 就立即保存，因此支持 resume，并避免长任务中途失败后丢失已完成结果。

单条 trajectory 由
[`coevo/rollout/collector.py`](correctability_coevolution/coevo/rollout/collector.py)
中的 `CorrectabilityCollector.collect_one()` 生成。

### 5.1 第一步：收集完整 trunk

```python
orchestrator = env.orchestrator(initial_history, "student")
orchestrator.initialize()
while not orchestrator.done:
    orchestrator.step()
trunk = orchestrator.get_trajectory()
```

这里的 Buyer 是当前 phase 冻结的 Buyer reference，Student 是当前 Student service。
得到的 trunk 包含：

- Buyer/User message；
- Student/Assistant message；
- Student 工具调用；
- Buyer 用户工具调用；
- 工具 observation；
- 完整任务历史。

### 5.2 第二步：定位可评分 Student turn

Collector 遍历 trunk 中的 `AssistantMessage`，然后调用：

```python
scorer.score_turn(trunk, message_index)
```

当前只评分满足以下条件的 turn：

- 是 Student 的 `AssistantMessage`；
- 包含非空文本；
- 不是一个 tool-call message；
- 文本内部存在合法语义边界。

### 5.3 第三步：生成合法 cutoff 候选

[`coevo/cutoff/boundaries.py`](correctability_coevolution/coevo/cutoff/boundaries.py)
中的 `semantic_boundaries()` 在 Student 完整文本中寻找句子边界。

它不会让模型返回任意字符 offset，而是先生成受约束的候选：

```text
candidate_id
char_offset
prefix_tail
suffix_head
boundary_type = sentence
```

同时要求 cutoff 前后都有足够文本，避免在回答起点或结尾制造无意义分支。

### 5.4 第四步：Teacher 选择 Top-K cutoff

[`coevo/cutoff/teacher_selector.py`](correctability_coevolution/coevo/cutoff/teacher_selector.py)
中的 `TeacherCutoffSelector.select()` 接收：

- cutoff 之前的历史；
- Student 完整回答；
- privileged resolution plan；
- 全部合法 candidate。

Teacher 通过严格 JSON schema 返回固定数量的候选 ID 和选择理由。

关键点：

- Teacher 的 reason 只用于审计；
- reason 不进入 reward；
- 真正的 reward 必须来自 terminal continuation 的 verifier outcome。

### 5.5 第五步：构造完全相同的 partial prefix

[`coevo/rollout/cutoff_scorer.py`](correctability_coevolution/coevo/rollout/cutoff_scorer.py)
把完整 Student message 截为：

```python
partial = message.content[:offset]
cutoff_history = [*history_before, partial_assistant_message]
```

Teacher 和 Student 都接收同一份 `cutoff_history`。

### 5.6 第六步：续写同一个未完成 message

[`coevo/rollout/prefix_branch.py`](correctability_coevolution/coevo/rollout/prefix_branch.py)
使用：

```text
continue_final_message = true
add_generation_prompt = false
tool_choice = none
```

这意味着 Teacher/Student 先续写同一个未完成的 Student message，而不是新开一个 assistant turn。

完成当前 message 后，再调用：

```python
environment.continue_to_terminal(completed_history, policy, seed)
```

让对应 policy、固定 Buyer reference、工具和环境继续交互到 terminal。

### 5.7 第七步：Verifier 评分并计算 `C`

`CorrectabilityEstimator` 对 Teacher 和 Student 使用相同 sample seed：

```python
teacher_run = continue(history, "teacher", seed)
student_run = continue(history, "student", seed)
```

Verifier 最终只输出成功/失败；Teacher 和 Student 的内部生成仍然可以不同。

### 5.8 第八步：多个 cutoff 聚合为 turn gate

一个 Student turn 可能有多个 Teacher-selected cutoff：

\[
C_t=\frac{1}{|\mathcal C_t|}\sum_{c\in\mathcal C_t}C(h_{t,c}).
\]

代码使用均值，不使用 `max`。因此一个偶然高分 cutoff 不会完全掩盖其他低价值 cutoff。

## 6. 三类核心数据文件

一次 `collect_dataset()` 最重要的输出是：

### 6.1 `trajectories.jsonl`

这是审计数据，每行保存：

```text
domain / task_split / task_id / seed
完整 trunk
每个 Student turn
每个 selected cutoff
cutoff history 和 state hash
qT / qS / correctability
turn correctability
```

当你怀疑 reward、状态恢复或 cutoff 有问题时，首先检查这个文件。

### 6.2 `student_gkd.jsonl`

由 `CorrectabilityCollector.student_rows()` 生成，每行大致是：

```json
{
  "messages": "Student 视角的完整历史与 Student 实际回答",
  "teacher_prompt": "当前用户输入 + privileged resolution steps",
  "correctability": "该 Student turn 的绝对 gate",
  "cutoff_count": "用于计算 turn gate 的 cutoff 数",
  "domain": "任务 domain",
  "task_split": "train",
  "task_id": "任务 ID"
}
```

注意：训练数据保存的是 Student 已经生成过的完整 response。Student 训练时不会重新生成另一条 response。

### 6.3 `buyer_grpo.jsonl`

由 `CorrectabilityCollector.buyer_row()` 生成，每行保存：

```text
Buyer 视角的初始 messages
domain / task_split / task_id
τ² initial history
可用的用户工具 schema
```

它不是一条离线 Buyer trajectory。真正的 Buyer action 在 GRPO phase 中由 rollout server 在线生成。

## 7. Student 如何使用 correctability 学习

Student 数据流是：

```text
student_gkd.jsonl
  → CorrectabilityGKDTrainer.training_step()
  → 使用已收集的 Student response
  → Teacher API 返回该 response 上的 top-k token 分布
  → 每个样本单独计算 GKD/JSD loss
  → 乘该样本的 correctability
  → batch mean
```

核心实现位于
[`coevo/training/gated_gkd.py`](correctability_coevolution/coevo/training/gated_gkd.py)。

### 7.1 为什么必须逐样本计算

目标是：

\[
\mathcal L_S=\frac{1}{B}\sum_i C_i\mathcal L_i^{\mathrm{OPD}}.
\]

如果先把整个 batch 的 token loss 混在一起，就无法准确给不同样本应用不同 gate。

因此 `_compute_jsd_loss()` 会逐样本切片：

```python
for index in range(batch_size):
    sample_loss = super()._compute_jsd_loss(...)
    losses.append(sample_loss)
return gated_example_mean(losses, gates)
```

[`coevo/training/gates.py`](correctability_coevolution/coevo/training/gates.py)
最终执行：

```python
mean(loss_i * gate_i)
```

### 7.2 绝对 gate 不能重新归一化

假设一个 batch 中所有状态都不可教：

```text
C = [0.01, 0.02, 0.01]
```

正确行为是整个 batch 只产生很小梯度。

如果把它们归一化为和为 1：

```text
[0.25, 0.50, 0.25]
```

就会凭空恢复完整训练预算，破坏 absolute correctability 的意义。

### 7.3 Teacher privilege 如何进入训练

Teacher 的输入包含：

```text
当前用户输入
+ privileged resolution steps
+ Student 实际生成的 response prefix
```

Student 的输入不包含 privileged steps。Teacher 只是对 Student 实际访问到的 token prefix 提供分布监督。

### 7.4 当前代码中的实际 loss

研究文档使用 Teacher-to-Student KL 表达 OPD 目标。当前 Swift baseline 实际使用：

- Teacher API top-20 logits；
- GKD/JSD loss；
- `beta=0.5`；
- `sft_alpha=0`；
- correctability 逐样本 gate。

因此论文公式表达的是方法目标，当前代码是一个 top-k JSD/GKD 工程实现。

## 8. Buyer 如何在线学习寻找弱点

Buyer 训练的核心是
[`coevo/training/buyer_scheduler.py`](correctability_coevolution/coevo/training/buyer_scheduler.py)
中的 `Tau2BuyerScheduler`。

### 8.1 为什么 Buyer 必须在线 rollout

Student 更新后，旧状态的 `qS` 和 `C` 可能已经过期。Buyer reward 必须在当前冻结的 `Student_{k+1}` 下重新计算。

因此 Buyer GRPO 不是读取固定 reward，而是：

```text
Buyer rollout server 生成动作
  → 当前 Student service 回答
  → 当前 Teacher/Student 从 cutoff 分叉
  → 当前 verifier 重新计算 C
  → 将 trajectory reward 返回 GRPO Trainer
```

### 8.2 每轮 Buyer action 的执行过程

`Tau2BuyerScheduler.run()` 的核心循环可以写成：

```python
for buyer_turn in range(max_turns):
    response = generate_buyer_action()
    save_buyer_token_ids_and_mask()

    if response_is_truncated:
        mark_invalid()
        break

    reward_info, finished = apply_buyer_action_to_tau2()

    if finished:
        break

    append_student_observation_for_next_buyer_turn()
```

### 8.3 Buyer 文本动作与工具动作

Buyer 生成文本时：

```text
Buyer message
  → environment.advance_student()
  → Student 可能调用若干工具
  → 直到 Student 对 Buyer 给出一个文本回答
  → 对该 Student turn 计算 correctability
```

Buyer 生成用户工具调用时：

```text
Buyer tool call
  → environment.execute_user_tools()
  → 恢复历史数据库状态
  → 执行用户工具
  → 把 tool observation 返回给 Buyer
```

### 8.4 Buyer 视角中的角色反转

[`coevo/rollout/views.py`](correctability_coevolution/coevo/rollout/views.py)
定义了两个不同视角。

Student 视角：

```text
τ² Buyer/User       → role=user
τ² Student/Assistant → role=assistant
```

Buyer 模型视角：

```text
τ² Buyer/User        → role=assistant
τ² Student/Assistant → role=user
```

这是因为在 Swift GRPO 中，正在训练的 policy 必须是 `assistant`。语义上它仍然是 τ² User/Buyer。

### 8.5 Buyer loss mask

每个 Buyer generation 的 token：

```text
response_loss_mask = 1
```

以下内容只作为 observation，不进入 Buyer loss：

- Student 回复；
- Teacher continuation；
- Student continuation；
- tool observation；
- environment 文本。

所以 GRPO 更新的确实是 Buyer policy，而不是 Student 或环境。

### 8.6 Buyer reward

每个有效 Student turn 得到 `C_t`，trajectory reward 是：

\[
R_B=G_{\mathrm{valid}}\cdot\frac{1}{T}\sum_t C_t.
\]

实现位于
[`coevo/rewards/buyer.py`](correctability_coevolution/coevo/rewards/buyer.py)。

当前 validity 会处理：

- 工具执行错误；
- 非法或空 Buyer action；
- 无法产生合法 Student text turn；
- Buyer completion 被 token 上限截断；
- 用完最大 turn 数但没有正常结束。

使用 turn mean 而不是 sum，避免 Buyer 仅靠延长对话累积 reward。

## 9. 为什么这会形成 moving curriculum

第 `k` 轮，Buyer 找到一个高价值状态：

\[
C_k(h)=q_T(h)(1-q_{S_k}(h)).
\]

Student 在该状态上接受 gated OPD 后，如果学习有效：

\[
q_{S_{k+1}}(h)>q_{S_k}(h).
\]

因此同一状态的下一轮 reward 自动降低：

\[
C_{k+1}(h)<C_k(h).
\]

Buyer 如果继续重复旧模式，就拿不到原来的高 reward。它必须改变用户动作，将当前 Student 推向新的 teacher-correctable failure。

这个 curriculum 没有手工指定“先学技能 A、再学技能 B”。课程由 Student 当前能力边界自动决定。

### 9.1 真正需要观察的共进化证据

Buyer reward 上升本身不足以证明共进化有效。需要同时观察：

1. 某类状态最初 `C` 高；
2. Student 接受训练后，该类状态的 `qS` 上升；
3. 同类状态的 `C` 下降；
4. Buyer 的高 `C` 状态迁移到新类别；
5. 固定 test split 上的 Student 能力持续提高；
6. 历史 Buyer/task 上没有明显遗忘。

## 10. 六条不能被破坏的核心不变量

### 不变量一：Teacher 和 Student 必须从同一个 prefix 出发

不能让 Teacher 重新生成一条自己的轨迹，再与 Student 结果比较。可纠正性衡量的是 Teacher 能否从 **Student 已经诱导出的状态** 接管。

### 不变量二：Teacher privilege 不能泄漏给 Student

Oracle plan 只能影响 Teacher cutoff ranking、Teacher continuation 和 Teacher token distribution。

### 不变量三：`C` 保留绝对尺度

不在 group/batch 内把 `C` 归一化为总和 1。全组低价值时，正确行为是接近零梯度或跳过。

### 不变量四：Student 不吃任务 RL reward

Verifier reward 只决定在哪里学习。Student 的方向由 Teacher 分布决定。

### 不变量五：Buyer 只对自己的 token 求梯度

Student、Teacher、工具和环境 observation 都必须被 mask 掉。

### 不变量六：训练与最终评测环境分离

训练使用 `train` split 和可学习 Buyer；最终评测使用官方 `test` split 和固定独立 user simulator。

## 11. 当前实现与完整研究方案的边界

理解代码时，需要明确哪些已经实现，哪些仍属于研究计划。

| 设计项 | 当前实现 |
|---|---|
| Correctability | Beta 平滑的 `qT(1-qS)` MVP |
| Robust score | 尚未实现 LCB/UCB 版本 |
| Cutoff | Student 文本 turn 内句子边界 |
| Tool/action cutoff | 尚未成为主实现 |
| Student objective | top-20 Teacher logits 的 gated GKD/JSD |
| Buyer reward | validity × turn-mean correctability |
| Realism | 主要依赖参考策略 KL；还没有完整 realism judge 约束 |
| Degeneration penalty | 尚未完整实现重复/循环等独立 penalty |
| Low-reward group skip | 方案中有，当前主代码未完整实现 |
| 工程链路 | Docker、preflight、单元/集成测试已通过 |
| 科学结论 | 正式全参多轮训练和 H1/H2/H3 尚未完成 |

## 12. 推荐的最短核心阅读顺序

如果目标是看懂核心，而不是逐个文件阅读，按下面顺序：

### 第一组：先理解状态和 reward

1. [`coevo/rollout/collector.py`](correctability_coevolution/coevo/rollout/collector.py)
2. [`coevo/rollout/cutoff_scorer.py`](correctability_coevolution/coevo/rollout/cutoff_scorer.py)
3. [`coevo/cutoff/boundaries.py`](correctability_coevolution/coevo/cutoff/boundaries.py)
4. [`coevo/cutoff/teacher_selector.py`](correctability_coevolution/coevo/cutoff/teacher_selector.py)
5. [`coevo/rollout/prefix_branch.py`](correctability_coevolution/coevo/rollout/prefix_branch.py)
6. [`coevo/rewards/correctability.py`](correctability_coevolution/coevo/rewards/correctability.py)

读完应该能回答：

- trunk 是什么？
- cutoff state 是什么？
- 为什么 Teacher 能看到完整 turn，却只从 partial prefix 分叉？
- 如何保证 Teacher/Student 起点一致？
- `qT`、`qS`、`C` 如何得到？

### 第二组：理解 Student 更新

1. `CorrectabilityCollector.student_rows()`
2. [`coevo/training/gated_gkd.py`](correctability_coevolution/coevo/training/gated_gkd.py)
3. [`coevo/training/gates.py`](correctability_coevolution/coevo/training/gates.py)

读完应该能回答：

- Student 训练使用的是哪条 response？
- Privileged plan 给了谁？
- Gate 是 turn-level 还是 token-level？
- 为什么 loss 必须逐样本计算？

### 第三组：理解 Buyer 更新

1. `CorrectabilityCollector.buyer_row()`
2. [`coevo/training/buyer_scheduler.py`](correctability_coevolution/coevo/training/buyer_scheduler.py)
3. [`coevo/rollout/views.py`](correctability_coevolution/coevo/rollout/views.py)
4. [`coevo/rewards/buyer.py`](correctability_coevolution/coevo/rewards/buyer.py)

读完应该能回答：

- 为什么 Buyer 训练必须在线？
- 为什么 Buyer 在 Swift 中是 assistant？
- 哪些 token 参与 Buyer loss？
- 最后一轮 Buyer action 是否真的进入环境？
- 为什么截断 trajectory 的 reward 是 0？

### 第四组：把两条训练线闭合

1. [`scripts/run_coevolution.py`](correctability_coevolution/scripts/run_coevolution.py)
2. [`coevo/orchestration/collection.py`](correctability_coevolution/coevo/orchestration/collection.py)

读完应该能完整复述：

```text
Student_k/Buyer_k
  → data_k
  → Student_{k+1}
  → online Buyer reward under Student_{k+1}
  → Buyer_{k+1}
  → data_{k+1}
```

## 13. 用测试确认你的理解

测试是最小的可执行例子：

- [`tests/test_core.py`](correctability_coevolution/tests/test_core.py)：公式、cutoff、turn mean 和 gate。
- [`tests/test_buyer_scheduler.py`](correctability_coevolution/tests/test_buyer_scheduler.py)：Buyer 多轮执行、最终动作、mask、截断和 validity。
- [`tests/test_tau2_integration.py`](correctability_coevolution/tests/test_tau2_integration.py)：信息隔离、工具状态恢复、官方 split 和 verifier。

理解完成后，你应该能不看实现，先预测每个测试为什么应该通过。

## 14. 最终伪代码

```python
student = Student_0
buyer = Buyer_0
teacher = FixedPrivilegedTeacher
verifier = FixedTau2Verifier

for round_id in rounds:
    # 1. 用当前 Student/Buyer 收集完整多轮对话
    trunks = collect_tau2_trajectories(student, buyer)

    student_rows = []
    for trunk in trunks:
        for student_turn in completed_text_turns(trunk):
            candidates = semantic_boundaries(student_turn)
            selected = teacher.select_cutoffs(
                history=history_before(student_turn),
                full_student_turn=student_turn,
                oracle_plan=task.oracle_plan,
                candidates=candidates,
            )

            cutoff_scores = []
            for cutoff in selected:
                shared_prefix = truncate_student_turn(trunk, cutoff)

                teacher_outcomes = rollout_to_terminal(
                    shared_prefix, teacher, verifier
                )
                student_outcomes = rollout_to_terminal(
                    shared_prefix, student, verifier
                )

                qT = beta_smoothed_success_rate(teacher_outcomes)
                qS = beta_smoothed_success_rate(student_outcomes)
                cutoff_scores.append(qT * (1 - qS))

            turn_C = mean(cutoff_scores)
            student_rows.append(
                collected_student_response_with_gate(student_turn, turn_C)
            )

    # 2. Student 只接受 Teacher 分布监督
    student = train_student_with_gated_opd(
        rows=student_rows,
        loss=lambda row: row.C * teacher_student_jsd(row),
    )

    # 3. 在更新后的冻结 Student 下在线训练 Buyer
    buyer = train_buyer_with_grpo(
        rollout=lambda buyer_action: run_tau2_and_recompute_correctability(
            buyer_action=buyer_action,
            frozen_student=student,
            fixed_teacher=teacher,
            verifier=verifier,
        ),
        reward=lambda trajectory: (
            trajectory.validity * mean(trajectory.turn_correctability)
        ),
        loss_mask="buyer tokens only",
    )
```

## 15. 判断自己是否真正看懂

如果你可以独立回答下面十个问题，就已经掌握完整核心逻辑：

1. Buyer 为什么必须看到 Student 当前回答后再生成下一轮动作？
2. 为什么 cutoff 必须位于 Student 自己生成的 turn 内？
3. Teacher 为什么既看完整 turn，又从 partial prefix 开始 continuation？
4. Teacher/Student continuation 如何共享任务状态和随机 seed？
5. `qT(1-qS)` 排除了哪两类无教学价值状态？
6. 为什么 `C` 不能在 group 内归一化？
7. Student 为什么只使用 Teacher 分布，而不直接优化 verifier reward？
8. Buyer 的哪些 token 参与 GRPO loss？
9. Student 更新后，为什么必须重新计算 Buyer reward？
10. 什么实验现象才能证明 moving curriculum 真实发生，而不是 Buyer 在 reward hacking？

最核心的记忆句仍然是：

> **环境持续发现固定 Teacher 能纠正、当前 Student 不能解决的状态；Student 通过纯 OPD 消除这些失败，进而自动改变下一轮环境奖励。**
