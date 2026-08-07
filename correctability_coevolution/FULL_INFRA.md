# Natural-Decision Environment–Student Co-Evolution

本文说明当前唯一受支持的共进化路径。系统不使用句内 token、字符或语义 cutoff；
所有反事实、数据选择、Student target 和 Buyer reward 都定义在完整的自然 Agent
action 边界上。

## 1. 决策单元

一次决策样本为：

```text
d_t = (s_t, a_t^S)
```

其中 `s_t` 是 Student 生成 action 前的完整 τ² 状态，`a_t^S` 是一个完整
`AssistantMessage`，只能是以下之一：

- 完整文本回复；
- 一次工具调用；
- 一组协议允许的并行工具调用。

`coevo/intervention/decision_state.py` 负责提取该状态，并分别记录 state hash 与
包含 Student action 的 sample hash。

## 2. 单动作 Teacher takeover

`coevo/intervention/teacher_action.py` 在相同 `s_t` 上让 privileged Teacher 生成一个
完整 action `a_t^T`。Teacher 只控制这一个 action，之后立即退出。

`coevo/intervention/action_branch.py` 构造两条分支：

```text
s_t -> a_t^S -> frozen Student continuation
s_t -> a_t^T -> frozen Student continuation
```

两条分支共享 task、初始环境状态、Student checkpoint、冻结 continuation user、
verifier、最大步数和 seed 集合。工具 action 的环境 observation 属于该 action 的状态
转移，observation 后仍由 Student 接管。

## 3. Category-balanced soft score

`coevo/rewards/tau2_soft_score.py` 从 τ² `RewardInfo` 的原子检查构造四类完成度：

- Action；
- Communication；
- Environment（DB exact match 与 environment assertions）；
- NL Assertions。

外层只对任务 `reward_basis` 真正启用的类别取平均。缺失的已启用检查 fail-closed，
大量 action checks 不会压过其他类别。

局部 intervention advantage 为相同 seed 的成对差值均值：

```text
delta_t = mean_m(soft_score_teacher_m - soft_score_student_m)
```

## 4. Structured Buyer 与硬合法性门控

Buyer 只生成 `coevo/models/buyer_plan.py` 定义的 private JSON plan。冻结的
`coevo/models/frozen_renderer.py` 将 plan 渲染为 public user text 或 user tool call。
private diagnosis、target skill 和 predicted gain 不进入 Student、Teacher、continuation
user 或 verifier history。

以下情况将 rollout validity 置零：

- schema、枚举或 payload 不合法；
- 请求不存在的 user tool；
- stop reason 不合法；
- public text 未通过 prompt-injection 检查；
- reveal-hidden-constraint 内容不在 hidden scenario；
- Buyer 自己发起的 user tool transition 失败。

Student 的错误工具、漏确认或错误回答不是 Buyer invalidity，因为它们正是环境需要发现
的能力边界。

## 5. Buyer reward

快速阶段使用 trajectory 内自然决策的 mean intervention advantage。存在固定预算的
shadow OPD 标签时，`buyer_scheduler.py` 使用真实或 critic 预测的 post-OPD gain，并以
validity 作硬门控。

`coevo/rewards/buyer.py` 提供 positive-delta turn credit 与 absolute group skip；当一个
GRPO group 的最佳 reward LCB 仍不大于零时，整组 reward 归零，不从有害数据中制造相对
赢家。

## 6. Student 数据

采集器只保留 `delta_t > 0` 的自然决策。主训练 target 是 Teacher 的完整 corrective
action：

```text
(s_t, a_t^T)
```

每行同时保留原始 Student branch，供 original-branch control 使用。数据中记录
intervention advantage、Student/Teacher continuation value、state/sample hash 与私有
Teacher hint 审计信息。

## 7. Shadow OPD 与 utility critic

- `coevo/training/shadow_opd.py`：固定 optimizer steps、learning rate、token budget、
  LoRA rank 与 batch size；临时 adapter 无论成功或异常都由 context manager 删除。
- `coevo/rewards/utility_critic.py`：用真实 shadow gain 周期性校准快速 utility 预测。
- `coevo/training/buyer_plan_aux.py`：记录 predicted takeover gain、failure type 和
  plan–action consistency 的独立辅助监督。

## 8. 运行配置

关键配置：

```text
COEVO_MAX_INTERVENTION_DECISIONS=0
COEVO_CONTINUATIONS=1
COEVO_BUYER_PLAN_MODE=structured
```

`COEVO_MAX_INTERVENTION_DECISIONS=0` 表示评估轨迹中的全部自然 Student actions。
Buyer GRPO 使用 Swift reward `tau2_buyer_utility`。

运行测试：

```bash
ruff check correctability_coevolution/coevo correctability_coevolution/tests
pytest -q correctability_coevolution/tests
```
