# 环境–Student 共进化：基于连续 checkpoint Teacher-gap 的 OPD 课程

## 摘要

标准 OPD 回答“Student 到达某个状态后，怎样从 Teacher 分布学习”。本工作研究
另一个问题：环境应该让当前 Student 到达哪些自然决策状态，才能持续形成与当前
学习阶段匹配的数据分布？

方法使用一个结构化 Buyer 作为闭环环境策略。每轮先用 `B_k,S_k` 收集 OPD 数据并
把 Student 更新为 `S_(k+1)`，然后在 Buyer GRPO 中同时冻结 `S_k` 与 `S_(k+1)`。
对于 Buyer 新生成的每个自然 decision，当前 checkpoint 在有/无 private skill 的
两个视图上构造唯一 sharpened Teacher target；连续两个 Student checkpoint 相对该
target 的 gap decrease 定义 stage learning progress。Buyer 学习把下一轮数据分布
移向“最近有正进步”的区域。

## 1. 科学问题

令环境策略为 `mu_phi`，Student 为 `pi_theta`，它们共同诱导状态分布：

```text
(s_t,a_t) ~ d(mu_phi, pi_theta)
```

目标不是给每个样本估计一次虚拟训练后的因果价值，而是让环境适应 Student stage：

```text
S_(k-1) -> S_k 改变 reward landscape
reward landscape -> B_k 改变下一轮数据分布
新数据分布 -> S_k 继续学习
```

因此，Buyer reward 描述“这个新例子是否落在最近的学习区域”，不描述“这个新例子
是否造成了最近的学习”。

## 2. 核心假设

H1：private skill 能在同一 checkpoint 内隔离有意义的 Teacher 信息。若有/无 skill
的 forward KL 较大，则该 skill 确实改变了 target token 分布。

H2：把 skill contrast 转换为逐 token temperature，可以增强受 skill 影响的 Teacher
分布，同时在 contrast 为零时精确退化为 raw hinted distillation。

H3：若同一 sharpened target 下 `S_k` 比 `S_(k-1)` 更接近 Teacher，则该状态属于
最近的 Student learning region。

H4：只奖励 positive learning progress，使已无进步或发生退化的状态在主 reward 中
归零。

H5：多轮后 Buyer 生成的 task/skill/state 分布会随 checkpoint pair 改变，并提升固定
环境、等预算下最终 Student held-out 能力。

## 3. Curriculum example

每个例子：

```text
x = (s, a_T)
```

- `s`：Student 可见的完整 pre-action state；
- `a_T`：private skill 下的一次完整 Teacher action；
- target token 在所有视图中 teacher-forced；
- action 可以是文本、一个工具调用或合法并行工具调用组。

系统不在句中或固定 token 比例切分。自然 action boundary 保证训练 target 与 Agent
协议一致。

## 4. 唯一 Teacher target

当前 checkpoint `S_k` 在完全相同的 state、target action 与 tokenizer 上评分：

```text
q_h,j = p(S_k | s, private skill, a_T,<j)
p_0,j = p(S_k | s,                a_T,<j)
c_j   = KL(q_h,j || p_0,j)
```

定义：

```text
g_j = clip((c_j-tau_low)/(tau_high-tau_low), 0, 1)
T_j = 1-g_j(1-T_min)
q_tilde_h,j = softmax(z_h,j/T_j)
```

这个 gate 只构造 target：

- `g=0`：`T=1`，完全保留 raw `q_h`；
- `g=1`：使用固定的最小 temperature；
- 中间值连续插值；
- 不改变 Teacher 排序与 argmax；
- 不再作为 Student loss 的乘法权重；
- 不直接加入 Buyer reward。

`q_h,p_0,g,T,q_tilde_h` 全部 detach 并缓存。Student 训练和两个 checkpoint gap 必须
复用同一个 `q_tilde_h` hash。

## 5. Stage learning progress

对同一 Student-visible state、同一 target tokens、同一 `q_tilde_h`：

```text
d_previous(x) = KL(q_tilde_h || p(S_(k-1) | s))
d_current(x)  = KL(q_tilde_h || p(S_k     | s))
LP_k(x)       = d_previous(x)-d_current(x)
```

实现使用 Teacher top-k + actual target token + aggregate tail bucket 的 forward KL，
并按 active target-token 数平均。

Buyer 单 decision 与 trajectory reward：

```text
r(x) = max(LP(x),0)
R_i  = q_valid * (1/N_i) * sum_x r(x)
```

同一 task/prompt 采样 `G` 条 Buyer trajectory，并以 `R_1,...,R_G` 做 GRPO group reward；
训练启用 `--scale_rewards group`。raw reward 原样记录，全零 group 不产生相对赢家。

## 6. Teacher target 生成边界

主训练路径在冻结 state 上只生成一次 skill-conditioned `a_T` 和对应 Teacher
distribution，不把 `a_T` 插入环境后继续 rollout，也不使用 terminal Teacher quality。
Teacher takeover continuation 只保留为 debugging/analysis，不进入 Buyer reward。

## 7. Student objective

Student 数据来自 `B_k,S_k` 的 on-policy rollout。每个有效 row 已携带 canonical
`TeacherTargetRecord`，训练目标为：

```text
L_student = (1/L) sum_j KL(q_tilde_h,j || p_student,j)
```

Student 不使用任务 reward 做 GRPO，也不使用 Buyer reward 做 weighting/filtering。这样
能单独检验：性能变化是否来自环境改变了 OPD 数据分布。

## 8. Buyer 与 Renderer

Buyer 输出 private structured plan：

```json
{
  "diagnosis": {
    "failure_type": "missing_confirmation",
    "evidence_turns": [4]
  },
  "target_skill": "policy_compliance",
  "next_move": "ask_about_total_cost",
  "payload": {},
  "predicted_learning_progress": 0.45,
  "stop": false
}
```

Frozen Renderer 将 plan 变成 public user action。private plan 不进入 Student、Teacher、
continuation user 或 verifier context。Schema、scenario fidelity、user tool 与 stop 均为
硬约束；invalid rollout reward 必须为零。

## 9. Round-synchronous alternating optimization

```text
Round k starts with S_k, B_k

A. freeze S_k and collect D_k with B_k
B. construct/cache canonical Teacher targets
C. train Student: S_k -> S_(k+1)
D. serve previous=S_k and current=S_(k+1)
E. Buyer online rollouts generate new examples
F. score one shared target under previous/current
G. GRPO update B_k -> B_(k+1)
H. commit manifest and enter next round
```

Buyer update后的旧高分样本不能直接冒充下一轮数据；下一轮必须由新 Buyer fresh
collection，才能证明环境策略有生成能力。

## 10. 数据划分与评测

官方 test split 只用于最终 Student 评测，不能用于 Buyer reward、target calibration、
checkpoint selection 或训练。

官方 train 应按 task family/skill/template 分为：

- environment-train：共进化；
- calibration：冻结 `tau_low,tau_high,T_min` 与 support threshold；
- audit：检查 reward landscape、collapse 与真实 Student improvement 的一致性。

最终 evaluator 使用固定独立 user simulator。learned Buyer 不作为 test user。

## 11. 关键基线与消融

等预算比较：

- Fixed Buyer；
- current-gap only；
- stage learning progress；
- 无 group normalization；
- raw hinted target；
- fixed-temperature target；
- skill-contrast-gated target。

所有条件固定 Student/Buyer 初始化、task split、optimizer steps、Teacher tokens、环境
交互、随机种子和 evaluator。

## 12. 关键指标

Student：official test pass@1、domain macro、tool/argument error、premature stop、token
cost、多个 seed 稳定性。

Curriculum：`d_previous,d_current,LP` 分布、positive LP 比例、raw/group-normalized reward、
gate/temperature/entropy、target support mass、all-zero group、state/skill diversity 和
scoring failure rate。

共进化证据：同一 Buyer plan 在不同 checkpoint pair 下的 reward 变化、生成 task/skill
分布迁移、单位 OPD token 的 held-out Student 增益。

## 13. Go / No-Go

Gate A：结构化 Buyer 合法，Renderer 不成为 reward-hacking 通道。

Gate B：target construction 满足 zero-contrast identity、entropy/order、token alignment 与
support coverage contract。

Gate C：两 checkpoint 在线评分稳定；swap pair 翻转 raw LP；相同 logits 给 LP 零；
endpoint 失败严格 fail closed。

Gate D：learned Buyer 的下一轮数据在等预算 Student 训练下优于 Fixed Buyer，并且
held-out Student 提升，而不是只有 Buyer reward 上升。

## 一句话版本

环境学习生成这样的 Student 状态：相对同一 skill-sharpened Teacher target，当前
Student 比上一 checkpoint 更接近 Teacher；这些正学习进步区域构成下一轮 OPD 数据课程。
