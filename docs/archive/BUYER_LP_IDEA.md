# Stage-Conditioned Environment–Student Co-Evolution for OPD

## 完整 Idea、训练流程与代码实现说明

## 0. 一句话概括

我们学习一个可训练的环境策略 Buyer，使它随着 Student 的更新持续改变交互数据分布，优先生成这样的自然 Agent 决策状态：

> 当前 Student 相比上一个 Student checkpoint，在同一个 skill-conditioned self-Teacher 目标上表现出了正的学习进度。

Student 与 Buyer 使用两个严格分离的训练信号：

- **Student**：蒸馏当前 checkpoint 在 private skill 条件下构造的 sharpened Teacher target；
- **Buyer**：最大化连续两个 Student checkpoint 在同一冻结 Teacher target 上的正 gap decrease。

当前主方法不使用：

- shadow OPD；
- utility critic；
- Teacher/Student paired continuation；
- intervention advantage；
- post-update virtual simulation；
- task reward 加权 Student loss。

---

## 1. 要解决的科学问题

标准 OPD 解决的是：

> 当 Student 已经访问到某个状态时，如何利用 Teacher 分布训练 Student？

但 OPD 数据的价值还取决于 Student 被环境带到了哪里。固定 user simulator 产生的数据可能：

- 太简单，Student 已经掌握；
- 太难，当前训练预算下几乎没有变化；
- 重复覆盖同一类状态；
- 与 Student 最近正在形成的能力无关；
- 虽然任务重要，但当前 Student 暂时无法吸收。

因此，本工作将环境策略本身设为可学习对象：

\[
B_k:\text{task/history/Student behavior}\rightarrow\text{next user action}.
\]

我们的目标不是让 Buyer 单纯制造失败，也不是让 Buyer 最大化终局任务难度，而是让它形成一个随 Student 变化的闭环课程：

\[
\text{Student 最近在哪些区域进步}
\Longrightarrow
\text{Buyer 下一轮更多生成哪些区域的数据}.
\]

核心研究假设是：

> 连续 checkpoint 之间出现正学习进度的状态，可以作为当前 Student 学习边界的廉价代理；让 Buyer 下一轮更多生成这些状态或其邻域，能够提高固定 OPD 预算下的真实能力增益。

这个假设必须通过等预算、独立 held-out 评测验证。Buyer reward 上升本身不能证明 Student 能力提高。

---

## 2. 核心概念与符号

Round \(k\) 开始时：

- Student：\(S_k\)；
- Buyer：\(B_k\)。

Student 完成本轮蒸馏后：

\[
S_k\rightarrow S_{k+1}.
\]

Buyer 训练阶段同时保留：

- previous Student：\(S_k\)；
- current Student：\(S_{k+1}\)。

为避免“current”产生歧义，本文在具体解释中也会写作：

~~~text
old = S_k
new = S_(k+1)
~~~

一次自然决策状态包含：

- \(s\)：完整 assistant action 产生前的 Student-visible state；
- \(a_S\)：真实 session 中 current Student 产生的完整 action；
- \(a_T\)：在相同 \(s\) 上，冻结的 Teacher-anchor checkpoint 带 private skill 生成的完整 Teacher action；collection 时 anchor 是 \(S_k\)，Student 更新后的 Buyer 阶段仍固定为 \(S_k\)；
- \(y_1,\ldots,y_L\)：\(a_T\) 的 target token 序列。

完整 action 可以是：

- 一次完整文本回复；
- 一次完整工具调用；
- 协议允许的一组并行工具调用。

本方法不使用句内 token cutoff 或字符位置作为主数据边界。

---

## 3. 系统组件

### 3.1 Student

Student 是部署侧 Agent，只能看到公开对话、工具状态和 domain policy，看不到 private skill。

Student 只通过 Teacher distribution distillation 更新，不使用 Buyer reward 做 Student-side GRPO。

### 3.2 Skill-conditioned self-Teacher

Teacher 的 assistant policy 不是独立的固定大模型，而是：

\[
\text{Teacher policy}
=
\text{本阶段冻结的 Teacher-anchor checkpoint}
+
\text{private skill}.
\]

也就是说，带 skill 和不带 skill 的 policy：

- 使用相同参数；
- 使用相同 tokenizer；
- 使用相同 chat template；
- 面对相同公开状态；
- 只在 private hint 上不同。

系统可以使用一个独立 hinter 根据 oracle resolution steps、domain policy 和当前历史生成 private hint。但 hinter 只生成私有计划，真正输出 Teacher assistant action 的仍然是本阶段的 Student checkpoint。collection 阶段是 \(S_k\)；完成 \(S_k\to S_{k+1}\) 后训练 Buyer 时仍锚定 \(S_k+\text{skill}\)，不会改成 \(S_{k+1}+\text{skill}\)。

### 3.3 Buyer

Buyer 是可训练的环境/用户策略。它接收：

- hidden user scenario；
- 当前公开交互历史；
- Student 最新行为；
- 可用 user tools；
- 当前环境状态。

Buyer 在线 GRPO 路径输出严格的 private structured plan。FrozenRenderer 将其转换为公开 user action。

private plan 不进入：

- Student context；
- Teacher 的公开历史；
- verifier context；
- τ² ground-truth task。

### 3.4 τ² 环境

τ² 提供：

- task 和 hidden scenario；
- domain policy；
- agent/user tools；
- database 和 environment state；
- transition execution；
- fixed held-out evaluation。

当前 Buyer 主 reward 不使用 τ² terminal success，也不使用 Teacher takeover 后的 terminal continuation difference。

---

## 4. 一轮共进化的严格时序

一轮完整训练遵循：

~~~text
Round k starts with Student S_k and Buyer B_k

1. B_k 与 S_k 收集 Teacher-supervised Student 数据 D_k
2. 使用 D_k 蒸馏 Student：
       S_k -> S_(k+1)
3. 同时冻结并服务两个 checkpoint：
       previous = S_k
       current  = S_(k+1)
4. Buyer B_k 与 current Student S_(k+1) 生成新的 GRPO sessions
5. 在每个自然决策状态上，由冻结的 previous S_k+skill 生成示范与 target
6. 比较无 skill 的 S_k 与 S_(k+1) 对同一 target 的 gap
7. 用 group GRPO 更新 Buyer：
       B_k -> B_(k+1)
8. 下一轮使用：
       B_(k+1) + S_(k+1)
   收集新的 Student 蒸馏数据
~~~

Student 和 Buyer 不在同一个 optimizer step 内同时改变。

Buyer reward 阶段生成的新 probe states 不属于导致 \(S_k\rightarrow S_{k+1}\) 的训练集。因此：

> 正 learning progress 说明该状态位于最近发生能力变化的区域，但不说明该 probe 样本因果上导致了这次变化。

---

## 5. Student 数据收集

### 5.1 生成 Student session

在 Round \(k\) 的 collection 阶段，当前 Buyer endpoint 与 \(S_k\) 交互，产生完整 τ² session。

真实 session 里自由生成 action 的是无 skill Student \(S_k\)。

### 5.2 提取自然决策边界

系统遍历 session 中所有完整 AssistantMessage。对于每个 assistant action，构造：

\[
\text{DecisionState}
=
\left(
\text{history before action},
\text{realized Student action}
\right).
\]

这里的 Student action 主要用于定位自然边界和审计；Student 的监督 target 将由 skill-conditioned Teacher 重新生成。

### 5.3 生成一次 Teacher macro-action

在相同 action 前状态 \(s\) 上，让：

\[
S_k+\text{private skill}
\]

生成一次完整 Teacher action \(a_T\)。

Teacher 在产生第一个合法 AssistantMessage 后停止。主训练路径不继续执行 Teacher branch，也不比较 Teacher/Student continuation。

### 5.4 构造并缓存 Teacher target

系统随后固定：

- Student-visible state；
- private hint；
- Teacher action；
- target token IDs；
- loss mask；
- hinted Teacher distribution；
- same-checkpoint unhinted distribution；
- skill-contrast gate；
- sharpened Teacher distribution；
- state/action/target/checkpoint hashes。

合法数据写入 Student 蒸馏 dataset；token、模板或 support 校验失败的数据保留在 audit trajectory 中，但不进入 Student training rows。

---

## 6. Skill-contrast Teacher target

### 6.1 两个相同 checkpoint 的信息视图

给定冻结状态 \(s\) 和 Teacher action token：

\[
a_T=(y_1,\ldots,y_L),
\]

当前 checkpoint 在每个 target token 位置进行 teacher-forced scoring。

带 skill 的分布：

\[
q_{h,j}(\cdot)
=
p_{S_k}
\left(
\cdot\mid s,h,y_{<j}
\right).
\]

不带 skill 的分布：

\[
p_{0,j}(\cdot)
=
p_{S_k}
\left(
\cdot\mid s,y_{<j}
\right).
\]

二者只有 private skill 不同。

### 6.2 Token-level skill contrast

每个 token 位置定义：

\[
c_j
=
D_{\mathrm{KL}}
\left(
q_{h,j}\|p_{0,j}
\right).
\]

含义：

- \(c_j\) 小：private skill 几乎没有改变当前 token 的判断；
- \(c_j\) 大：private skill 对该 token 的决策分布产生明显影响。

### 6.3 Sharpening gate

\[
g_j
=
\operatorname{clip}
\left(
\frac{c_j-\tau_{\mathrm{low}}}
{\tau_{\mathrm{high}}-\tau_{\mathrm{low}}},
0,1
\right).
\]

根据 gate 调整温度：

\[
T_j
=
1-g_j(1-T_{\min}),
\qquad
0<T_{\min}<1.
\]

最终 target：

\[
\widetilde q_{h,j}
=
\operatorname{softmax}
\left(
\frac{z^h_j}{T_j}
\right).
\]

等价地：

\[
\widetilde q_{h,j}
\propto
q_{h,j}^{1/T_j}.
\]

因此：

- \(g_j=0\Rightarrow T_j=1\)：保留原始 hinted Teacher distribution；
- \(g_j=1\Rightarrow T_j=T_{\min}\)：达到最强配置 sharpening；
- 温度始终为正，显式 token support 的排序和 Teacher argmax 不变；
- gate 构造 target，不作为 Student loss 的额外连续权重。

### 6.4 Sparse support 与 tail

实现使用：

- raw hinted Teacher top-k；
- 实际 target token；
- 一个 aggregate tail bucket。

Student/Teacher KL 在固定显式 support 加 tail bucket 上计算。Teacher support mass 低于配置阈值时 fail closed，避免用覆盖不足的 sparse distribution 构造 target。

---

## 7. Student 训练目标

Student 对普通、无 private skill 的 prompt 做 forward，学习缓存的 sharpened Teacher target：

\[
\mathcal L_{\mathrm{Student}}
=
\frac{1}{N}
\sum_{j\in\text{active target tokens}}
D_{\mathrm{KL}}
\left(
\widetilde q_{h,j}
\|
p_{\mathrm{Student},j}
\right).
\]

所有 Teacher target tensor 均 detached。

以下信号不进入 Student loss：

- Buyer raw reward；
- GRPO advantage；
- checkpoint learning progress；
- terminal task score；
- Teacher continuation score；
- intervention advantage。

当前实现也不根据 learning progress 对 Student row 做连续加权。

---

## 8. “只用 current Student 生成一次 trajectory”的准确含义

假设 Student 已更新完成：

~~~text
previous = S_k
current  = S_(k+1)
~~~

### 8.1 只生成一条自主交互历史

Buyer GRPO session 中，只有 current Student \(S_{k+1}\) 自主生成 assistant action 并改变历史：

~~~text
Buyer action 1
    ↓
S_(k+1) 自由生成 Student action 1
    ↓
Buyer action 2
    ↓
S_(k+1) 自由生成 Student action 2
~~~

previous Student \(S_k\) 不参与自由 rollout，不生成另一条 session。

### 8.2 Teacher action 是单步标签，不是第二条 session

对 current Student session 中的每个自然决策状态 \(s\)，系统额外调用：

\[
S_k+\text{skill}
\]

生成一个完整 Teacher action \(a_T\)。

这是一条监督标签，不是另一条 terminal trajectory，也没有 Teacher takeover continuation。

### 8.3 previous/current 只做 teacher-forced scoring

系统冻结：

- \(s\)；
- \(a_T\)；
- target tokens；
- loss mask；
- \(\widetilde q_h\)。

然后让 previous 和 current Student 在无 skill 条件下对相同 target token 打分。

它们不会选择 action，也不会改变历史，只计算：

> 给定同一个 prompt 和相同 Teacher token prefix，我对下一个固定 Teacher token 的概率是多少？

这样避免了独立 rollout 带来的状态分布混淆。

---

## 9. Buyer learning-progress reward

### 9.1 Previous-skill-anchored Teacher target

Buyer reward 阶段的 Teacher target 来自：

\[
S_k+\text{skill}.
\]

先用 previous checkpoint 的两个信息视图构造一次：

\[
q_h=p_{S_k}(\cdot\mid s,h),
\]

\[
p_0=p_{S_k}(\cdot\mid s),
\]

\[
\widetilde q_h
=
\operatorname{Sharpen}(q_h,p_0).
\]

这个 \(\widetilde q_h\) 随后完全冻结。

current Student 不带 skill，也不重建 Teacher target。这样 \(S_{k+1}\) 的参数变化
只出现在被测 Student 一侧，不会同时移动监督锚点。

### 9.2 两个 checkpoint 对同一 target 的 gap

\[
d_{\mathrm{previous}}(s)
=
\frac1N
\sum_j
D_{\mathrm{KL}}
\left(
\widetilde q_{h,j}
\|
p_{S_k,j}
\right),
\]

\[
d_{\mathrm{current}}(s)
=
\frac1N
\sum_j
D_{\mathrm{KL}}
\left(
\widetilde q_{h,j}
\|
p_{S_{k+1},j}
\right).
\]

定义 raw learning progress：

\[
LP(s)
=
d_{\mathrm{previous}}(s)
-
d_{\mathrm{current}}(s).
\]

解释：

- \(LP>0\)：new Student 比 old Student 更接近冻结的 \(S_k+\text{skill}\) Teacher；
- \(LP=0\)：两个 checkpoint 在该 target 上没有可测差异；
- \(LP<0\)：new Student 在该 target 上反而更远。

单 decision reward 只保留正进度：

\[
r(s)=[LP(s)]_+=\max(LP(s),0).
\]

当前公式没有：

- Teacher-quality multiplier；
- residual-gap gate；
- exploration bonus；
- terminal success shaping。

### 9.3 Trajectory reward

一条 Buyer trajectory 包含 \(N_i\) 个成功 score 的自然 decision：

\[
R_i
=
q_{\mathrm{valid},i}
\frac{1}{N_i}
\sum_{t=1}^{N_i}
[LP_{i,t}]_+.
\]

这里是 decision reward 的算术平均，不按 target token 数额外加权。

---

## 10. Buyer GRPO group

对相同 task、hidden scenario 和初始 prompt，采样 \(G\) 条不同 Buyer trajectories：

\[
\tau^{(1)},\ldots,\tau^{(G)}.
\]

所有 trajectory：

- 只与 current Student 交互；
- 使用相同 previous/current checkpoint pair；
- 使用相同 τ² task；
- 只在 Buyer 采样计划上不同。

得到：

\[
R_1,\ldots,R_G.
\]

训练脚本使用 group reward normalization：

\[
A_i
=
\frac{R_i-\operatorname{mean}(R)}
{\operatorname{std}(R)+\epsilon}.
\]

需要区分四类数值：

1. 单 decision raw \(LP\)；
2. trajectory raw reward \(R_i\)；
3. group mean/std；
4. 最终 GRPO normalized advantage。

因此 raw reward 为 \(10^{-5}\) 量级不必然意味着没有训练信号。是否有有效更新取决于同组 reward 差异、group std、有效样本比例和最终 advantage。

如果整个 group 的 absolute reward 都为零，则保持全零，不允许组内标准化凭空制造相对赢家。

---

## 11. 高 reward 数据如何影响 OPD

当前实现不是：

~~~text
Buyer 产生高 reward state
        ↓
立刻选择这些 state 训练当前 Student
~~~

而是：

~~~text
Buyer 产生高 reward state
        ↓
这些 reward 更新 Buyer
        ↓
Buyer 更倾向生成相似状态
        ↓
下一轮数据分布发生改变
        ↓
下一轮 Student 在新的 Teacher targets 上做 OPD
~~~

因此：

- high-reward probe 是 Buyer 的直接训练信号；
- 它通过 \(B_k\rightarrow B_{k+1}\) 间接改变下一轮 Student 数据；
- 当前 Student loss 不乘 Buyer reward；
- 当前代码没有把 Buyer GRPO 的高 reward probes 直接回流进 Student replay buffer。

如果实验目标变成“本轮高 reward probes 也直接作为 Student 训练数据”，则需要新增明确的数据回流/选择路径。这属于另一种算法，不是当前实现。

---

## 12. 具体例子

假设 Round \(k\) 中：

\[
S_k\rightarrow S_{k+1}.
\]

Buyer 让 current Student 进入以下状态：

> 用户已经完成身份验证，要求取消机票，但尚未明确确认 30 美元取消费用。

current Student 在真实 session 中产生自己的 action。系统取 action 前状态 \(s\)，让：

\[
S_k+\text{skill}
\]

产生 Teacher target：

> 取消将产生 30 美元费用，请确认是否继续。

冻结该 action、token 和 sharpened Teacher distribution 后，得到：

\[
d_{\mathrm{previous}}=0.30,
\qquad
d_{\mathrm{current}}=0.12.
\]

于是：

\[
LP=0.30-0.12=0.18,
\]

\[
r=0.18.
\]

另一个状态上：

\[
d_{\mathrm{previous}}=0.08,
\qquad
d_{\mathrm{current}}=0.11.
\]

则：

\[
LP=-0.03,
\qquad
r=0.
\]

如果该 trajectory 只有这两个有效决策：

\[
R=\frac{0.18+0}{2}=0.09.
\]

该 trajectory 会让 Buyer 更偏向产生这类最近出现能力进步的状态。它本身不会在当前实现中直接加权 Student loss。

以上数字仅用于说明公式，不是实际实验结果。

---

## 13. Validity 与 fail-closed

以下情况 trajectory reward 为零：

- Buyer structured plan 无法解析；
- Buyer 使用非法 user tool；
- plan 与 action 不一致；
- rollout 截断；
- 没有产生自然 Student action；
- previous/current checkpoint 缺失或错误；
- previous/current scoring endpoint 失败；
- target token 与本地 tokenizer 不对齐；
- 实际 target token 不在 scoring support；
- hinted Teacher support coverage 不足；
- Teacher target hash 校验失败；
- 任一 stage scoring error。

Student 自身出现以下错误不属于 Buyer invalidity：

- 答错；
- 漏确认；
- 工具名错误；
- 工具参数错误；
- 提前结束；
- policy violation。

这些正是 Buyer 需要探索的 Student 能力区域。

---

## 14. 代码实现与调用链

### 14.1 Round controller

[correctability_coevolution/scripts/run_coevolution.py](correctability_coevolution/scripts/run_coevolution.py)

负责：

- Round 级 collect → train Student → train Buyer；
- 保存 Student update 前后的 checkpoint；
- 启动 previous/current 两个 Student services；
- 校验 checkpoint 顺序和身份；
- Buyer checkpoint refresh；
- manifest、resume 和失败 rollback。

核心顺序：

~~~text
collect_round.py
    ↓
train_student_full.sh
    ↓
policy_previous = pre-update Student
policy          = post-update Student
    ↓
train_buyer_full.sh
    ↓
refresh Buyer
~~~

### 14.2 Collection

[correctability_coevolution/coevo/orchestration/collection.py](correctability_coevolution/coevo/orchestration/collection.py)

负责：

- 按 task/seed 创建 τ² environment；
- 调用 NaturalDecisionCollector；
- 持久化 trajectory、Student rows 和 Buyer prompt rows；
- resume/fingerprint/schema 校验。

[correctability_coevolution/coevo/rollout/collector.py](correctability_coevolution/coevo/rollout/collector.py)

负责：

- 生成完整 Student session；
- 找到完整 AssistantMessage 边界；
- 为每个状态生成 Teacher target label；
- materialize 并缓存 skill-contrast target；
- 输出 Student schema v4 rows。

### 14.3 Teacher action 与 target

[correctability_coevolution/coevo/intervention/teacher_action.py](correctability_coevolution/coevo/intervention/teacher_action.py)

负责：

- 在冻结 DecisionState 上生成一次 privileged Teacher macro-action；
- 得到第一个合法 AssistantMessage 后立即停止；
- 不运行主训练 continuation。

[correctability_coevolution/coevo/scoring/teacher_target.py](correctability_coevolution/coevo/scoring/teacher_target.py)

负责：

- TeacherTargetRecord schema v2；
- target/action/state/checkpoint hash；
- hinted/unhinted view 一致性；
- TeacherTargetLabeler；
- analysis-only TeacherTargetValidator。

[correctability_coevolution/coevo/scoring/skill_contrast.py](correctability_coevolution/coevo/scoring/skill_contrast.py)

负责：

- token forward skill KL；
- gate；
- temperature sharpening；
- support/tail；
- ordering、entropy 和 support-mass 校验。

[correctability_coevolution/coevo/scoring/stage_gap.py](correctability_coevolution/coevo/scoring/stage_gap.py)

负责：

- previous hinted view；
- previous unhinted view；
- current unhinted view；
- target cache；
- teacher-forced prompt log-prob scoring；
- previous/current gap。

### 14.4 Student training

[correctability_coevolution/coevo/training/gated_gkd.py](correctability_coevolution/coevo/training/gated_gkd.py)

负责：

- 读取 Student dataset schema v4；
- 验证 target/action/tokenizer/hash；
- 对 cached sharpened target 计算 forward KL；
- 确保 learning progress 和 Buyer reward 不进入 Student loss。

### 14.5 Buyer rollout 与 reward

[correctability_coevolution/coevo/training/buyer_scheduler.py](correctability_coevolution/coevo/training/buyer_scheduler.py)

负责：

- 解析 Buyer private structured plan；
- FrozenRenderer 生成 public user action；
- current Student 推进真实 session；
- 提取自然 Student decisions；
- 由 previous checkpoint 生成 \(S_k+\text{skill}\) Teacher target；
- 调用 StageGapScorer；
- 聚合 trajectory reward；
- validity 和 scoring error 记录。

[correctability_coevolution/coevo/rewards/stage_progress.py](correctability_coevolution/coevo/rewards/stage_progress.py)

当前唯一主 reward：

\[
r(x)=\max(d_{\mathrm{previous}}-d_{\mathrm{current}},0).
\]

[correctability_coevolution/coevo/training/swift_plugin.py](correctability_coevolution/coevo/training/swift_plugin.py)

负责：

- 注册 Tau2BuyerScheduler；
- 注册 tau2_stage_learning_progress reward；
- absolute all-zero group skip；
- Buyer group telemetry；
- 注册 Student GKD trainer 和 Buyer GRPO trainer。

[correctability_coevolution/scripts/train_buyer_full.sh](correctability_coevolution/scripts/train_buyer_full.sh)

负责：

- Swift/TRL GRPO 配置；
- 同 prompt 的多 trajectory sampling；
- group reward normalization；
- W&B logging。

---

## 15. 运行时服务拓扑

Buyer training 阶段主要服务为：

| Role | 默认端口 | 含义 |
|---|---:|---|
| policy | 8000 | current unprivileged Student \(S_{k+1}\) |
| policy_previous | 8001 | previous Student \(S_k\) 与 \(S_k+\text{skill}\) Teacher anchor |
| buyer | 8002 | 当前 Buyer endpoint |
| rollout | 8003 | Swift online Buyer rollout |

Teacher action、previous hinted logits 和 previous unhinted logits 都来自
policy_previous 服务。policy 服务只推进真实 session，并对同一冻结 target 做
current unhinted teacher-forced scoring。

---

## 16. Artifact contract

Student dataset 使用 schema v4；Teacher target 使用 schema v2。

每个 TeacherTargetRecord 包含：

- state hash；
- Teacher action hash；
- raw target hash；
- sharpened target hash；
- Teacher-anchor checkpoint ID；
- private hint hash；
- Student-visible messages；
- hinted Teacher messages；
- target token IDs；
- target loss mask；
- hinted sparse logits；
- same-checkpoint unhinted sparse logits；
- support mass；
- skill contrast；
- gate；
- sharpening temperature；
- raw/sharpened entropy。

Round manifest 保存：

- previous Student checkpoint；
- current Student checkpoint；
- Buyer checkpoint；
- tokenizer/revision；
- reward name/formula version；
- dataset fingerprint；
- collection/training phase；
- resume 和 rollback 状态。

merge、resume 或 Buyer phase 遇到 checkpoint、schema、tokenizer、target identity 不一致时应拒绝继续，而不是静默降级。

---

## 17. 当前方法和旧方法的区别

### 17.1 不再使用 shadow OPD

旧思路：

~~~text
为每批候选数据临时训练 LoRA
    ↓
在 probe set 上测 post-update gain
    ↓
把临时增益作为 Buyer reward
~~~

当前方法：

~~~text
保留真实连续 Student checkpoints
    ↓
在同一冻结 target 上比较 gap
    ↓
使用真实 checkpoint learning progress
~~~

优点：

- 不需要每条 Buyer trajectory 进行 temporary optimizer update；
- 计算量更低；
- reward 与真实训练阶段变化相关；
- 避免大量临时 adapter 管理。

代价：

- learning progress 是阶段级、回顾性的；
- 不能直接估计某个新样本如果现在训练会带来多少因果收益。

### 17.2 不再使用 Teacher takeover reward

旧 reward：

\[
V(s_{t+1}^{T})-V(s_{t+1}^{S}).
\]

当前 reward：

\[
D_{\mathrm{KL}}(\widetilde q_h\|p_{\mathrm{previous}})
-
D_{\mathrm{KL}}(\widetilde q_h\|p_{\mathrm{current}}).
\]

Teacher action 仍用于监督 target，但不通过 paired continuation 计算 Buyer reward。

### 17.3 不再使用 utility critic

当前没有：

- shadow labels；
- post-update gain predictor；
- utility critic fallback；
- intervention/utility/hybrid reward mode。

---

## 18. 实验设计

### Stage 0：静态接口与不变量

验证：

- 自然 action boundary 正确；
- text/tool/parallel tool target 序列化一致；
- hinted/unhinted view 只有 private skill 不同；
- no-hint 时 \(c=0,g=0,T=1\)；
- sharpening 不改变 Teacher ordering/argmax；
- target/action/hash 可以稳定复现；
- previous/current 相同时 \(LP=0\)；
- previous/current 对调时 raw LP 符号翻转；
- scoring failure fail closed；
- W&B 使用 offline smoke，不触发模型下载。

### Stage 1：Student target 验证

比较：

- raw hinted Teacher target；
- skill-contrast sharpened target；
- hard Teacher action SFT；
- 无 private skill target。

测量：

- Student loss；
- target support mass；
- skill contrast；
- gate；
- temperature；
- Teacher entropy；
- tool name/argument/confirmation accuracy。

目标是确认 sharpening 真正突出 skill-sensitive token，而不是仅仅让所有 token 过度尖锐。

### Stage 2：冻结 checkpoint pair 的 reward 校准

固定：

- previous Student；
- current Student；
- Buyer candidates；
- task/prefix；
- target construction；
- tokenizer；
- generation seed。

采集大量新 states，记录：

- \(d_{\mathrm{previous}}\)；
- \(d_{\mathrm{current}}\)；
- raw LP；
- positive LP；
- validity；
- group mean/std；
- normalized advantage。

按照 high/medium/low LP 分层，在严格相同 token 预算下进行后续 Student 训练审计。

这里不使用 shadow OPD 作为在线 reward；可以离线进行真实等预算训练，以验证 LP 是否预测下一阶段数据价值。

### Stage 3：单轮 Buyer GRPO Pilot

固定同一个 Student 初始 checkpoint，比较：

1. fixed official user simulator；
2. random structured Buyer；
3. Buyer-SFT；
4. Buyer-SFT + positive-LP GRPO。

每个配置：

- 相同 task split；
- 相同 Student 初始 checkpoint；
- 相同 Teacher target；
- 相同 Student OPD token 数；
- 相同 optimizer steps 和 learning rate；
- 独立 fresh collection；
- 固定 held-out evaluation user。

核心问题：

> GRPO Buyer 在下一轮生成的数据，是否比 fixed/SFT Buyer 数据产生更高的真实 Student gain per OPD token？

### Stage 4：多轮共进化

执行：

~~~text
B_k + S_k collect
    ↓
S_k -> S_(k+1)
    ↓
Buyer stage-progress GRPO
    ↓
B_k -> B_(k+1)
    ↓
next round
~~~

观察：

- old Buyer plans 的 reward 是否随 Student 掌握而衰减；
- Buyer 是否迁移到新的 failure types；
- Student 是否持续提升；
- 数据分布是否坍缩；
- 是否出现旧技能遗忘。

### Stage 5：Held-out benchmark

最终评测必须：

- 使用官方 test split；
- 使用固定独立 user simulator；
- 不使用 learned Buyer 作为 test user；
- 不用 test split 选择 checkpoint；
- 报告多个随机种子和 bootstrap confidence interval。

---

## 19. 主要基线与消融

### 19.1 环境基线

- Fixed official user；
- Random structured Buyer；
- Buyer-SFT；
- Buyer-GRPO with task reward；
- Buyer-GRPO with absolute previous-skill Teacher gap；
- Buyer-GRPO with positive checkpoint LP。

### 19.2 Teacher target 消融

- 无 private skill；
- raw hinted distribution；
- hard Teacher action；
- skill-contrast sharpening；
- 固定全局 temperature sharpening；
- 不同 \(\tau_{\mathrm{low}},\tau_{\mathrm{high}},T_{\min}\)。

### 19.3 Learning-progress 消融

- previous/current 同 checkpoint；
- current/previous 对调；
- previous-skill-anchored Teacher target（主方法）；
- current-skill-anchored Teacher target（moving-target 对照）；
- raw signed LP；
- positive-clipped LP；
- 是否加入 residual-gap gate，作为后续单独消融而不是当前主方法。

### 19.4 数据闭环消融

- 仅用高 reward probes 更新 Buyer；
- 高 reward probes 直接回流 Student；
- Buyer 更新后 fresh collection；
- 复用 Buyer GRPO probes 而不 fresh collect。

该组消融用于判断提升究竟来自：

- Buyer 改变了下一轮数据分布；
- 直接的数据筛选；
- 或仅仅复用了高 reward 样本。

---

## 20. 指标

### 20.1 Student 最终能力

- τ² official test pass@1；
- 各领域 pass@1；
- macro average；
- tool name accuracy；
- tool argument accuracy；
- confirmation behavior；
- DB end state；
- premature termination rate；
- episode steps；
- token cost。

### 20.2 Student 数据效率

- actual held-out gain per OPD token；
- actual gain per optimizer step；
- high/medium/low LP 数据的等预算训练增益；
- Student loss；
- forgetting；
- 不同 action 类型的数据贡献。

### 20.3 Buyer reward

- raw previous gap；
- raw current gap；
- raw signed LP；
- positive LP；
- trajectory reward；
- group mean/std；
- normalized advantage；
- all-zero group fraction；
- invalid fraction；
- scoring-error fraction。

### 20.4 Skill target

- skill contrast mean/quantiles；
- gate mean/quantiles；
- temperature；
- raw/sharpened entropy；
- support mass；
- target token count；
- text/tool target 分布。

### 20.5 环境行为

- schema validity；
- scenario fidelity；
- plan-action consistency；
- illegal user tool rate；
- premature stop；
- prompt injection；
- repeated interaction；
- plan diversity；
- skill/failure-type coverage。

### 20.6 共进化证据

- 同一旧 plan 跨 Student rounds 的 reward 曲线；
- environment-induced state distribution shift；
- Buyer failure-type 分布迁移；
- 已掌握区域的 LP 是否归零；
- 每轮 held-out gain per token；
- 多轮 Student forgetting。

---

## 21. Go / No-Go 标准

### Gate A：Teacher target 正确

- hinted/unhinted token 完全对齐；
- no-hint gate 关闭；
- sharpening 保持 argmax；
- target hash 可复现；
- Student loss 只使用 cached Teacher target。

### Gate B：Learning-progress reward 有效

- previous/current 同 checkpoint 时 reward 为零；
- checkpoint 对调时 raw LP 符号符合预期；
- positive LP 分层与后续真实等预算 gain 存在稳定关系；
- all-zero group 比例可接受；
- reward 不是由 scoring failure 或无效 plan 驱动。

### Gate C：Buyer 真正改变下一轮数据

- 更新后的 Buyer 在 fresh collection 中仍产生高 LP 相关状态；
- 相比 Buyer-SFT/fixed user，数据的实际 gain per token 更高；
- 不是单纯增加 trajectory 长度或 target token 数；
- 数据多样性没有明显坍缩。

### Gate D：最终能力提高

- official held-out Student 指标提高；
- 多 seed 稳定；
- 不是仅 Buyer reward 上升；
- fixed test user 下仍然成立；
- 不以旧技能遗忘换取局部提升。

---

## 22. 主要失败模式

### 22.1 Buyer reward 上升，但 Student 不提升

说明 recent checkpoint LP 不是未来数据价值的可靠代理，或者 Buyer 在利用 moving target。

需要检查：

- high/medium/low LP 的真实等预算 gain；
- previous-skill anchor 与实际 Student update 的对齐；
- task/probe/test alignment；
- Buyer 数据多样性；
- group reward 方差。

### 22.2 Reward 大量为零

可能原因：

- previous/current checkpoint 太接近；
- Student update 太小；
- 当前状态不受本轮训练影响；
- Teacher target 过于接近 previous unhinted Student；
- negative LP 被截断；
- scoring error 被 fail closed；
- 整个 group 的 Buyer 计划过于相似。

诊断时必须同时报告：

- previous gap；
- current gap；
- signed LP；
- positive LP；
- group std；
- invalid/scoring-error fraction。

### 22.3 Buyer 只生成刚刚学会、已经饱和的区域

当前纯 positive LP reward没有显式 residual-gap gate。一个刚被完全学会的状态仍可能在当前轮得到正 reward。

如果它下一轮已经饱和，连续 checkpoint gap decrease 应逐渐消失；但是否足以形成稳定课程需要实验验证。

### 22.4 Previous-skill anchor 与 Student update 不对齐

主方法的 \(\widetilde q_h\) 由冻结的 \(S_k+\text{skill}\) 构造，避免 target 随
current Student 一起移动。但如果 \(S_k\to S_{k+1}\) 的训练数据或 skill 与 Buyer
probe 完全不相关，固定旧 target 的 LP 仍可能不是未来数据价值的好代理。

必须通过以下消融判断结果是否依赖 anchor 选择：

- fixed Teacher target；
- previous-skill-anchored target；
- current-skill-anchored moving target；
- checkpoint swap；
- identical checkpoint；
- target hash 稳定性。

### 22.5 Buyer 记忆 task/scenario

需要：

- 匿名化固定 ID；
- 按 task family/skill 分层划分；
- 未见 scenario 测试；
- 新 Student checkpoint 测试；
- 独立 fixed test user。

---

## 23. 当前实现边界与需要注意的地方

### 23.1 Learning progress 不是因果数据价值

Buyer probes 在 Student update 后生成，因此它们测量的是：

> 本次 Student update 泛化到了哪些新状态？

而不是：

> 哪个 probe 样本导致了本次 Student update？

### 23.2 当前没有直接高 reward 数据回流

Buyer reward 通过更新 Buyer 间接改变下一轮数据，不直接筛选或加权当前 Student training rows。

### 23.3 纯 positive LP 没有 residual 难度约束

当前 reward 只要求：

\[
d_{\mathrm{previous}}>d_{\mathrm{current}}.
\]

它不显式要求：

\[
d_{\mathrm{current}}>0.
\]

因此“仍有多少可学习空间”不是当前 reward 的直接组成部分。

### 23.4 Collection 与 GRPO 的 Buyer action path 目前不完全相同

Buyer GRPO 使用：

~~~text
structured Buyer Planner
    ↓
FrozenRenderer
    ↓
public user action
~~~

但当前 collect_round 路径通过 τ² buyer_reference UserSimulator 调用 Buyer endpoint，没有直接复用 Tau2BuyerScheduler/FrozenRenderer。

因此：

- 下一轮 collection 确实使用更新后的 Buyer service/checkpoint；
- 但 collection 与 GRPO 的 prompting 和 action rendering 路径并非完全相同。

正式实验前应验证这一差异是否会削弱“更新后的 structured Buyer 改变 fresh Student 数据分布”的实验主张。

### 23.5 文档版本

当前主方法以以下内容为准：

- 本文件；
- README.md 当前 stage-conditioned 部分；
- correctability_coevolution/FULL_INFRA.md；
- stage_progress.py 中的实际 reward 公式。

旧 correctability、shadow OPD、utility critic 或带 q_teacher/residual/exploration 的公式不属于当前主实现。

---

## 24. 端到端伪代码

~~~python
student = S_0
buyer = B_0

for round_index in range(num_rounds):
    # ---------------------------------------------------------
    # A. Collect Student training data with the pre-update pair
    # ---------------------------------------------------------
    student_rows = []

    sessions = collect_sessions(
        buyer=buyer,
        student=student,
    )

    for session in sessions:
        for decision_state in natural_assistant_decisions(session):
            teacher_action, private_skill = generate_one_teacher_action(
                checkpoint=student,
                state=decision_state.state_before_action,
                with_skill=True,
            )

            q_h = teacher_forced_distribution(
                checkpoint=student,
                state=decision_state.state_before_action,
                action=teacher_action,
                with_skill=True,
            )

            p_0 = teacher_forced_distribution(
                checkpoint=student,
                state=decision_state.state_before_action,
                action=teacher_action,
                with_skill=False,
            )

            q_tilde = skill_contrast_sharpen(
                hinted=q_h,
                unhinted=p_0,
            )

            student_rows.append(
                freeze_target(
                    state=decision_state.state_before_action,
                    teacher_action=teacher_action,
                    target_distribution=q_tilde,
                    target_tokens=teacher_action.tokens,
                    loss_mask=teacher_action.loss_mask,
                )
            )

    # ---------------------------------------------------------
    # B. Student update
    # ---------------------------------------------------------
    previous_student = freeze(student)

    current_student = train_student(
        initialization=student,
        rows=student_rows,
        loss="mean forward KL to cached q_tilde",
    )

    # ---------------------------------------------------------
    # C. Buyer GRPO with a frozen checkpoint pair
    # ---------------------------------------------------------
    for task_prompt in buyer_training_prompts:
        group_rewards = []
        group_trajectories = []

        for sample_index in range(G):
            # Only current Student autonomously generates this session.
            trajectory = rollout(
                buyer=buyer,
                student=current_student,
                prompt=task_prompt,
            )

            decision_rewards = []
            validity = validate_buyer_trajectory(trajectory)

            for state in natural_decision_states(trajectory):
                teacher_action, private_skill = generate_one_teacher_action(
                    checkpoint=previous_student,
                    state=state,
                    with_skill=True,
                )

                q_h = teacher_forced_distribution(
                    checkpoint=previous_student,
                    state=state,
                    action=teacher_action,
                    with_skill=True,
                )

                p_previous = teacher_forced_distribution(
                    checkpoint=previous_student,
                    state=state,
                    action=teacher_action,
                    with_skill=False,
                )

                q_tilde = freeze(
                    skill_contrast_sharpen(
                        hinted=q_h,
                        unhinted=p_previous,
                    )
                )

                p_current = teacher_forced_distribution(
                    checkpoint=current_student,
                    state=state,
                    action=teacher_action,
                    with_skill=False,
                )

                d_previous = mean_forward_kl(q_tilde, p_previous)
                d_current = mean_forward_kl(q_tilde, p_current)

                learning_progress = d_previous - d_current
                decision_rewards.append(
                    max(learning_progress, 0.0)
                )

            trajectory_reward = (
                validity * mean(decision_rewards)
                if decision_rewards
                else 0.0
            )

            group_trajectories.append(trajectory)
            group_rewards.append(trajectory_reward)

        if max(group_rewards) > 0:
            advantages = group_normalize(group_rewards)
            buyer = grpo_update(
                buyer,
                group_trajectories,
                advantages,
            )

    # ---------------------------------------------------------
    # D. Advance the co-evolution state
    # ---------------------------------------------------------
    student = current_student
    buyer = freeze_updated_buyer(buyer)
~~~

---

## 25. 论文主叙事

标准 OPD 只回答 Student 到达某个状态后如何使用 Teacher distribution，却没有解决环境应该把当前 Student 带到哪里。

我们学习一个随 Student checkpoint 更新的闭环环境策略。对于 current Student
新生成 session 中的每个自然决策状态，我们使用冻结的 previous checkpoint
\(S_k\) 的 skill-conditioned 与 unconditioned views 构造一个 token-level
skill-contrast sharpened self-Teacher target。随后，在完全相同的状态、target
action、token 序列和 loss mask 上，对无 skill 的 \(S_k\) 与 \(S_{k+1}\) 进行
teacher-forced scoring，并以相对 \(S_k+\text{skill}\) 固定示范的 gap 正下降作为
Buyer reward。

Student 与 Buyer 按 round 交替更新：Student 蒸馏 private-skill Teacher target；Buyer 学习生成最近出现 checkpoint-level learning progress 的状态，从而改变下一轮 OPD 数据分布。

### 英文一句话

> The environment learns to generate current-Student states that exhibit positive learning progress across consecutive Student checkpoints under the same frozen skill-conditioned self-Teacher target.

### 中文一句话

> 环境学习生成这样的当前 Student 状态：相对于同一个冻结的 skill-conditioned self-Teacher 目标，新 Student 比旧 Student 表现出正的学习进度。

---

## 26. 最终流程图

~~~text
Buyer B_k + Student S_k
          ↓
收集自然 Student sessions
          ↓
提取完整 assistant decision states
          ↓
S_k + private skill 生成单步 Teacher action
          ↓
hinted / unhinted same-checkpoint logits
          ↓
skill contrast gate + Teacher target sharpening
          ↓
缓存 TeacherTargetRecord
          ↓
Student forward-KL distillation
          ↓
S_k -> S_(k+1)
          ↓
同时冻结 previous=S_k 与 current=S_(k+1)
          ↓
Buyer B_k 与 current Student 采样 G 条新 trajectories
          ↓
每个状态由 previous S_k+skill 构造一个冻结 Teacher target
          ↓
previous/current 无 skill teacher-forced scoring
          ↓
LP = KL(q_tilde || previous) - KL(q_tilde || current)
          ↓
r = max(LP, 0)
          ↓
trajectory validity × mean positive LP
          ↓
group-normalized Buyer GRPO
          ↓
B_k -> B_(k+1)
          ↓
下一轮 B_(k+1) + S_(k+1) 改变 OPD 数据分布
          ↓
固定独立 user simulator 做 held-out Student evaluation
~~~
