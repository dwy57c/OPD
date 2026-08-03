# 教师可纠正性驱动的环境—学生共进化

**Correctability-Driven Environment–Student Co-Evolution for On-Policy Distillation**

> **核心命题：环境不断发现教师能够纠正、而当前学生尚未掌握的失败；学生再通过纯 On-Policy Distillation 消除这些失败。**

---

## 摘要

现有 On-Policy Distillation（OPD）主要研究：当学生已经访问到某个状态时，如何利用教师在该学生诱导状态上的 token-level 分布进行监督。然而，标准 OPD 通常默认训练输入和交互状态来自固定环境，因而没有回答两个更基础的问题：

1. 教师在当前学生诱导状态上是否真的能够提供可靠纠正；
2. 训练环境能否主动产生对当前学生最有教学价值的交互状态。

本文提出一个由固定特权教师引导的环境—学生共进化框架。Buyer 不是一次性生成初始 prompt，而是在多轮对话中观察学生当前回答，再决定下一轮用户输入，因此它构成一个 closed-loop environment policy。对于学生生成的每个 turn，我们在同一输出序列中设置多个 cutoff，并从相同学生前缀分别让固定教师和当前学生继续完成任务。借助独立 verifier，估计教师从该状态成功的概率与学生自行恢复的概率，并据此定义状态的**绝对可纠正性**：教师能够完成，而学生当前不能完成。

该信号承担两个彼此一致的角色：

- 对学生，它作为不经组内归一化的 turn-level gate，控制该 turn 上 token-level OPD 的监督强度；
- 对 Buyer，它作为环境奖励，推动环境主动寻找新的、有效的、教师可纠正的学生失败。

学生侧只使用 OPD，不使用任务 reward 做强化学习。Verifier 只负责判断哪些状态值得学习，教师分布负责告诉学生具体应该如何学习。随着学生通过 OPD 掌握某类状态，其自行成功率上升，该类状态的可纠正性奖励自动下降，Buyer 因而被迫转向新的学生能力缺口。该框架的目标不是生成最难的环境，而是生成对当前学生最有教学价值的环境。

---

## 1. 研究出发点

### 1.1 OPD 回答了“怎么学”，但没有回答“去哪里学”

OPD 让学生从自身策略产生 rollout，并在学生实际访问到的 prefix 上获得教师分布监督。相比只在教师轨迹上做离线蒸馏，这种训练方式能够缓解训练—部署状态分布不一致的问题。

但标准 OPD 的状态分布通常仍由固定数据集、固定用户模拟器或固定任务生成过程决定。换言之，OPD 优化的是：

> 在已经到达的学生状态上，如何向教师靠近。

它没有直接优化：

> 应该让学生进入哪些状态，才能最有效地暴露并修复当前能力缺口。

因此，本文将 OPD 从一个固定环境中的学生更新方法，扩展为一个环境分布也可以随学生能力变化的闭环训练系统。

### 1.2 相对权重无法表达“整组都不值得学”

设同一初始条件下有 $K$ 条学生 rollout，教师对第 $i$ 条轨迹给出长度归一化 log-likelihood：

$$
s_i=\frac{1}{N_i}\sum_k \log \pi_T(y_{i,k}\mid x,y_{i,<k}).
$$

一种常见做法是在组内进行 softmax：

$$
w_i=\frac{\exp(s_i/\tau)}{\sum_j\exp(s_j/\tau)}.
$$

该权重能够表达组内相对偏好，却无法保留绝对尺度。例如：

$$
s^{(A)}=[-0.2,-0.3,-0.4],\qquad
s^{(B)}=[-5.2,-5.3,-5.4].
$$

两组只相差一个整体平移，因此得到完全相同的 softmax 权重。然而，A 组可能是教师整体熟悉的状态，B 组则可能是教师整体不熟悉、无法可靠纠正的状态。只要权重在组内被强制归一化为 1，即使整组都没有可靠监督价值，也仍然会获得完整的 OPD 预算。

本文因此区分：

$$
\text{relative preference}\neq\text{absolute correctability}.
$$

相对权重可以回答“组内哪一条更好”，但只有绝对可纠正性才能回答“这一条是否值得学习，以及整组是否应该被跳过”。

### 1.3 训练环境不应只追求“更难”

若 Buyer 只最大化学生失败率，它最容易学到的是制造不可解、矛盾、极端或分布外任务。这样的环境虽然能让学生失败，却无法产生可用监督。

若 Buyer 只最大化教师成功率，它又会偏向教师和学生都能完成的简单任务，缺乏训练价值。

因此，真正需要的区域是：

$$
\text{teacher succeeds}\quad\land\quad\text{student fails}.
$$

也就是：

> 学生暂时不会，但固定教师能够可靠纠正。

这一区域既排除了“学生已经掌握”的简单状态，也排除了“连教师都无法处理”的不可教状态。

---

## 2. 核心设计决策

本工作的第一版采用四个明确决策。

### 2.1 学生只使用 OPD

学生不使用任务 reward 做 GRPO 或其他 policy-gradient 更新。学生唯一的学习信号是固定教师在学生 on-policy prefix 上提供的分布监督：

$$
\text{Student update}=\text{correctability-gated OPD}.
$$

这样可以将论文的因果链保持得足够干净：Buyer 产生高价值状态，教师提供监督，学生通过 OPD 消除相应失败。最终性能提升不再混杂于任务 RL 与蒸馏两条更新通道之间。

### 2.2 Buyer 可以使用 GRPO，但 GRPO 只是环境策略的优化器

Buyer 本身是可训练的生成策略，因此可以使用 GRPO、PPO 或其他策略优化方法更新。但这不改变“学生只用 OPD”的定义：

- Buyer 的 reward 是可纠正性；
- Buyer-GRPO 只是最大化这一环境 reward 的实现方式；
- 学生不接收该 reward，也不做任务强化学习。

### 2.3 教师固定并拥有特权信息

教师记为 $\pi_T^P$，其中上标 $P$ 表示教师可以访问 plan、hint、标准操作流程、隐藏目标或其他部署学生不可见的特权信息。

教师最好固定，因为它在系统中承担可纠正性标尺。如果教师、Buyer 和学生同时快速变化，就很难判断 reward 变化来自学生真正进步、环境变简单，还是教师标尺本身发生漂移。

因此，本文研究的是：

> **fixed privileged teacher 引导的 environment–student co-evolution。**

### 2.4 Verifier 只判断“是否成功”，不直接训练学生

独立环境或 verifier $V$ 输出任务是否完成。它用于估计教师和学生从同一状态继续后的结果，但不直接对学生提供 policy-gradient reward。

其职责分工为：

> Verifier 决定在哪里学，Teacher logits 决定具体学什么。

---

## 3. 整体框架

框架包含四个组件：

- 学生策略：$\pi_\theta$；
- Buyer／环境策略：$\mu_\phi$；
- 固定特权教师：$\pi_T^P$；
- 独立环境或验证器：$V$。

在同一初始条件 $c_g$ 下，Buyer 与学生进行多轮交互。第 $t$ 轮有：

$$
u_{g,i,t}\sim\mu_\phi(\cdot\mid H_{g,i,t}),
$$

$$
a_{g,i,t}\sim\pi_\theta(\cdot\mid H_{g,i,t},u_{g,i,t}),
$$

其中：

- $H_{g,i,t}$ 是第 $t$ 轮之前的完整历史；
- $u_{g,i,t}$ 是 Buyer 生成的用户输入；
- $a_{g,i,t}$ 是学生生成的 assistant 输出。

Buyer 在生成下一轮输入前能够看到学生当前回答，因此它不是一个静态任务生成器，而是一个逐轮响应学生行为的 closed-loop environment transition policy。

该闭环结构很重要。一次性 prompt generator 只能控制任务起点，而 Buyer 可以在交互过程中：

- 根据学生已经暴露的理解偏差追问；
- 改变约束、目标或用户偏好；
- 将对话推进到学生容易犯错但教师能够处理的中间状态；
- 在学生掌握某种模式后转向新的能力缺口。

---

## 4. 多 cutoff 的绝对可纠正性

### 4.1 Student-induced state

学生在第 $t$ 个 turn 的输出为：

$$
a_{g,i,t}=(a_{g,i,t,1},\ldots,a_{g,i,t,L_t}).
$$

在同一输出序列中选择多个 cutoff：

$$
c\in\mathcal C_t,
$$

例如早期、中期和后期三个位置，或 10%、30%、50%、70%、90% 等位置。由此构造学生诱导状态：

$$
h_{g,i,t,c}=
\left(H_{g,i,t},u_{g,i,t},a_{g,i,t,<c}\right).
$$

多 cutoff 的目的不是为每个 token 单独发明一个置信度，而是在同一 turn 中获得多个可干预状态，从而减少只观察最终完整回答所带来的偶然性。

第一版实现中，应尽量把 cutoff 放在合法的语义边界上，例如句子边界、工具调用边界或 action 边界，避免截断在 JSON、代码或函数参数中间。

### 4.2 Teacher–student continuation

从同一个状态 $h=h_{g,i,t,c}$ 出发：

1. 让带有 plan／hint 的固定教师继续完成任务；
2. 让当前学生从相同状态自行继续；
3. 分别采样 $M$ 次；
4. 尽可能共享环境 seed 或外部随机条件；
5. 使用 verifier 判断最终任务是否成功。

记：

$$
Z_m^T(h)\in\{0,1\},\qquad Z_m^S(h)\in\{0,1\},
$$

分别表示第 $m$ 次教师 continuation 和学生 continuation 是否成功。定义：

$$
q_T(h)=\Pr(Z^T=1\mid h,p),
$$

$$
q_S(h)=\Pr(Z^S=1\mid h).
$$

其中：

- $q_T(h)$ 表示教师从该学生诱导状态完成任务的能力；
- $q_S(h)$ 表示学生从该状态自行恢复并完成任务的能力。

### 4.3 核心 reward：教师能纠正，而学生当前不能

最简单的可纠正性定义为：

$$
\boxed{
C(h)=q_T(h)\left[1-q_S(h)\right]
}
$$

它具有直观的边界性质：

| 教师成功率 $q_T$ | 学生成功率 $q_S$ | 教学价值 |
|---:|---:|---|
| 高 | 高 | 低：学生已经掌握 |
| 低 | 低 | 低：教师也无法纠正 |
| 低 | 高 | 接近零：教师不是可靠标尺 |
| 高 | 低 | 最高：教师会、学生不会 |

如果使用严格配对的 continuation，也可以直接估计：

$$
\widehat C_{\mathrm{pair}}(h)
=
\frac{1}{M}\sum_{m=1}^M
\mathbf 1\left[Z_m^T(h)=1\land Z_m^S(h)=0\right].
$$

第一版主方法可以使用边际成功率构造的 $q_T(1-q_S)$，并将 paired event 作为消融。这样定义更简单，也不要求将教师与学生内部采样随机性解释为唯一的联合分布。

### 4.4 小样本下的稳健估计

每个 cutoff 的 continuation 次数有限。可先使用 Beta–Binomial 平滑：

$$
\widehat q_T=
\frac{n_T+\alpha}{M+\alpha+\beta},
\qquad
\widehat q_S=
\frac{n_S+\alpha}{M+\alpha+\beta}.
$$

为了避免一次偶然的教师成功带来过高奖励，可定义稳健可纠正性：

$$
\boxed{
C_{\mathrm{robust}}(h)
=
\operatorname{LCB}(q_T(h))
\left[1-\operatorname{UCB}(q_S(h))\right]
}
$$

这里它被称为 **robust correctability score**，而不是联合事件概率的严格置信下界。它表达的是一个保守决策原则：只有当教师成功率的保守估计仍然较高、学生成功率的保守上界仍然较低时，状态才获得高奖励。

若工程上希望从最小版本开始，可先使用：

$$
C_{\mathrm{MVP}}(h)=\widehat q_T(h)[1-\widehat q_S(h)],
$$

待验证信号有效后再加入置信区间。

### 4.5 Turn-level 聚合

同一个 turn 的多个 cutoff 使用均匀平均：

$$
\boxed{
C_{g,i,t}
=
\frac{1}{|\mathcal C_t|}
\sum_{c\in\mathcal C_t}
C_{\mathrm{robust}}(h_{g,i,t,c})
}
$$

第一版不建议使用 $\max_c C(h_{t,c})$，因为它容易让一个偶然可恢复的 cutoff 掩盖其余位置完全不可纠正的事实。均值更接近“这一整个 turn 产生了多少稳定的 OPD 机会”。

---

## 5. Buyer reward

### 5.1 最小且完整的奖励

Buyer 在第 $t$ 轮的即时奖励直接定义为：

$$
\boxed{r_t^B=C_t.}
$$

固定 episode 长度时，轨迹奖励使用 turn 平均：

$$
\boxed{
R_B(\tau)
=
G_{\mathrm{valid}}(\tau)
\frac{1}{T}\sum_{t=1}^T C_t
-
\lambda_{\mathrm{deg}}P_{\mathrm{deg}}(\tau)
}
$$

其中：

- $G_{\mathrm{valid}}\in\{0,1\}$ 是硬门控，用于排除自相矛盾、无法验证、违反环境规则或本质不可执行的任务；
- $P_{\mathrm{deg}}$ 惩罚重复、循环、无意义延长、格式攻击、任意改变要求等明显刷分行为；
- 对 $C_t$ 取平均而不是求和，可防止 Buyer 仅通过延长对话累积奖励。

若 episode 长度不固定，可使用折扣后的长度归一化形式：

$$
R_B(\tau)
=
G_{\mathrm{valid}}(\tau)
\frac{\sum_{t=1}^T\gamma^{t-1}C_t}
{\sum_{t=1}^T\gamma^{t-1}}
-
\lambda_{\mathrm{deg}}P_{\mathrm{deg}}(\tau).
$$

### 5.2 Realism 作为约束，而不是教学奖励

不建议将 realism judge 与 correctability 直接相乘，例如：

$$
G_{\mathrm{valid}}G_{\mathrm{realism}}C_t.
$$

这样会把“状态是否有教学价值”和“Buyer 是否像真实用户”混为同一个不稳定分数，也会增加 Buyer 同时攻击多个 judge 的空间。

更简单的做法是让 Buyer 受到参考策略约束：

$$
\boxed{
J_B(\phi)
=
\mathbb E_{\tau\sim\mu_\phi}[R_B(\tau)]
-
\beta D_{\mathrm{KL}}(\mu_\phi\|\mu_{\mathrm{ref}})
}
$$

其中 $\mu_{\mathrm{ref}}$ 可以是真实用户数据微调得到的 Buyer、训练前的 Buyer checkpoint，或固定用户模拟器。

分工因此保持清晰：

- $C_t$ 衡量教学价值；
- validity 判断任务是否合法；
- reference KL 维持用户行为真实性；
- degeneration penalty 防止明显刷分。

### 5.3 Buyer 使用 GRPO 时，绝对 reward 不能被抹掉

对于同一初始条件下的 $K$ 条 Buyer–student 轨迹，可以基于完整轨迹 reward 构造 Buyer-GRPO advantage：

$$
A_i^B=
\frac{R_i^B-\operatorname{mean}_jR_j^B}
{\operatorname{std}_jR_j^B+\epsilon}.
$$

这里必须区分：

- $C_t$ 和 $R_i^B$ 是绝对教学价值；
- $A_i^B$ 只是 Buyer 优化器内部使用的相对更新信号。

不能先把每组 $C_t$ 强制归一化为和为 1。否则当整组状态都没有教学价值时，系统仍会从噪声中制造一个较大的正 advantage。

可以加入简单的 group skip 规则：

$$
\max_i R_i^B<\tau_{\mathrm{skip}}
\quad\Longrightarrow\quad
\text{跳过该 Buyer group 或重新采样。}
$$

### 5.4 为什么 reward 不直接使用 OPD KL

不建议令 Buyer 最大化：

$$
D_{\mathrm{KL}}(\pi_T^P\|\pi_\theta)
$$

或：

$$
C_tD_{\mathrm{KL}}(\pi_T^P\|\pi_\theta).
$$

大 KL 只表示教师和学生分布差异大，并不保证教师能够完成任务，也不保证该分歧与任务成功有关。Buyer 可能通过制造严重 OOD、格式异常、表达风格差异或已经崩坏的 prefix 来刷高 KL。

因此，本框架坚持以下分工：

$$
\boxed{\text{Buyer reward}=C_t}
$$

回答“去哪里制造训练状态”；

$$
\boxed{\text{Student gradient}=C_t\mathcal L_{\mathrm{OPD},t}}
$$

回答“到达该状态后具体向教师学习什么”。

---

## 6. 学生训练：纯 correctability-gated OPD

### 6.1 Token-level OPD

对于第 $t$ 个学生 turn，教师在学生实际生成的 token prefix 上给出分布：

$$
\mathcal L_{g,i,t}^{\mathrm{OPD}}
=
\frac{1}{L_t}\sum_{k=1}^{L_t}
D_{\mathrm{KL}}\left(
\pi_T^P(\cdot\mid h_{g,i,t,k},p_{g,i,t})
\;\|\;
\pi_\theta(\cdot\mid h_{g,i,t,k})
\right).
$$

plan／hint 只提供给教师。学生训练和部署时都不访问该特权信息。

### 6.2 Turn-level gate

完整学生目标为：

$$
\boxed{
\mathcal L_S(\theta)
=
\mathbb E_{g,i,t}
\left[
\operatorname{sg}(C_{g,i,t})
\mathcal L_{g,i,t}^{\mathrm{OPD}}
\right]
}
$$

其中 $\operatorname{sg}$ 表示 stop-gradient。$C_{g,i,t}$ 只控制监督强度，不通过学生 loss 反向传播到 reward 估计过程。

该目标保留一个关键性质：

$$
C_{g,i,t}\approx 0
\quad\Longrightarrow\quad
\text{该 turn 的 OPD 梯度接近零。}
$$

因此，绝对可纠正性不能在每个 group 内被重新归一化成总和为 1。整个 group 都不可靠时，正确行为是不给或少给监督，而不是强制从中分配完整训练预算。

### 6.3 为什么不加入学生侧 GRPO

学生侧若同时使用任务 GRPO，会产生两个不同的课程偏好：

- OPD 偏好“学生不会、教师会”的可纠正失败；
- GRPO 更依赖同组学生 rollout 中存在足够的结果差异。

这会让 Buyer reward 同时承担两套不完全一致的目标，也使性能提升难以归因。

本文第一版主动舍弃学生侧 GRPO，使核心科学问题保持为：

> 相比固定环境，主动生成教师可纠正的学生失败，能否提高纯 OPD 的训练效率和最终能力？

---

## 7. 环境—学生共进化机制

在第 $k$ 个训练阶段，固定当前学生 $\pi_{\theta_k}$，Buyer 寻找高可纠正性状态：

$$
\phi_{k+1}
\leftarrow
\arg\max_\phi
\mathbb E_{h\sim d_{\mu_\phi,\pi_{\theta_k}}}
[C_{\theta_k}(h)].
$$

然后固定 Buyer 与教师，学生在这些状态上做 OPD：

$$
\theta_{k+1}
\leftarrow
\arg\min_\theta
\mathbb E_{h\sim d_{\mu_{\phi_{k+1}},\pi_{\theta_k}}}
\left[
C_{\theta_k}(h)
D_{\mathrm{KL}}(\pi_T^P\|\pi_\theta)
\right].
$$

若 OPD 有效，学生在原先高价值状态上的成功率会提高：

$$
q_S^{(k+1)}(h)>q_S^{(k)}(h).
$$

因此：

$$
C_{\theta_{k+1}}(h)
=
q_T(h)[1-q_S^{(k+1)}(h)]
<
C_{\theta_k}(h).
$$

原本能够给 Buyer 带来高 reward 的状态，在学生掌握之后自动失去价值。Buyer 若想继续获得奖励，只能寻找新的学生弱点。

这构成了完整的 moving curriculum：

1. Buyer 发现当前学生的教师可纠正失败；
2. 教师在这些学生 on-policy 状态上提供 token-level 分布；
3. 学生通过 OPD 消除相应失败；
4. 已被掌握状态的 Buyer reward 自动下降；
5. Buyer 转向新的能力缺口。

最简洁的概括是：

> **The environment discovers teacher-correctable student failures, and OPD removes them.**

---

## 8. 交替训练流程

```text
Inputs:
  student πθ
  closed-loop Buyer μφ
  fixed privileged teacher πT^P
  fixed verifier V
  reference Buyer μref

Repeat for training rounds k = 1, 2, ...:

  A. Collect current on-policy interactions
     1. Sample initial conditions.
     2. Let Buyer and student interact for T turns.
     3. Save every student-generated turn and its history.

  B. Estimate absolute correctability
     1. Select several semantic cutoffs in each student turn.
     2. From each cutoff, sample teacher continuations and student continuations.
     3. Use V to estimate qT and qS.
     4. Compute C(h) and aggregate it into turn-level Ct.

  C. Update the student
     1. Query teacher token distributions on student on-policy prefixes.
     2. Optimize Σt stop_grad(Ct) · LOPD,t.
     3. Do not use verifier reward to update the student.

  D. Refresh Buyer data under the updated/frozen student
     1. Collect new Buyer–student trajectories.
     2. Recompute absolute correctability rewards.

  E. Update the Buyer
     1. Compute trajectory-average correctability reward.
     2. Apply validity gate, degeneration penalty and reference KL.
     3. Optimize Buyer with GRPO/PPO.
     4. Skip groups whose absolute rewards are all below threshold.

  F. Evaluate and stabilize
     1. Evaluate the student on a fixed test environment.
     2. Track old Buyer checkpoints or replay tasks.
     3. Monitor validity, repetition, realism and curriculum drift.
```

训练时不建议让 Buyer 与学生在每一个 minibatch 上同时快速变化。更稳妥的做法是：

- 学生更新频率高于 Buyer；
- Buyer 更新前重新采集当前学生下的 reward，避免使用过时可纠正性；
- 教师和 verifier 固定；
- 保存历史 Buyer 或高价值任务，防止环境只追逐当前瞬时弱点。

---

## 9. 论文叙事与创新边界

### 9.1 不应声称的内容

本文不宜声称：

- 第一个研究教师可靠性；
- 第一个使用 cutoff 或 teacher takeover；
- 第一个让训练环境随学生能力变化；
- 第一个根据 teacher–student gap 生成课程；
- 第一个交替训练两个语言模型策略；
- 第一个使用 OPD。

这些组件分别已有相邻研究覆盖。

### 9.2 更可信的贡献组合

本文的贡献应落在以下组合，而不是单个模块：

1. 在逐轮 closed-loop Buyer–student 交互中，对学生生成 turn 构造多个可干预状态；
2. 使用真实 continuation outcome，而不是只依赖 PPL、entropy 或组内相对分数，估计绝对教师可纠正性；
3. 以“教师成功、学生失败”为核心教学区域；
4. 用同一个绝对信号同时作为 Buyer reward 和学生 OPD gate；
5. 学生只通过 OPD 更新，从而形成干净的环境发现—教师监督—学生消除闭环；
6. 当学生掌握已有状态后，reward 自动下降，驱动环境产生新的能力边界。

可以将贡献句写成：

> We train a closed-loop environment policy to seek teacher-correctable student states. Correctability is estimated through teacher–student continuations from multiple cutoffs of student-generated turns and is used both as an absolute OPD gate and as the environment-learning reward. The student is trained solely by OPD, so learning automatically reshapes the environment reward landscape and induces a moving curriculum.

中文版本：

> 我们训练一个逐轮响应学生的闭环环境策略，使其主动寻找教师能够纠正、而当前学生尚未掌握的状态。该绝对可纠正性同时门控学生的 OPD 更新并训练环境策略；随着学生通过纯 OPD 掌握已有状态，其环境奖励自动下降，从而形成持续迁移的教学课程。

---

## 10. 核心科学假设

整篇论文可以被压缩为一个可证伪假设：

$$
\boxed{
\text{更多 valid teacher-correctable failures}
\Longrightarrow
\text{更有效的 OPD 学习}
}
$$

这个假设需要分三步验证。

### H1：可纠正性比相对教师分数更能识别值得学习的状态

固定环境与 Buyer，比较：

- uniform OPD；
- 组内相对教师分数加权；
- 仅使用 $q_T$；
- 使用 $q_T-q_S$；
- 使用 $q_T(1-q_S)$；
- paired teacher-success/student-failure 事件；
- robust correctability。

关键观察是：绝对值接近零时，整个 turn 是否真的应当获得接近零的 OPD 梯度。

### H2：Buyer 能否学会产生更多有效可纠正状态

固定学生，比较：

- 固定环境；
- 只生成初始 prompt 的环境；
- 最大化学生失败率的 Buyer；
- 最大化教师成功率的 Buyer；
- 目标成功率／难度驱动 Buyer；
- correctability-driven closed-loop Buyer。

要证明训练后的 Buyer 提高的是 valid high-$C$ state density，而不是简单提高失败率或制造不可解任务。

### H3：交替训练是否形成真实的 moving curriculum

在共进化阶段，应观察：

1. 某类状态初始具有较高 $C$；
2. 学生在这些状态上接受 gated OPD 后，$q_S$ 上升；
3. 同类状态的 $C$ 随之下降；
4. Buyer 生成的高-$C$ 状态迁移到新的技能或交互区域；
5. 固定真实测试环境上的学生能力持续提升，而不是只适应当前 Buyer。

---

## 11. 最关键的实验与指标

### 11.1 内层信号验证

在相同学生状态上记录：

- 教师 PPL；
- 教师 entropy／confidence；
- 单次 teacher takeover；
- 多 cutoff 教师成功率；
- 学生自行成功率；
- $q_T-q_S$；
- $q_T(1-q_S)$；
- paired correctability；
- robust correctability。

主要评估这些信号与实际 OPD 后性能变化的关系，可报告：

- 对正向学习增益的 AUROC；
- Spearman 相关系数；
- calibration／reliability diagram；
- 不同信号阈值下的 OPD token efficiency。

这里无需把真实 learning progress 做成训练 reward；它只作为离线验证，检验 correctability 是否是合理代理。

### 11.2 Absolute-shift stress test

构造组内相对排序相同、但绝对教师可靠性不同的两组状态。例如一组教师整体高成功，另一组教师整体低成功。验证：

- 相对归一化方法是否给两组近似相同的总监督预算；
- absolute correctability gate 是否能让低可靠组的总 OPD 梯度接近零。

### 11.3 Buyer reward 消融

比较：

$$
1-q_S,
$$

$$
q_T,
$$

$$
q_T-q_S,
$$

$$
q_T(1-q_S),
$$

以及 robust／paired 版本。

这组实验用于回答：Buyer 只追求学生失败、只追求教师成功，和追求教师可纠正失败之间有什么本质差别。

### 11.4 共进化指标

除最终任务成功率外，还应报告：

- 每千个 Buyer token 产生的 high-$C$ turn 数；
- 每个 teacher token 带来的测试性能提升；
- 教师与学生 continuation 的成功率分布；
- high-$C$ 状态在技能类别上的迁移；
- Buyer 任务有效率、矛盾率和不可解比例；
- 重复、循环和平均 episode 长度；
- Buyer 与参考用户策略的 KL；
- 固定测试环境和历史 Buyer 上的学生性能；
- 总 teacher tokens、student tokens、environment steps 与 verifier calls。

所有主要基线应匹配总训练 token 和环境调用预算，避免仅用更多 takeover 计算换取性能。

---

## 12. 风险与限制

### 12.1 教师能够接管，不必然等于学生容易学会

$q_T(1-q_S)$ 直接测量的是教师替换学生后能否完成，而不是一次 OPD 更新后学生一定会进步。这是核心代理假设，而非数学定理。因此必须通过固定环境下的 post-update 实验验证其预测能力。

第一版可以保持简单，不把更复杂的“短干预后交还学生”纳入主方法；但应将它保留为后续扩展或反例分析，用于区分“教师能替代完成”和“教师局部行为真能帮助学生恢复”。

### 12.2 Buyer reward hacking

Buyer 可能利用教师特权信息、verifier 缺陷、固定学生的偶然弱点或不自然表达方式刷高 reward。应至少使用：

- validity hard gate；
- 参考 Buyer KL；
- 重复与循环惩罚；
- 隐藏 verifier 或人工抽检；
- 跨学生 checkpoint 的 Buyer 泛化测试。

### 12.3 Takeover 成本

多 turn、多 cutoff 和多次 continuation 会显著增加采样成本。工程上应优先采用：

- 少量语义 cutoff；
- 原学生 rollout 复用为一个 $q_S$ 样本；
- 置信区间已明显时提前停止采样；
- 只在不确定或高潜力状态上增加 continuation 次数；
- 严格报告 teacher-token 和 verifier-call 开销。

### 12.4 非平稳与遗忘

Buyer 奖励随学生变化是共进化的来源，也会造成非平稳。Buyer 可能只追逐当前瞬时弱点，学生也可能遗忘旧能力。因此需要固定评测集、历史 Buyer population 或 replay buffer 来判断能力是否累积，而不是在少数弱点间循环。

### 12.5 特权教师的信息泄漏

若 plan／hint 直接包含最终答案，教师成功率可能主要反映答案泄漏，而非可蒸馏策略。需要明确 plan 来源，并进行无 plan、打乱 plan、弱化 plan 等消融，确认教师输出确实能形成可迁移监督。

---

## 13. 最小可行研究路线

### 阶段一：验证内层信号

固定 Buyer 与环境，仅研究：

1. 多 cutoff continuation 是否能稳定估计 $q_T$ 与 $q_S$；
2. absolute correctability 是否优于相对教师分数；
3. correctability-gated OPD 是否优于 uniform OPD；
4. 高 $C$ 是否真的预测更大的 OPD 后学习增益。

如果这一阶段不成立，就不应训练 Buyer。

### 阶段二：固定学生训练 Buyer

在一个冻结学生上训练 Buyer，验证它能否：

- 提高 valid high-$C$ state density；
- 相比 failure-only 或 difficulty-only Buyer 产生更多可用监督；
- 保持用户行为真实性；
- 避免不可解、矛盾、重复和循环。

### 阶段三：交替共进化

最后交替更新：

- Student：pure correctability-gated OPD；
- Buyer：correctability reward，使用 GRPO/PPO 优化；
- Teacher：固定；
- Verifier：固定。

此阶段的核心证据不是 Buyer reward 单调上升，而是：旧状态的 $C$ 因学生学会而下降，高-$C$ 状态持续迁移，同时固定测试环境上的学生能力累积提升。

---

## 14. 论文主叙事

标准 OPD 解决的是学生在自身状态分布上如何接受教师监督，但训练环境仍然是固定的。本文进一步让输入和交互状态分布也参与学习：一个 closed-loop Buyer 观察学生当前回答，并主动将后续交互推进到教师能够完成、而学生当前不能完成的区域。我们通过学生 turn 内的多 cutoff teacher–student continuation 估计绝对可纠正性，并使用该信号同时训练 Buyer 和门控学生 OPD。学生不使用任务强化学习；verifier 只负责判断状态是否值得学习，固定特权教师则提供 token-level 学习目标。随着学生掌握已有状态，学生自行成功率上升，相应的可纠正性 reward 自动下降，迫使 Buyer 转向新的能力缺口。由此，环境负责持续发现可纠正失败，OPD 负责持续消除这些失败，二者形成一个由学生能力变化驱动的闭环课程。

---

## 15. 一句话版本

> **环境寻找的不是最难状态，而是学生暂时不会、教师能够可靠纠正的状态。**

英文：

> **The environment seeks not the hardest states, but the states the current student cannot solve and a fixed privileged teacher can reliably correct.**

---

## 16. 暂定标题

首选：

**Correctability-Driven Environment–Student Co-Evolution for On-Policy Distillation**

备选：

- **Learning to Generate Teachable Interactions with On-Policy Distillation**
- **Teacher-Certified Environment Generation for On-Policy Distillation**
- **Beyond Difficulty: Teacher-Correctable Curricula for On-Policy Distillation**

