复现CDF中利用QP避障的算法，并迁移到多指手
1. 复现CDF中的实现方法：
Task setting: n 维（例如 7DoF）的机器人手臂，通过一个 QP 控制器，在避障的同时逐步靠近目标姿态。每个时间步迭代地求解以下QP问题：
$$\begin{aligned}&u^*k = \arg\min{u_k} ; e(q_k)^T H e(q_k) + u_k^T R u_k\\ &\text{s.t.}\quad q_{k+1} = A q_k + B u_k\\ &-\nabla_q f_c(p, q) u_k \Delta t \le \ln(f_c(p, q) + \gamma)\end{aligned}$$
 其中：
- $$e(q_k) = q_k - q_{desire}$$：与目标的误差；
- $$f_c(p,q)$$：CDF，表示“机器人 q 配置下与目标/障碍接触所需的 joint-space 距离”；
- $$\nabla_q f_c(p, q)$$：梯度方向.
- 其中系统模型$$q_{k+1} = A q_k + B u_k$$:
def create_system_matrices(n, dt):
    A_d = np.eye(n)
    B_d = np.eye(n) * dt
    return A_d, B_d
即假设机器人是单积分器系统：$$q_{k+1} = q_k + u_k \Delta t$$
- 目标函数：$$J = (q_{k+1} - q_{goal})^T Q (q_{k+1} - q_{goal}) + u_k^T R u_k$$ 惩罚偏离目标（位置误差）和控制力度；
- 碰撞约束$$\nabla_q f_c(p,q) u_k \Delta t \le \ln(f_c(p,q) + \gamma)$$
当距离 $$f_c$$ 很大（离障碍远）时，右边大 → 约束宽松；当接近障碍时，右边小 → 限制更紧；梯度项强制控制方向避开障碍。
(相当于基于 QP 的 Model Predictive Control (MPC) 框架，每步只求一时刻的最优控制。）
2. 重新formulate QP problem
上面的CDF中的QP问题，已知目标config(而非目标接触）
(1) 这里将目标设为一个确定的目标位姿 $$q_{goal}$$,从而能够将tracking error(与当前q成线性)表达为一个二次型。
(2) 关于避障的限制体现在碰撞约束$$\nabla_q f_c(p,q) u_k \Delta t -\ln(f_c(p,q) + \gamma)\le 0$$中，从而表达为关于控制变量$$u_k$$的线性约束
为了利用$$f_c(p,q)$$同时处理接触和避障，需要重新formulate QP 问题，把原来的 joint-space tracking QP（跟踪 $$q_{\text{goal}}$$）改成一个 task-space contact-distance QP（最小化与目标点集 $$p_{\text{goal}}$$的接触距离），核心区别在于目标函数从 $$|q_{k+1} - q_{\text{goal}}|_Q^2$$变成$$f_c(p{\text{goal}}, q_{k+1})^2$$ 其中 $$f_c(p,q)$$ 是一个可微函数，输出机器人配置$$q$$到目标点$$p$$的接触距离。
但是它是非线性的, 要形成QP，我们对其 一阶线性化：
在当前状态 (q_k) 附近展开：$$f_c(p_{\text{goal}}, q_{k+1})
 \approx f_c(p_{\text{goal}}, q_k)+\nabla_q f_c(p_{\text{goal}}, q_k) (q_{k+1} - q_k)$$
代入$$q_{k+1} = q_k + B u_k \Delta t$$：
$$f_c(p_{\text{goal}}, q_{k+1})\approx f_c(p_{\text{goal}}, q_k)+\nabla_q f_c(p_{\text{goal}}, q_k) B u_k \Delta t$$
于是目标函数可以写为：
$$J(u_k) = \frac{1}{2}\big(f_c(p_{\text{goal}}, q_k)+\nabla_q f_c(p_{\text{goal}}, q_k) B u_k \Delta t \big)^2+|u_k|_R^2$$
展开后是关于 $$u_k$$ 的二次型目标函数，符合QP结构：
$$J(u_k) = \frac{1}{2}u_k^T H u_k + 2 h^T u_k + c$$
其中：
$$\begin{aligned}
 H &= (\Delta t)^2 B^T \nabla_q f_c(p_{goal},q)^T \nabla_q f_c(p_{goal},q) B + 2R \\
&=(\nabla_q f_c(p_{goal},q) B\Delta t)^2+2R\\
 h &= \Delta t B^T \nabla_q f_c(p_{goal},q)^T f_c(p_{\text{goal}}, q_k)\\
c&\text{是与}u_k\text{无关的常数}
 \end{aligned}$$
从而完整的QP 问题：
$$\begin{aligned}
 \min_{u_k} &\quad
 \big(f_c(p_{\text{goal}}, q_k)+\nabla_q f_c(p_{\text{goal}}, q_k) B u_k \Delta t \big)^2+|u_k|_R^2 \\ \text{s.t. } &-\nabla_q f_c(p_{\text{obs}}, q_k) B u_k \Delta t-\ln(f_c(p_{\text{obs}}, q_k) + 1- \gamma) \le 0
 \end{aligned}$$
但是当obs和targ离得很近的时候，obs的排斥力还是会干扰到接近的过程，需要针对任务对$$\gamma$$调参。目前safety_buffer=0.3是能够产生接触的。
Safety_buffer =$$\lambda$$