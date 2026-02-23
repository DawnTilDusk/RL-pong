# PongEnv v1-5 (Q-learning)

本目录包含基于 Q-learning 的 Pong 游戏强化学习实现。此系列版本展示了从基础的离散状态到引入物理特性的逐步演进过程。

**注意：v1-v2 仅作为早期验证版本，功能相对简单且不成熟。推荐以 v5 版本作为主要入口进行研究与使用。**

## 版本演进与核心特点

### v5 (推荐入口) & v4: 物理特性与策略深化
*   **引入摩擦力 (Mu)**: 在 v4 和 v5 版本中，挡板与球的碰撞引入了摩擦系数 `mu`。这意味着挡板的移动速度会影响球反弹后的 Y 轴速度（切向速度）。
*   **策略性提升**: 这一物理特性的加入，使得智能体不仅仅需要接住球，还可以通过控制击球时的挡板速度（上滑、下滑或静止）来改变球的反射角度，从而学会“扣杀”或“切球”等高级技巧，大大增加了对局的策略深度。
*   **双侧博弈 (v5)**: v5 版本完善了环境逻辑，支持左右互搏或人机对战，进一步提升了训练的对抗性。
*   **改进的训练稳定性 (v5)**: 
    *   **Q表零初始化**: 相比随机初始化，零初始化在初期更稳定。
    *   **奖励裁剪**: 防止异常的高额奖励导致 Q 值发散。
    *   **动态 Epsilon 衰减**: 采用分段衰减策略，平衡探索与利用。
    *   **精确碰撞检测**: 修复了重复触发碰撞奖励的 bug，并引入浮点数物理计算，边界判定更精准。

### v3 (单侧训练)
*   **单侧视角**: 仅针对右侧智能体进行训练，左侧使用简单的规则 Bot。
*   **极简状态**: 状态空间设计较为粗糙（仅包含球的运动方向和与挡板的相对距离），导致决策不够细腻。
*   **无摩擦力**: 碰撞仅改变 X 轴速度方向，物理反馈单一，难以训练出复杂的战术。

## 快速开始 (推荐 v5)

1.  确保已安装依赖环境。
2.  运行 v5 版本：
    ```bash
    python pongenv-v5.py
    ```
3.  **交互说明**:
    *   程序默认运行训练模式。
    *   `F2`: 切换 Bot 类型（例如切换左侧为规则 Bot 或 Q-learning Bot）。
    *   `F4`: 手动备份当前 Q 表（支持按时间戳保存）。
    *   退出时会自动保存 Q 表。

## 代码原理与架构详解

### 通用架构
所有版本均包含 `PongEnv` 环境类和 `Bot_Q` 智能体类。
*   **环境 (PongEnv)**: 处理 Pygame 渲染、物理更新、碰撞检测及分数统计。
*   **智能体 (Bot_Q)**: 维护 Q 表，根据当前状态选择动作，并根据奖励更新 Q 值。

### 各版本实现细节

#### v5 (完整版)
*   **状态定义 (`get_state`)**: 更加精细的 6 维离散状态。
    *   `ball_dx`, `ball_dy`: 球的 X/Y 速度分量（离散化为 5 个等级）。
    *   `ball_x`, `ball_y`: 球的绝对位置（离散化为 10x10 网格）。
    *   `l_y`, `r_y`: 左右挡板的 Y 坐标（离散化为 10 个等级）。
*   **物理引擎**:
    *   碰撞公式：`ball_speed[1] += pad_speed * mu`。挡板速度直接叠加到球的切向速度上。
    *   使用浮点数记录位置，避免整数截断带来的累积误差。
*   **奖励函数**: 综合了过程引导（靠近球）、结果奖励（击球、得分）和惩罚（失分、边界滞留）。

#### v4 (物理引入版)
*   **物理特性**: 首次引入 `mu` 摩擦力参数，代码结构与 v5 类似，但部分训练超参数（如 epsilon 衰减）和 bug 修复不如 v5 完善。
*   **双侧 Q 表支持**: 代码中保留了 `save_q_table_left` 等接口，尝试进行双侧智能体的独立保存。

#### v3 (基础版)
*   **极简状态**: 
    *   `ball_x_dir`, `ball_y_dir`: 仅记录方向（0或1）。
    *   `l_y_diff`, `r_y_diff`: 仅记录球与挡板中心的相对高度差。
*   **局限性**: 由于状态过于简略，智能体无法感知球的具体位置和速度大小，只能做出简单的反应式动作，胜率上限较低（约 35%）。

## 启动与控制

- v3 启动：
  - 在目录下执行 `python pongenv-v3.py` [pongenv-v3.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v3.py#L423-L446)
  - 左侧挡板可通过方向键控制速度（↑/↓ 或 W/S）[pongenv-v3.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v3.py#L435-L441)
  - 不支持手动模式切换；默认训练右侧 Bot_Q
- v4 启动与切换：
  - 在目录下执行 `python pongenv-v4.py`，运行时可切换手动模式 [pongenv-v4.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v4.py#L596-L626)
  - 手动切换：按 F4 切换 manned（手动控制左侧挡板）[pongenv-v4.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v4.py#L612-L615)
  - 保存快照：按 F2 版本化保存 Q 表（含 left）[pongenv-v4.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v4.py#L609-L613)
- v5 启动与切换：
  - 在目录下执行 `python pongenv-v5.py` [pongenv-v5.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v5.py#L700-L717)
  - 手动切换：按 F4 切换 manned（手动控制左侧挡板）[pongenv-v5.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v5.py#L732-L735)
  - 保存快照：按 F2 版本化保存左右侧 Q 表，并在退出时保存训练历史与图表 [pongenv-v5.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v5.py#L722-L727)

## 双 Bot 切换回合设置

- v4：左侧陪练在每 100 回合切换一次（规则 Bot 与左侧 Q Bot）[pongenv-v4.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v4.py#L336-L341)
- v5：左侧陪练在每 1000 回合切换一次（更长的策略适应期）[pongenv-v5.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v5.py#L430-L435)
- 如需调整频率，修改上述文件中的取模参数（`% 100` 或 `% 1000`）。

## 超参数总览（示例）

- 物理/环境：
  - `mu=0.3` 摩擦系数（挡板速度叠加到球的切向速度）[pongenv-v5.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v5.py#L353)
  - `accellerate=0.8`、`decellerate=1.2` 挡板加/减速参数 [pongenv-v5.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v5.py#L351-L353)
  - `pad_maxspeed=10`、`ball_maxspeed=20` 最大速度 [pongenv-v5.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v5.py#L349-L357)
- 智能体（v5 示例）：
  - 左侧 Bot_Q：`alpha=0.3`、`epsilon=0.3` [pongenv-v5.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v5.py#L369-L371)
  - 右侧 Bot_Q：`alpha=0.2`、`epsilon=0.1` [pongenv-v5.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v5.py#L371-L373)
  - Bot_Q 默认：`gamma=0.9`、`decaying=0.99999995`（见类构造参数）[pongenv-v5.py](file:///c:/Users/Notebook/Desktop/work/project_dev/pong/pong-repo/pongenv-v1-5-qlearning/pongenv-v5.py#L38-L48)
