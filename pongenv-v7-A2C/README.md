# PongEnv v7 (A2C)

本目录包含基于 Advantage Actor-Critic (A2C) 的 Pong 游戏强化学习实现。此版本采用策略梯度方法，同时维护策略网络 (Actor) 和价值网络 (Critic)。

## 核心特点

1.  **Actor-Critic 架构**: 
    *   **Actor (PolicyNet)**: 输出动作的概率分布，直接优化策略。
    *   **Critic (ValueNet)**: 估计当前状态的价值 (V-value)，用于计算优势函数 (Advantage)。
2.  **On-Policy 训练**: 边交互边收集数据，使用 `step_buffer` 缓存当前策略下的轨迹片段进行更新。
3.  **熵正则化**: 虽然代码中未显式强调，但 PolicyNet 输出概率分布，保留了探索性。
4.  **高效并行 (模拟)**: 虽然是单线程环境，但采用了 Batch Update 机制（每 N 步更新一次），模拟了 A2C 的更新节奏。
5.  **HUD 实时显示**: 显示 V 值均值、近期胜率、平均奖励等关键指标。

## 实现原理

*   **状态空间**: 同 v6-DQN，8 维连续向量。
*   **网络结构**: 
    *   `PolicyNet`: Input -> 64 -> 128 -> 64 -> Output (Softmax logits)。
    *   `ValueNet`: Input -> 64 -> 128 -> 64 -> 1 (Scalar V-value)。
*   **损失函数**:
    *   **Critic Loss**: MSE Loss (预测 V 值 vs TD 目标)。
    *   **Actor Loss**: Policy Gradient Loss (log_prob * advantage)。
*   **更新频率**: 每 `batch_size` (默认 32) 步进行一次反向传播更新。

## 代码架构

*   `PongEnv.py`: 游戏环境类。
*   `pong_bot_A2C.py`: 包含 `PolicyNet`, `ValueNet`, `Bot_A2C` 以及主循环。
*   `A2C.pth`: 默认模型保存文件。
*   `history/`: 自动备份目录。

## 快速开始

1.  确保已安装 `pygame`, `numpy`, `torch`, `matplotlib`。
2.  在当前目录下运行：
    ```bash
    python pong_bot_A2C.py
    ```
3.  **交互说明**:
    *   `F2`: 切换控制模式。
    *   `F4`: 手动备份。
    *   训练过程中会实时输出 V-mean 等指标，便于观察收敛情况（V 值通常随训练趋于稳定或上升）。
