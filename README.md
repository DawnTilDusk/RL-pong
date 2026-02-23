# Pong RL Training Repository

这是一个包含多种强化学习算法（Q-learning, DQN, A2C）用于训练 Pong 游戏智能体的仓库。本项目提供了完整的训练环境、算法实现以及可视化工具。

## 整体结构

仓库主要包含三个核心版本的环境与算法实现：

*   **pongenv-v1-5-qlearning**: 基于 Q-table 的 Q-learning 算法实现。适用于理解强化学习基础概念。
*   **pongenv-v6-DQN**: 基于 PyTorch 的 Deep Q-Network (DQN) 实现。引入了神经网络、经验回放等机制，支持连续状态空间。
*   **pongenv-v7-A2C**: 基于 PyTorch 的 Advantage Actor-Critic (A2C) 实现。使用策略网络与价值网络，支持更高效的训练。

## 快速开始

### 1. 安装依赖

请确保已安装 Python 3.13 及以下依赖库（详细列表见 `requirements.txt`）：

```bash
pip install -r requirements.txt
```

主要依赖包括：`pygame`, `numpy`, `torch`, `matplotlib`。

### 2. 运行环境

进入对应版本的目录，运行主脚本即可开始训练或观看演示。

**Q-learning 版本:**
```bash
cd pongenv-v1-5-qlearning
python pongenv-v3.py
```

**DQN 版本:**
```bash
cd pongenv-v6-DQN
python pong_bot_DQN.py
```

**A2C 版本:**
```bash
cd pongenv-v7-A2C
python pong_bot_A2C.py
```

## 版本概览与入口

| 版本 | 算法 | 目录 | 入口脚本 | 核心特点 |
| :--- | :--- | :--- | :--- | :--- |
| **v1-5** | Q-learning | `pongenv-v1-5-qlearning` | `pongenv-v3.py` | 离散状态空间，使用 Q 表存储，包含基础可视化。 |
| **v6** | DQN | `pongenv-v6-DQN` | `pong_bot_DQN.py` | 连续状态空间，神经网络近似 Q 值，支持经验回放与 Target Network，HUD 实时统计。 |
| **v7** | A2C | `pongenv-v7-A2C` | `pong_bot_A2C.py` | Actor-Critic 架构，同时训练策略与价值网络，支持 On-policy 更新。 |

## 保存地址及说明

训练过程中的模型权重、Q 表以及历史统计数据默认保存在各版本目录下的 `history` 文件夹或根目录中。

*   **模型文件**: 通常命名为 `dqn.pth` (DQN), `A2C.pth` (A2C), 或 `q_table-v*.pkl` (Q-learning)。
*   **历史记录**: 
    *   DQN 和 A2C 版本会自动在当前脚本所在目录下的 `history` 文件夹中创建备份（已配置为相对路径）。
    *   备份内容包括：带时间戳的模型权重文件、训练曲线图表等。

## 注意事项

*   运行时请确保当前目录下包含 `bg.png` 背景图片文件，否则环境将使用纯色背景。
*   部分版本支持按键交互（如 `F2` 切换人机/自动模式，`F4` 手动保存快照），具体请参考各版本的 README 说明。
