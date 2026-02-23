# 功能说明（中文）：
# 本文件实现 DQN 智能体训练 Pong 游戏，新增如下统计与可视化功能：
# 1) 实时统计与记录：reward 历史、每回合总奖励、胜负、Q值、epsilon；
# 2) 渲染开启时，窗口左上角实时显示：总回合数、epsilon、近期10回合胜率、近期10回合平均奖励；
# 3) 退出时绘制折线图：Q值收敛程度、epsilon 下降曲线、近期胜率与近期平均奖励变化；
# 4) F2 切换左侧人控/自动，F4 保存带时间戳的模型；退出时自动保存到 dqn.pth（可通过 AUTO_SAVE_ON_EXIT 控制）。
# 统计变量（均在 Bot_DQN 中维护）：
# - reward_history: 最近100个步级奖励（deque）
# - episode_rewards: 每回合总奖励（list）
# - episode_wins: 每回合是否胜利（list, 1/0）
# - q_values_history: 训练过程中采样的 Q 值（list，均值）
# - epsilon_history: epsilon 随时间的记录（list）
# - win_rate_history: 近期10回合胜率（list）
# - avg_reward_history: 近期10回合平均奖励（list）

from PongEnv import PongEnv
import collections
import random
import torch
import torch.nn as nn
import numpy as np
import pygame
import sys
import os
from datetime import datetime
from collections import deque
# matplotlib is optional for plotting on exit
try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    MPL_AVAILABLE = True
except Exception:
    MPL_AVAILABLE = False
    plt = None
    fm = None

AUTO_SAVE_ON_EXIT = True

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class Replay_Buffer:
    def __init__(self, size):
        self.buffer = collections.deque(maxlen = size)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)
    
    def length(self):
        return len(self.buffer)
        

class Bot_DQN:
    def __init__(self, env: PongEnv, side = 1, alpha = 1e-4, gamma = 0.99, epsilon = 0.3, decaying = 0.999995, epsilon_min = 0.00, stats_stride: int = 10):
        self.env = env
        self.side = side
        
        self.input_dim = 8
        self.output_dim = 3
        self.q_main = DQNNetwork(self.input_dim, self.output_dim)
        self.q_target = DQNNetwork(self.input_dim, self.output_dim)
    
        # hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decaying = decaying
        self.epsilon_min = epsilon_min
        
        self.buffer = Replay_Buffer(100000)
        self.warmup = 2000
        self.batch_size = 32
        
        self.optimizer = torch.optim.Adam(self.q_main.parameters(), lr=self.alpha, weight_decay = 1e-5)
        self.state = self.get_state()
        self.last_action = None
        self.training_steps = 0

        # 统计变量
        self.reward_history = deque(maxlen=100)  # 最近100个步级奖励
        self.episode_rewards = []  # 每回合总奖励
        self.episode_wins = []     # 每回合是否胜利（1/0）
        self.q_values_history = [] # 采样的Q值均值（全量保存）
        self.epsilon_history = []  # epsilon历史（全量保存）
        self.win_rate_history = [] # 近期10回合胜率
        self.avg_reward_history = [] # 近期10回合平均奖励

        self.current_episode_reward = 0.0
        self.total_episodes = 0
        # 新增：逐帧相关统计
        self.current_episode_frames = 0
        self.last_q_mean = 0.0
        # 新增：采样步幅与总步计数
        self.stats_stride = stats_stride
        self.total_steps = 0
        # 缓存用于HUD显示
        self.recent_win_rate = 0.0
        self.recent_avg_reward = 0.0

    def get_state(self):
        # speeds normalized to [-1, 1]
        ball_dx = self.env.ball_speed[0] / self.env.ball_maxspeed
        ball_dy = self.env.ball_speed[1] / self.env.ball_maxspeed
        l_dy = self.env.pad_speed[0] / self.env.pad_maxspeed
        r_dy = self.env.pad_speed[1] / self.env.pad_maxspeed
        # positions normalized to [-1, 1]
        ball_x = (self.env.ball_x / self.env.WINDOW_WIDTH) * 2 - 1
        ball_y = (self.env.ball_y / self.env.WINDOW_HEIGHT) * 2 - 1
        l_y = (self.env.pad_y[0] / self.env.WINDOW_HEIGHT) * 2 - 1
        r_y = (self.env.pad_y[1] / self.env.WINDOW_HEIGHT) * 2 - 1
        return (ball_dx, ball_dy, l_dy, r_dy, ball_x, ball_y, l_y, r_y)
        
    def take_action(self, epsilon):
        state = torch.tensor(self.get_state(), dtype=torch.float32)
        with torch.no_grad():
            q_values = self.q_main(state)
        # 记录Q均值用于HUD逐帧显示
        self.last_q_mean = float(q_values.detach().mean().item())
        # 总步计数递增（用于步幅采样）
        self.total_steps += 1
        # epsilon-greedy 选择动作
        if random.random() < epsilon:
            action = random.randint(0, self.output_dim - 1)
        else:
            action = torch.argmax(q_values).item()
        # 更新epsilon（每步更新，但记录按步幅采样）
        epsilon = max(epsilon * self.decaying, self.epsilon_min)
        self.epsilon = epsilon
        # 步幅采样记录：仅在满足步幅时追加到历史
        if self.total_steps % self.stats_stride == 0:
            self.q_values_history.append(self.last_q_mean)
            self.epsilon_history.append(self.epsilon)
        return action

    def L2_reward(self, a, b):
        return (100 - ((a - b) / 100) ** 2) / 5.0

    def calculate_r_v0(self):
        state = self.get_state()
        action = self.last_action
        current_pad_y = self.env.pad_y[self.side]  # 当前 Bot 控制的挡板Y坐标
        current_pad_center = current_pad_y + self.env.pad_height // 2  # 当前挡板中心坐标
        
        # encourage moving towards the ball when ball is moving towards the pad
        if (self.side == 0 and self.env.ball_speed[0] < 0) or (self.side == 1 and self.env.ball_speed[0] > 0):
            # 球向当前挡板飞来：奖励靠近球（L2距离）
            reward = self.L2_reward(current_pad_center, self.env.ball_y)
        else: 
            reward = self.L2_reward(current_pad_center, self.env.WINDOW_HEIGHT // 2)
            
        if current_pad_y < 50:
            reward -= 10  # 惩罚当前挡板贴顶
        elif current_pad_y + self.env.pad_height > self.env.WINDOW_HEIGHT - 50:
            reward -= 10  # 惩罚当前挡板贴底
        
        if action is not None:
            #1 stay punishment and discourage staying when ball is moving away from the pad
            if action == 1 or (self.env.pad_speed[self.side] == 0 and 
                ((self.side == 0 and self.env.ball_speed[0] <= 0) or (self.side == 1 and self.env.ball_speed[0] >= 0))):
                reward -= 5
            # discourage moving away from the ball when ball is moving towards the pad
            if (self.side == 0 and self.env.ball_speed[0] < 0) or (self.side == 1 and self.env.ball_speed[0] > 0):
                if (self.env.ball_y < current_pad_center and action == 0) or (self.env.ball_y > current_pad_center and action == 2):
                    reward -= 5
            
            
        if (self.side == 0 and self.env.l_collision()):
            reward += 200  # 左侧挡板击中球，奖励
        elif (self.side == 1 and self.env.r_collision()):
            reward += 200  # 右侧挡板击中球，奖励
            
        if (self.side == 0 and self.env.r_collision()) or (self.side == 1 and self.env.l_collision()):
            reward -= 10  # 惩罚对方挡板击中球

        if self.side == 0 and self.env.ball_x > self.env.WINDOW_WIDTH - self.env.pad_width:
            reward += 50  # 左侧 Bot 得分（球从右侧出界）
        elif self.side == 1 and self.env.ball_x < self.env.pad_width:
            reward += 50  # 右侧 Bot 得分（球从左侧出界）
        if self.side == 0 and self.env.ball_x < self.env.pad_width:
            reward -= 200  # 左侧 Bot 失分（球从左侧出界）
        elif self.side == 1 and self.env.ball_x > self.env.WINDOW_WIDTH - self.env.pad_width:
            reward -= 200  # 右侧 Bot 失分（球从右侧出界）
        
        # 记录奖励到统计容器
        self.reward_history.append(float(reward))
        self.current_episode_reward += float(reward)
        # 新增：逐帧累计当前回合帧数
        self.current_episode_frames += 1
        return reward
 
    def calculate_r(self):
        """奖励函数版本2 - 修复版本"""
        state = self.get_state()
        action = self.last_action
        current_pad_y = self.env.pad_y[self.side]  
        current_pad_center = current_pad_y + self.env.pad_height // 2  
        reward = 0

        # 过程奖励：球向当前挡板飞来时的引导
        if (self.side == 0 and self.env.ball_speed[0] < 0) or (self.side == 1 and self.env.ball_speed[0] > 0):
            distance = abs(current_pad_center - self.env.ball_y)
            if distance < 50:
                reward += max(0, 3 - (distance / 20))
            else:
                reward -= 0.5
        else: 
            distance_to_mid = abs(current_pad_center - self.env.WINDOW_HEIGHT // 2)
            reward += max(0, 1 - (distance_to_mid / 100))

        # 碰撞奖励：基础奖励+速度加成
        if (self.side == 0 and self.env.l_collision()):
            ball_speed = abs(self.env.ball_speed[0])
            reward += 200 + min(50, ball_speed * 2)
        elif (self.side == 1 and self.env.r_collision()):
            ball_speed = abs(self.env.ball_speed[0])
            reward += 200 + min(50, ball_speed * 2)

        # 无效靠近惩罚
        is_ball_coming = (self.side == 0 and self.env.ball_speed[0] < 0) or (self.side == 1 and self.env.ball_speed[0] > 0)
        is_close_enough = abs(current_pad_center - self.env.ball_y) < 30
        is_missed = (self.side == 0 and self.env.ball_x < self.env.pad_width) or (self.side == 1 and self.env.ball_x > self.env.WINDOW_WIDTH - self.env.pad_width)
        if is_ball_coming and is_close_enough and is_missed:
            reward -= 100

        # 失分惩罚 - 修复bug：确保两侧都有失分惩罚
        if self.side == 0 and self.env.ball_x < self.env.pad_width:
            reward -= 200  # 左侧失分
        elif self.side == 1 and self.env.ball_x > self.env.WINDOW_WIDTH - self.env.pad_width:
            reward -= 200  # 右侧失分 - 这是修复的关键bug
        
        #记录奖励到统计容器
        self.reward_history.append(float(reward))
        self.current_episode_reward += float(reward)
        # 新增：逐帧累计当前回合帧数
        self.current_episode_frames += 1
        return reward
        
               
    def update_Q(self, batch, device="cpu"):
        states, actions, rewards, next_states, dones = batch
        states = torch.tensor(states, dtype=torch.float32)   # [B, 8]
        actions = torch.tensor(actions, dtype=torch.long)     # [B]
        rewards = torch.tensor(rewards, dtype=torch.float32)  # [B]
        next_states = torch.tensor(next_states, dtype=torch.float32) # [B, 8]
        dones = torch.tensor(dones, dtype=torch.float32)    # [B]
        
        q_values = self.q_main(states)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            next_q_max = self.q_target(next_states).max(dim=1).values
            q_target = rewards + (1 - dones) * self.gamma * next_q_max
        # 使用Huber损失增强鲁棒性
        loss = nn.SmoothL1Loss()(q_taken, q_target)
        
        self.optimizer.zero_grad()
        self.training_steps += 1
        loss.backward()
        # 梯度裁剪，防止不稳定更新
        torch.nn.utils.clip_grad_norm_(self.q_main.parameters(), 5.0)
        self.optimizer.step()
        if (self.training_steps+1) % 100 == 0:
            self.q_target.load_state_dict(self.q_main.state_dict())
    
    def save_model(self, path="dqn.pth", timestamp=False, prefix="dqn"):
        save_path = path
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{prefix}_{ts}.pth"
            save_path = base
        torch.save({
            "q_main": self.q_main.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps
        }, save_path)
        print(f"模型已保存: {save_path}")
    
    def load_model(self, path="dqn.pth"):
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu")
            if "q_main" in data:
                self.q_main.load_state_dict(data["q_main"])
            if "q_target" in data:
                self.q_target.load_state_dict(data["q_target"])
            if "optimizer" in data:
                self.optimizer.load_state_dict(data["optimizer"])
            self.epsilon = data.get("epsilon", self.epsilon)
            self.training_steps = data.get("training_steps", self.training_steps)
            print(f"已加载模型: {path}")

    # 统计辅助方法
    def get_recent_win_rate(self, k=100):
        if len(self.episode_wins) == 0:
            return 0.0
        return float(np.mean(self.episode_wins[-k:]))

    def get_recent_win_rate_50(self):
        # 更稳定的50回合胜率
        if len(self.episode_wins) == 0:
            return 0.0
        k = 50 if len(self.episode_wins) >= 50 else len(self.episode_wins)
        return float(np.mean(self.episode_wins[-k:]))

    def get_recent_avg_reward(self, k=10):
        if len(self.episode_rewards) == 0:
            return 0.0
        return float(np.mean(self.episode_rewards[-k:]))

    def update_statistics_on_episode_end(self, left_score, right_score):
        # 胜负判定：基于当前回合球出界方向（在reset前球仍在出界状态）
        if self.side == 1:  # 右侧为训练方
            win = 1 if self.env.ball_x < 0 else 0  # 球从左侧出界 -> 右侧得分
        else:  # 左侧为训练方
            win = 1 if self.env.ball_x > self.env.WINDOW_WIDTH else 0  # 球从右侧出界 -> 左侧得分
        self.episode_wins.append(win)
        self.episode_rewards.append(self.current_episode_reward)
        self.total_episodes += 1
        # 派生统计（逐回合更新）
        wr = self.get_recent_win_rate(100)
        ar = self.get_recent_avg_reward(10)
        self.win_rate_history.append(wr)
        self.avg_reward_history.append(ar)
        # 缓存用于HUD显示
        self.recent_win_rate = wr
        self.recent_avg_reward = ar
        # HUD 文本准备（回合结束时也更新一次）
        self.update_hud()
        # 清零当前回合累计奖励与帧数
        self.current_episode_reward = 0.0
        self.current_episode_frames = 0

    def update_hud(self):
        # 逐帧HUD更新：根据当前统计与环境得分刷新左上角文本
        # 近期胜率与近期平均奖励（逐回合更新后逐帧显示）
        wr = self.recent_win_rate if self.total_episodes > 0 else self.get_recent_win_rate(100)
        ar10 = self.recent_avg_reward if self.total_episodes > 0 else self.get_recent_avg_reward(10)
        # 当前回合逐帧平均奖励
        ar_ep = (self.current_episode_reward / self.current_episode_frames) if self.current_episode_frames > 0 else 0.0
        left_score = self.env.left_score
        right_score = self.env.right_score
        # 新增：Q均值逐帧显示
        q_mean = self.last_q_mean
        self.env.hud_lines = [
            f"Episode: {self.total_episodes}",
            f"epsilon: {self.epsilon:.5f}",
            f"WinRate(100): {wr*100:.1f}%",
            f"AvgReward(10): {ar10:.2f}",
            f"AvgReward(ep): {ar_ep:.2f}",
            f"Q(mean): {q_mean:.3f}",
            f"Score L:{left_score} R:{right_score}",
        ]

    def plot_training_curves(self, save_path=None, show=True):
        # 若未安装matplotlib，跳过绘图
        if not MPL_AVAILABLE:
            print("未安装matplotlib，跳过训练曲线绘制与保存。")
            return
        # 绘制 Q 均值、epsilon、近期胜率、近期平均奖励曲线，并处理中文字体回退
        has_cn = False
        try:
            fm.findfont('SimHei', fallback_to_default=False)
            has_cn = True
        except Exception:
            try:
                fm.findfont('Microsoft YaHei', fallback_to_default=False)
                has_cn = True
            except Exception:
                has_cn = False
        if has_cn:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
            plt.rcParams['axes.unicode_minus'] = False
            titles = ['Q均值收敛趋势', 'epsilon 下降', '近期100回合胜率', '近期10回合平均奖励']
        else:
            titles = ['Mean Q', 'Epsilon', 'Win Rate(100)', 'Avg Reward(10)']
        # 使用2x3布局，并在第5个位置添加超参数表格
        fig = plt.figure(figsize=(12, 8))
        # 子图1：Q值均值（添加顶端真实步数x轴）
        ax1 = plt.subplot(2,3,1)
        x1 = np.arange(1, len(self.q_values_history)+1)
        ax1.plot(x1, self.q_values_history, label='mean Q')
        # 顶端第二x轴显示真实total_steps刻度
        if len(self.q_values_history) > 0:
            ax1_top = ax1.twiny()
            # 设置顶端刻度为稀疏的索引刻度
            tick_idx = np.linspace(0, len(self.q_values_history)-1, num=6, dtype=int)
            ax1_top.set_xticks(tick_idx)
            # 将刻度标签映射为真实的 total_steps（索引×stats_stride）
            stride = max(self.stats_stride, 1)
            tick_labels = [int(idx * stride) for idx in tick_idx]
            ax1_top.set_xticklabels(tick_labels)
            try:
                ax1_top.set_xlim(ax1.get_xlim())
            except Exception:
                pass
            ax1_top.set_xlabel('total steps')
        ax1.set_title(titles[0])
        ax1.set_xlabel('step')
        ax1.set_ylabel('Q')
        ax1.grid(True)
        # 子图2：epsilon（添加顶端真实步数x轴）
        ax2 = plt.subplot(2,3,2)
        x2 = np.arange(1, len(self.epsilon_history)+1)
        ax2.plot(x2, self.epsilon_history, label='epsilon')
        if len(self.epsilon_history) > 0:
            ax2_top = ax2.twiny()
            tick_idx = np.linspace(0, len(self.epsilon_history)-1, num=6, dtype=int)
            ax2_top.set_xticks(tick_idx)
            stride = max(self.stats_stride, 1)
            tick_labels = [int(idx * stride) for idx in tick_idx]
            ax2_top.set_xticklabels(tick_labels)
            try:
                ax2_top.set_xlim(ax2.get_xlim())
            except Exception:
                pass
            ax2_top.set_xlabel('total steps')
        ax2.set_title(titles[1])
        ax2.set_xlabel('step')
        ax2.set_ylabel('epsilon')
        ax2.grid(True)
        # 子图3：近期胜率（放在第4格）
        ax3 = plt.subplot(2,3,4)
        ax3.plot(self.win_rate_history, label='win rate(100)')
        ax3.set_title(titles[2])
        ax3.set_xlabel('episode')
        ax3.set_ylabel('win rate')
        ax3.grid(True)
        # 子图4：近期平均奖励（放在第6格）
        ax4 = plt.subplot(2,3,6)
        ax4.plot(self.avg_reward_history, label='avg reward(10)')
        ax4.set_title(titles[3])
        ax4.set_xlabel('episode')
        ax4.set_ylabel('avg reward')
        ax4.grid(True)
        # 子图5：超参数表格（放在第5格）
        axp = plt.subplot(2,3,5)
        axp.axis('off')
        # 收集超参数
        try:
            buffer_capacity = self.buffer.buffer.maxlen
        except Exception:
            buffer_capacity = 'N/A'
        # 从优化器中获取权重衰减
        try:
            weight_decay = self.optimizer.param_groups[0].get('weight_decay', 0)
        except Exception:
            weight_decay = 'N/A'
        param_rows = [
            ['学习率 alpha', f'{self.alpha}'],
            ['折扣因子 gamma', f'{self.gamma}'],
            ['初始 epsilon', f'{getattr(self, "epsilon_init", self.epsilon)}'],
            ['epsilon 衰减 decaying', f'{self.decaying}'],
            ['epsilon 最小值', f'{getattr(self, "epsilon_min", "N/A")}'],
            ['经验池容量', f'{buffer_capacity}'],
            ['批次大小 batch_size', f'{self.batch_size}'],
            ['warmup', f'{self.warmup}'],
            ['stats_stride', f'{getattr(self, "stats_stride", "N/A")}'],
            ['优化器', 'Adam'],
            ['权重衰减 weight_decay', f'{weight_decay}'],
        ]
        table = axp.table(cellText=param_rows, colLabels=['超参数', '值'], loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.4)
        axp.set_title('当前训练超参数', fontsize=11)
        plt.tight_layout()
        if save_path:
            try:
                plt.savefig(save_path)
                print(f"训练曲线已保存: {save_path}")
            except Exception as e:
                print(f"保存训练曲线失败: {e}")
        if show:
            plt.show()
        else:
            plt.close()

    def save_backup_package(self, show_plot=False):
        # 创建时间戳文件夹，保存模型与训练图表
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        # 集中备份目录（相对路径）
        current_script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.join(current_script_dir, "history")
        try:
            os.makedirs(base_dir, exist_ok=True)
        except Exception as e:
            print(f"创建history目录失败: {e}")
            return None
        folder = os.path.join(base_dir, f"backup_{ts}")
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            print(f"创建备份文件夹失败: {e}")
            return None
        model_path = os.path.join(folder, f"dqn_{ts}.pth")
        plot_path = os.path.join(folder, f"training_plot_{ts}.png")
        # 保存模型（显式路径）
        try:
            self.save_model(model_path)
            print(f"模型已保存: {model_path}")
        except Exception as e:
            print(f"保存模型失败: {e}")
        # 保存图表（若可用）
        if MPL_AVAILABLE:
            self.plot_training_curves(save_path=plot_path, show=show_plot)
        else:
            print("未安装matplotlib，跳过保存训练图表。")
        print(f"备份已生成: {folder}")
        return folder

if __name__ == "__main__":
    env = PongEnv()
    bot_DQN = Bot_DQN(env, stats_stride = 160)
    
    if os.path.exists("dqn.pth"):
        bot_DQN.load_model("dqn.pth")
    else:
        # 如果没有预训练模型，使用Kaiming初始化（适配ReLU）并设置小随机偏置
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
                torch.nn.init.uniform_(m.bias, -0.01, 0.01)

        bot_DQN.q_main.apply(init_weights)
        bot_DQN.q_target.apply(init_weights)
        print("DQN网络已使用Kaiming初始化权重与小随机偏置")

        # 诊断：打印各层权重/偏置统计信息
        def print_layer_stats(net, net_name):
            for layer_name in ['fc1','fc2','fc3','fc4']:
                m = getattr(net, layer_name)
                w = m.weight.detach()
                b = m.bias.detach()
                print(f"{net_name}.{layer_name}: w(mean={w.mean().item():.4g}, std={w.std().item():.4g})  b(mean={b.mean().item():.4g}, std={b.std().item():.4g})")
        print_layer_stats(bot_DQN.q_main, "q_main")
        print_layer_stats(bot_DQN.q_target, "q_target")

        # 诊断：输出分布检查，避免动作偏向
        with torch.no_grad():
            sample_states = torch.randn(1024, bot_DQN.input_dim)
            q_out = bot_DQN.q_main(sample_states)
            per_action_mean = q_out.mean(dim=0)
            print(f"初始化后各动作Q均值: {per_action_mean.numpy()}")
    
    episode_cnt = 0
    last_episode_cnt = -1
    
    while env.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.running = False
                if AUTO_SAVE_ON_EXIT:
                    bot_DQN.save_model("dqn.pth")
                # 退出时自动按时间戳保存模型与图表（不弹窗）
                try:
                    bot_DQN.save_backup_package(show_plot=True)
                except Exception as e:
                    print("退出备份失败:", e)
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F2:
                    env.switch = not env.switch
                if event.key == pygame.K_F4:
                    # 快照：保存模型与训练图表到同一时间戳文件夹，并弹出图像
                    try:
                        bot_DQN.save_backup_package(show_plot=True)
                    except Exception as e:
                        print("F4备份失败:", e)
    
        if env.switch:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                env.update_speed(1, 0)
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                env.update_speed(-1, 0)
            else:
                env.update_speed(0, 0)
    
        s_t = bot_DQN.get_state()
        a_t = bot_DQN.take_action(bot_DQN.epsilon)
        bot_DQN.last_action = a_t
        env.update_speed(a_t-1, bot_DQN.side)
        env.step()
        s_nt = bot_DQN.get_state()
        r_t = bot_DQN.calculate_r()
        done_t = env.check_over
        bot_DQN.buffer.add(s_t, a_t, r_t, s_nt, done_t)
        # 使用逐帧计数控制训练频率，避免依赖training_steps（仅在update_Q内递增）
        if bot_DQN.buffer.length() > bot_DQN.warmup and bot_DQN.current_episode_frames % 2 == 0:
            batch = bot_DQN.buffer.sample(bot_DQN.batch_size)
            bot_DQN.update_Q(batch)
        
        # 每帧更新HUD显示
        bot_DQN.update_hud()
        
        if episode_cnt % 5 == 0 and last_episode_cnt != episode_cnt:
            last_episode_cnt = episode_cnt
            print(f"Episode {episode_cnt:4d}  Steps={bot_DQN.total_steps}  score={env.left_score}:{env.right_score}  WinRate(100)={bot_DQN.get_recent_win_rate(100)*100:.1f}%  AvgReward(10)={bot_DQN.get_recent_avg_reward(10):.2f}  Qmean={bot_DQN.last_q_mean:.3f}  ε={bot_DQN.epsilon:.5f}")
        
        if done_t: 
            episode_cnt += 1
            # 更新统计与HUD
            bot_DQN.update_statistics_on_episode_end(env.left_score, env.right_score)
            env.reset()
            # 每1000回合自动快照保存（不弹窗显示）
            if episode_cnt % 1000 == 0:
                bot_DQN.save_backup_package(show_plot=False)
    
        env.render()
        
pygame.quit()