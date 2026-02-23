from PongEnv import PongEnv
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from collections import deque
import pygame
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt 
from matplotlib import font_manager as fm

AUTO_SAVE_ON_EXIT = True

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.apply(self.init_weights)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

class ValueNet(nn.Module):
    def __init__(self, input_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.apply(self.init_weights)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    
    
class Bot_A2C:
    def __init__(self, env: PongEnv, side = 1, alpha = 1e-4, batch_size = 32, gamma = 0.99, stats_stride = 10):
        self.env = env
        self.side = side
        
        #hyperparameters
        self.alpha = alpha
        self.gamma = 0.99
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.input_dim = 8
        self.output_dim = 3
        
        self.step_cnt = 1 # 记录当前步数，用于并批
        self.training_steps = 0 # 记录训练步数
        self.step_buffer = deque(maxlen=self.batch_size)
        
        self.state = self.env.get_state()
        
        self.actor = PolicyNet(self.input_dim, self.output_dim)
        self.critic = ValueNet(self.input_dim)        
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=self.alpha)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=self.alpha)
        
        # 统计变量
        self.reward_history = deque(maxlen=100)  # 最近100个步级奖励
        self.episode_rewards = []  # 每回合总奖励
        self.episode_wins = []     # 每回合是否胜利（1/0）
        self.v_mean_history = [] # 采样的V值均值（全量保存）
        self.win_rate_history = [] # 近期10回合胜率
        self.avg_reward_history = [] # 近期10回合平均奖励

        self.current_episode_reward = 0.0
        self.total_episodes = 0
        # 新增：逐帧相关统计
        self.current_episode_frames = 0
        self.last_v_mean = 0.0
        # 新增：采样步幅与总步计数
        self.stats_stride = stats_stride
        self.total_steps = 0
        # 缓存用于HUD显示
        self.recent_win_rate = 0.0
        self.recent_avg_reward = 0.0
        
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        logits = self.actor(state)
        action_dist = torch.distributions.Categorical(logits=logits)
        
        # 统计模块
        self.total_steps += 1
        with torch.no_grad():
            self.last_v_mean = self.critic(state).detach().mean().item()
        if self.total_steps % self.stats_stride == 0:
            self.v_mean_history.append(self.last_v_mean)
        
        return action_dist.sample().item()
    
    def update(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        cur_v = self.critic(states)
        with torch.no_grad():
            next_v = self.critic(next_states)
            td_targets = rewards + self.gamma * next_v * (1 - dones)
        td_delta = td_targets - cur_v
        loss_critic = F.mse_loss(cur_v, td_targets)
        
        logits = self.actor(states)
        log_probs = torch.distributions.Categorical(logits=logits).log_prob(actions).unsqueeze(1)
        loss_actor = torch.mean(-log_probs * td_delta.detach())
        
        self.critic_optim.zero_grad()
        loss_critic.backward()
        self.critic_optim.step()
        
        self.actor_optim.zero_grad()
        loss_actor.backward()
        self.actor_optim.step()
        
        self.training_steps += 1
        
    def calculate_r(self):
        """奖励函数版本2 - 修复版本"""
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

        # 无效靠近惩罚 开始训练阶段不需要这么严格
        """
        is_ball_coming = (self.side == 0 and self.env.ball_speed[0] < 0) or (self.side == 1 and self.env.ball_speed[0] > 0)
        is_close_enough = abs(current_pad_center - self.env.ball_y) < 30
        is_missed = (self.side == 0 and self.env.ball_x < self.env.pad_width) or (self.side == 1 and self.env.ball_x > self.env.WINDOW_WIDTH - self.env.pad_width)
        if is_ball_coming and is_close_enough and is_missed:
            reward -= 100
        """

        # 两侧都有失分惩罚
        if self.side == 0 and self.env.ball_x < self.env.pad_width:
            reward -= 200  # 左侧失分
        elif self.side == 1 and self.env.ball_x > self.env.WINDOW_WIDTH - self.env.pad_width:
            reward -= 200  # 右侧失分 
        
        # 记录奖励到统计容器
        self.reward_history.append(float(reward))
        self.current_episode_reward += float(reward)
        #print(f"current_episode_reward: {self.current_episode_reward}")
        self.current_episode_frames += 1
        
        return reward
    
    def save_model(self, path="A2C.pth", timestamp=False, prefix="A2C"):
        save_path = path
        if timestamp:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"{prefix}_{ts}.pth"
            save_path = base
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optim": self.actor_optim.state_dict(),
            "critic_optim": self.critic_optim.state_dict(),
            "training_steps": self.training_steps
        }, save_path)
        print(f"模型已保存: {save_path}")
    
    def load_model(self, path="A2C.pth"):
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu")
            if "actor" in data:
                self.actor.load_state_dict(data["actor"])
            if "critic" in data:
                self.critic.load_state_dict(data["critic"])
            if "actor_optim" in data:
                self.actor_optim.load_state_dict(data["actor_optim"])
            if "critic_optim" in data:
                self.critic_optim.load_state_dict(data["critic_optim"])
            self.training_steps = data.get("training_steps", self.training_steps)
            print(f"已加载模型: {path}")
    
    def update_statistics_on_episode_end(self, left_score, right_score):
        # 胜负判定：基于当前回合球出界方向（在reset前球仍在出界状态）
        if self.side == 1:  # 右侧为训练方
            win = 1 if self.env.ball_x < 0 else 0  # 球从左侧出界 -> 右侧得分
        else:  # 左侧为训练方
            win = 1 if self.env.ball_x > self.env.WINDOW_WIDTH else 0  # 球从右侧出界 -> 左侧得分
        self.episode_wins.append(win)
        self.episode_rewards.append(float(self.current_episode_reward/self.current_episode_frames))
        self.total_episodes += 1
        # 派生统计（逐回合更新）
        wr = self.get_recent_win_rate(100)
        ar = self.get_recent_avg_reward(1)
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
        
    def get_recent_win_rate(self, k=100):
        k = min(k, len(self.episode_wins))
        if len(self.episode_wins) == 0:
            return 0.0
        return float(np.mean(self.episode_wins[-k:]))

    def get_recent_avg_reward(self, k=1):
        k = min(k, len(self.episode_rewards))
        if len(self.episode_rewards) == 0:
            return 0.0
        return float(np.mean(self.episode_rewards[-k:]))
    
    def update_hud(self):
        # 逐帧HUD更新：根据当前统计与环境得分刷新左上角文本
        # 近期胜率与近期平均奖励（逐回合更新后逐帧显示）
        wr = self.recent_win_rate if self.total_episodes > 0 else self.get_recent_win_rate(100)
        ar10 = self.recent_avg_reward if self.total_episodes > 0 else self.get_recent_avg_reward(10)
        # 当前回合逐帧平均奖励
        ar_ep = (self.current_episode_reward / self.current_episode_frames) if self.current_episode_frames > 0 else 0.0
        left_score = self.env.left_score
        right_score = self.env.right_score
        # V均值逐帧显示
        v_mean = self.last_v_mean
        self.env.hud_lines = [
            f"Episode: {self.total_episodes}",
            f"Current_episode_frames: {self.current_episode_frames}",
            f"WinRate(100): {wr*100:.1f}%",
            f"AvgReward(10): {ar10:.2f}",
            f"AvgReward(ep): {ar_ep:.2f}",
            f"V(mean): {v_mean:.3f}",
            f"Score L:{left_score} R:{right_score}",
        ]
    
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
        # 保存图表（若可用)
        self.plot_training_curves(save_path=plot_path, show=show_plot)
        
        print(f"备份已生成: {folder}")
        return folder

def batch_update(bot_A2C):
    batch = list(bot_A2C.step_buffer)
    s_t, a, r, s_t1, done = zip(*batch)
    bot_A2C.update(s_t, a, r, s_t1, done)
    bot_A2C.step_cnt = 1
    #print("A2C updated")  
      
if __name__ == '__main__':
    env = PongEnv()
    bot_A2C = Bot_A2C(env, side=1)
    
    if os.path.exists("A2C.pth"):
        bot_A2C.load_model("A2C.pth")
    else:
        # 如果没有预训练模型，
        print("无预训练模型，网络已使用默认初始化权重")

        # 诊断：打印各层权重/偏置统计信息
        def print_layer_stats(net, net_name):
            for layer_name in ['fc1','fc2','fc3','fc4']:
                m = getattr(net, layer_name)
                w = m.weight.detach()
                b = m.bias.detach()
                print(f"{net_name}.{layer_name}: w(mean={w.mean().item():.4g}, std={w.std().item():.4g})  b(mean={b.mean().item():.4g}, std={b.std().item():.4g})")
        print_layer_stats(bot_A2C.actor, "policy_net")
        print_layer_stats(bot_A2C.critic, "value_net")

        # 诊断：输出分布检查，避免动作偏向
        with torch.no_grad():
            sample_states = torch.randn(1024, bot_A2C.input_dim)
            v_out = bot_A2C.critic(sample_states)
            print(f"初始化后V值均值: {v_out.mean().item():.4g}")
    
    episode_cnt = 0
    last_episode_cnt = -1
    
    while env.running:
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.running = False
                if AUTO_SAVE_ON_EXIT:
                    bot_A2C.save_model("A2C.pth")
                # 退出时自动按时间戳保存模型与图表（不弹窗）
                try:
                    bot_A2C.save_backup_package(show_plot=True)
                except Exception as e:
                    print("退出备份失败:", e)
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F2:
                    env.switch = not env.switch
                if event.key == pygame.K_F4:
                    # 快照：保存模型与训练图表到同一时间戳文件夹，并弹出图像
                    try:
                        bot_A2C.save_backup_package(show_plot=True)
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
            
        # 每batch_size步更新一次bot_A2C网络
        if(bot_A2C.step_cnt == bot_A2C.batch_size):
            batch_update(bot_A2C)
        
        s_t = env.get_state()
        a = bot_A2C.take_action(s_t)
        env.update_speed(a-1, bot_A2C.side)
        env.step()
        r = bot_A2C.calculate_r()
        s_t1 = env.get_state()
        done = env.check_over
        
        bot_A2C.step_buffer.append((s_t, a, r, s_t1, done))
        
        # 更新hud
        bot_A2C.update_hud()
        #控制台输出日志
        if episode_cnt % 128 == 0 and last_episode_cnt != episode_cnt:
            last_episode_cnt = episode_cnt
            print(f"Episode {episode_cnt:4d}  Steps={bot_A2C.total_steps}  score={env.left_score}:{env.right_score}  WinRate(100)={bot_A2C.get_recent_win_rate(100)*100:.1f}%  AvgReward={bot_A2C.get_recent_avg_reward():.2f}  AvgReward(100)={bot_A2C.get_recent_avg_reward(100):.2f}  Vmean={bot_A2C.last_v_mean:.3f}")
        
        if(done):
            batch_update(bot_A2C)
            
            # 更新回合统计
            episode_cnt += 1
            bot_A2C.update_statistics_on_episode_end(env.left_score, env.right_score)
            env.reset()
            
        bot_A2C.step_cnt +=1
            
        env.render()