from PongEnv import PongEnv
import collections
import random
import torch
import torch.nn as nn
import numpy as np
import pygame
import sys
import os

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)
    
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
    def __init__(self, env: PongEnv, side = 1, alpha = 1e-3, gamma = 0.99, epsilon = 0.1, decaying = 0.99995):
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
        
        self.buffer = Replay_Buffer(10000)
        self.warmup = 2000
        self.batch_size = 32
        
        self.optimizer = torch.optim.Adam(self.q_main.parameters(), lr=self.alpha)
        self.state = self.get_state()
        self.last_action = None
        self.training_steps = 0
        
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
        if np.random.rand() < epsilon:
            return np.random.randint(0, self.output_dim)
        else:
            state = self.get_state()
            q_values = self.q_main(torch.FloatTensor(state))
            epsilon = max(epsilon * self.decaying, 0.00001)
            self.epsilon = epsilon
            return torch.argmax(q_values).item()
 
    def L2_reward(self, a, b):
        return (100 - ((a - b) / 100) ** 2) / 5.0
 
    def calculate_r(self):
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
        
        # record reward history        
#        self.reward_history.append(reward)
#        if len(self.reward_history) > 500:
#            self.reward_history.pop(0)
#        self.total_reward += reward
            
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
        loss = nn.MSELoss()(q_taken, q_target)
        
        self.optimizer.zero_grad()
        self.training_steps += 1
        loss.backward()
        self.optimizer.step()
        if (self.training_steps+1) % 100 == 0:
            self.q_target.load_state_dict(self.q_main.state_dict())
    
    def save_model(self, path="dqn.pth"):
        torch.save({
            "q_main": self.q_main.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_steps": self.training_steps
        }, path)
    
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
        
if __name__ == "__main__":
    env = PongEnv()
    bot_DQN = Bot_DQN(env)
    if os.path.exists("dqn.pth"):
        bot_DQN.load_model("dqn.pth")
    right_manned = False
    episode_cnt = 0
    while env.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.running = False
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F2:
                    env.switch = not env.switch
                if event.key == pygame.K_F4:
                    bot_DQN.save_model("dqn.pth")
                    print("已保存模型到 dqn.pth")
    
        if right_manned:
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
        if bot_DQN.buffer.length() > bot_DQN.warmup:
            batch = bot_DQN.buffer.sample(bot_DQN.batch_size)
            bot_DQN.update_Q(batch)
        
        if done_t: 
            print(f"Episode {episode_cnt:4d}  score={env.left_score}:{env.right_score}  ε={bot_DQN.epsilon:.5f}")
            episode_cnt += 1
            env.reset()
        
        env.render()
        
pygame.quit()