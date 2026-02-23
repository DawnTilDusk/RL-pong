from PongEnv import PongEnv
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from collections import deque
import pygame
import sys

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
    def __init__(self, env: PongEnv, side = 1, alpha = 1e-4, batch_size = 32, gamma = 0.99):
        self.env = env
        self.side = side
        
        #hyperparameters
        self.alpha = alpha
        self.gamma = 0.99
        self.gamma = gamma
        self.batch_size = batch_size
        
        self.input_dim = 8
        self.output_dim = 3
        
        self.step_cnt = 1
        self.step_buffer = deque(maxlen=self.batch_size)
        
        self.state = self.env.get_state()
        
        self.actor = PolicyNet(self.input_dim, self.output_dim)
        self.critic = ValueNet(self.input_dim)
        
        self.actor_optim = torch.optim.AdamW(self.actor.parameters(), lr=self.alpha)
        self.critic_optim = torch.optim.AdamW(self.critic.parameters(), lr=self.alpha)
        
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)
        logits = self.actor(state)
        action_dist = torch.distributions.Categorical(logits=logits)
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

        # 无效靠近惩罚
        is_ball_coming = (self.side == 0 and self.env.ball_speed[0] < 0) or (self.side == 1 and self.env.ball_speed[0] > 0)
        is_close_enough = abs(current_pad_center - self.env.ball_y) < 30
        is_missed = (self.side == 0 and self.env.ball_x < self.env.pad_width) or (self.side == 1 and self.env.ball_x > self.env.WINDOW_WIDTH - self.env.pad_width)
        if is_ball_coming and is_close_enough and is_missed:
            reward -= 100

        # 两侧都有失分惩罚
        if self.side == 0 and self.env.ball_x < self.env.pad_width:
            reward -= 200  # 左侧失分
        elif self.side == 1 and self.env.ball_x > self.env.WINDOW_WIDTH - self.env.pad_width:
            reward -= 200  # 右侧失分 
        
        return reward

def batch_update(bot_A2C):
    batch = list(bot_A2C.step_buffer)
    s_t, a, r, s_t1, done = zip(*batch)
    bot_A2C.update(s_t, a, r, s_t1, done)
    bot_A2C.step_cnt = 1
    print("A2C updated")  
      
if __name__ == '__main__':
    env = PongEnv()
    bot_A2C = Bot_A2C(env, side=1)
    
    while env.running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.running = False
                pygame.quit()
                sys.exit()
        
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
        
        if(done):
            env.reset()
            batch_update(bot_A2C)
            
        bot_A2C.step_cnt +=1
            
        env.render()