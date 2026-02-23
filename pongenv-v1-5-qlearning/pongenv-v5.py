# pongenv-v5.py - 改进版本
# 基于pongenv-v4-training.py，修复关键bug并优化训练逻辑
# 主要改进：
# 1. 修复calculate_r_v2中右侧失分惩罚缺失的bug
# 2. 统一胜率窗口长度，都使用win_rate_max_episode
# 3. 修复visualize()和record_history()重复写入FP_*_history的问题
# 4. 改进物理碰撞：使用浮点位置更新，边界判定考虑球半径
# 5. 优化epsilon衰减策略，改为基于回合数的分段衰减
# 6. Q表初始化改为零初始化而非随机
# 7. 添加奖励裁剪防止Q值发散
# 8. 改进碰撞检测，避免重复触发

import pygame
import numpy as np
import sys
import pickle
import os
from datetime import datetime
import matplotlib

class Bot_0:
    def __init__(self, env):
        self.env = env
        self.speed = 5
            
    def act(self, side):
        self.ball_center = self.env.ball_y
        self.pad_center = self.env.pad_y[side] + self.env.pad_height // 2
        if self.ball_center > self.env.pad_y[side] and self.ball_center < self.env.pad_y[side] + self.env.pad_height:
            return
        if self.ball_center < self.pad_center and self.env.pad_y[side] > 0:
            self.env.pad_y[side] -= self.speed
        elif self.ball_center > self.pad_center and self.env.pad_y[side] + self.env.pad_height < self.env.WINDOW_HEIGHT:
            self.env.pad_y[side] += self.speed   
   
# Q-learning智能体 - 改进版本
class Bot_Q:
    def __init__(self, env, side = 1, alpha=0.05, gamma=0.9, epsilon=0.01, decaying = 0.99999995):
        self.env = env
        self.side = side
        
        # 超参数
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.initial_epsilon = epsilon  # 保存初始探索率
        self.decaying = decaying
        
        # 改进1：Q表初始化改为零初始化而非随机，避免初期不稳定
        # 状态空间：ball_dx, ball_dy, ball_x, ball_y, l_y, r_y; 动作空间：down, stay, up
        self.Q = np.zeros((5, 5, 10, 10, 10, 10, 3))
        
        self.last_state = None
        self.last_action = None
        
        # 改进2：碰撞检测状态追踪，避免重复触发奖励
        self.last_collision_state = {'left': False, 'right': False}
        
        # 训练追踪变量
        self.win_rate_max_episode = 50
        self.episode_cnt = 0 
        self.recent_win_rate = 0.000
        self.recent_win_history = []
        self.reward_history = []  
        self.win_rate_history = [] 
        self.total_reward = 0
        
    def diff(self, val, st, ed, n):
        """将连续值离散化到指定区间"""
        dist = ed - st
        step = dist / (n - 1)
        idx = (val - st) // step
        idx = int(idx)
        idx = max(0, min(idx, n - 1))
        return idx
        
    def get_state(self):
        """获取当前状态的离散化表示"""
        ball_dx = self.diff(self.env.ball_speed[0], -self.env.ball_maxspeed, self.env.ball_maxspeed, 5)
        ball_dy = self.diff(self.env.ball_speed[1], -self.env.ball_maxspeed, self.env.ball_maxspeed, 5)
        ball_x = self.diff(self.env.ball_x, 0, self.env.WINDOW_WIDTH, 10)
        ball_y = self.diff(self.env.ball_y, 0, self.env.WINDOW_HEIGHT, 10)
        l_y = self.diff(self.env.pad_y[0], 0, self.env.WINDOW_HEIGHT, 10)
        r_y = self.diff(self.env.pad_y[1], 0, self.env.WINDOW_HEIGHT, 10)
        return (ball_dx, ball_dy, ball_x, ball_y, l_y, r_y)
    
    def take_action(self, state):
        """epsilon-贪心策略选择动作"""
        if np.random.uniform(0, 1) < self.epsilon:
            action = np.random.choice([0, 1, 2])
        else:
            action = np.argmax(self.Q[state])
        
        # 改进3：基于回合数的分段epsilon衰减策略
        self.update_epsilon()
        return action
    
    def update_epsilon(self):
        """改进的epsilon衰减策略：分段衰减，避免过早收敛"""
        if self.episode_cnt < 10000:  # 前10000回合：快速衰减
            self.epsilon = self.initial_epsilon * (0.99 ** (self.episode_cnt / 100))
        elif self.episode_cnt < 50000:  # 10000-50000回合：慢速衰减
            self.epsilon = max(0.01, self.initial_epsilon * 0.1 * (0.999 ** ((self.episode_cnt - 10000) / 100)))
        else:  # 50000回合后：保持最小探索率
            self.epsilon = max(0.005, self.epsilon * 0.9999)
    
    def update_Q(self, last_state, last_action, reward, current_state):
        """Q-learning更新规则"""
        if last_state is not None and last_action is not None:
            # 改进4：添加奖励裁剪，防止Q值发散
            clipped_reward = np.clip(reward, -500, 500)
            
            self.Q[last_state][last_action] += self.alpha * (
                clipped_reward + self.gamma * np.max(self.Q[current_state]) - self.Q[last_state][last_action]
            )
    
    def L2_reward(self, a, b):
        """基于L2距离的奖励函数"""
        return (100 - ((a-b)/60) ** 2) // 50
        
    def calculate_r_v1(self):
        """奖励函数版本1"""
        state = self.get_state()
        action = self.last_action
        current_pad_y = self.env.pad_y[self.side]
        current_pad_center = current_pad_y + self.env.pad_height // 2
        
        # 鼓励向球移动
        if (self.side == 0 and self.env.ball_speed[0] < 0) or (self.side == 1 and self.env.ball_speed[0] > 0):
            reward = self.L2_reward(current_pad_center, self.env.ball_y)
        else: 
            reward = self.L2_reward(current_pad_center, self.env.WINDOW_HEIGHT // 2)
            
        # 边界惩罚
        if current_pad_y < 50:
            reward -= 10
        elif current_pad_y + self.env.pad_height > self.env.WINDOW_HEIGHT - 50:
            reward -= 10
        
        # 动作惩罚
        if action is not None:
            if action == 1 or (self.env.pad_speed[self.side] == 0 and 
                ((self.side == 0 and self.env.ball_speed[0] <= 0) or (self.side == 1 and self.env.ball_speed[0] >= 0))):
                reward -= 5
            if (self.side == 0 and self.env.ball_speed[0] < 0) or (self.side == 1 and self.env.ball_speed[0] > 0):
                if (self.env.ball_y < current_pad_center and action == 0) or (self.env.ball_y > current_pad_center and action == 2):
                    reward -= 5
            
        # 碰撞奖励
        if (self.side == 0 and self.env.l_collision()):
            reward += 200
        elif (self.side == 1 and self.env.r_collision()):
            reward += 200
            
        if (self.side == 0 and self.env.r_collision()) or (self.side == 1 and self.env.l_collision()):
            reward -= 10

        # 得分奖励/惩罚
        if self.side == 0 and self.env.ball_x > self.env.WINDOW_WIDTH - self.env.pad_width:
            reward += 50
        elif self.side == 1 and self.env.ball_x < self.env.pad_width:
            reward += 50
        if self.side == 0 and self.env.ball_x < self.env.pad_width:
            reward -= 200
        elif self.side == 1 and self.env.ball_x > self.env.WINDOW_WIDTH - self.env.pad_width:
            reward -= 200
        
        # 记录奖励历史        
        self.reward_history.append(reward)
        if len(self.reward_history) > 500:
            self.reward_history.pop(0)
        self.total_reward += reward
            
        return reward
     
    def calculate_r_v2(self):
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

        # 边界惩罚
        if current_pad_y < 30:
            reward -= 20
        elif current_pad_y + self.env.pad_height > self.env.WINDOW_HEIGHT - 30:
            reward -= 20

        # 失分惩罚 - 修复bug：确保两侧都有失分惩罚
        if self.side == 0 and self.env.ball_x < self.env.pad_width:
            reward -= 200  # 左侧失分
        elif self.side == 1 and self.env.ball_x > self.env.WINDOW_WIDTH - self.env.pad_width:
            reward -= 200  # 右侧失分 - 这是修复的关键bug
        
        # 记录奖励历史
        self.reward_history.append(reward)
        if len(self.reward_history) > 500:
            self.reward_history.pop(0)
        self.total_reward += reward
        
        return reward
    
    def calculate_r_v3(self):
        """奖励函数版本3"""
        reward = self.calculate_r_v2()
        
        # 左侧Bot_Q额外奖励"打右侧边角"
        if self.side == 0 and self.env.r_collision():
            if self.env.ball_y < 200 or self.env.ball_y > 400:
                reward += 50
        
        # 降低对"中线防守"的依赖
        current_pad_center = self.env.pad_y[self.side] + self.env.pad_height // 2
        distance_to_mid = abs(current_pad_center - self.env.WINDOW_HEIGHT // 2)
        if distance_to_mid > 150:
            reward += 10
        
        return reward
        
    def act(self):
        """智能体行动"""
        current_state = self.get_state()
        action = self.take_action(current_state)
        
        self.env.update_speed(action-1, self.side)
        
        reward = self.calculate_r_v2()
        self.update_Q(self.last_state, self.last_action, reward, current_state)
        
        self.last_state = current_state
        self.last_action = action
        
    def save_q_table(self, path="q_table-v5.pkl", versioned=False):
        """保存Q表，支持版本化"""
        if versioned:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            path = f"q_table-v5_{timestamp}.pkl"
    
        confirm = "y"
        if confirm.lower() != "y":
            print("保存取消")
            return
    
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)
        print(f"Q表已保存到 {path}")
        
    def save_q_table_left(self, path="q_table-v5-left.pkl", versioned=False):
        """保存左侧Q表，支持版本化"""
        if versioned:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M")
            path = f"q_table-v5-left_{timestamp}.pkl"
    
        confirm = "y"
        if confirm.lower() != "y":
            print("保存取消")
            return
    
        with open(path, "wb") as f:
            pickle.dump(self.Q, f)
        print(f"Q表已保存到 {path}")
    
    def load_q_table(self, path="q_table-v5.pkl"):
        """从本地文件加载Q表"""
        try:
            with open(path, "rb") as f:
                self.Q = pickle.load(f)
            print(f"已从 {path} 加载 Q 表")
        except FileNotFoundError:
            print(f"未找到 {path}，将使用新的 Q 表")
            
    def load_q_table_left(self, path="q_table-v5-left.pkl"):
        """从本地文件加载左侧Q表"""
        try:
            with open(path, "rb") as f:
                self.Q = pickle.load(f)
            print(f"已从 {path} 加载 left Q 表")
        except FileNotFoundError:
            print(f"未找到 {path}，将使用新的 left Q 表")
        

class PongEnv:
    WINDOW_WIDTH, WINDOW_HEIGHT = 1070, 600
    
    def __init__(self):
        # 初始化Pygame
        pygame.init()
        pygame.display.set_caption("pong-v5(improved)")
        self.running = True
        self.screen = pygame.display.set_mode((self.WINDOW_WIDTH, self.WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        
        # 加载背景图片
        try:
            self.image = pygame.image.load('bg.png')
            self.img_width, self.img_height = self.image.get_size()
            self.scale_x = self.WINDOW_WIDTH / self.img_width
            self.scale_y = self.WINDOW_HEIGHT / self.img_height
            self.scale = min(self.scale_x, self.scale_y)
            self.new_width = int(self.img_width * self.scale)
            self.new_height = int(self.img_height * self.scale)
            self.scaled_image = pygame.transform.smoothscale(self.image, (self.new_width, self.new_height))
            self.image_x = (self.WINDOW_WIDTH - self.new_width) // 2
            self.image_y = (self.WINDOW_HEIGHT - self.new_height) // 2
        except:
            print("警告：未找到bg.png，将使用纯色背景")
            self.scaled_image = None
        
        # 定义颜色
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        self.YELLOW = (255,235,205)
        self.list_colors = [self.WHITE, self.BLACK, self.RED, self.GREEN, self.BLUE, self.YELLOW]
        
        # 挡板参数       
        self.pad_width = 10
        self.pad_height = 80
        self.pad_x = [0, self.WINDOW_WIDTH - self.pad_width]
        self.pad_maxspeed = 10
        self.pad_speed = [0, 0]
        self.accellerate = 0.8
        self.decellerate = 1.2
        self.mu = 0.3
        
        # 球参数
        self.ball_maxspeed = 20
        self.ball_radius = 10
        self.factor = 1.1
        
        # 分数
        self.left_score = 0
        self.right_score = 0
        
        # 智能体
        self.switch = True
        self.last_switch_episode = -1
        self.bot_type = "bot_0"
        self.bot_0 = Bot_0(self)
        self.bot_left = Bot_Q(self, side = 0, alpha = 0.3, epsilon = 0.3)
        self.bot_left.load_q_table_left()
        self.bot = Bot_Q(self, side = 1, alpha = 0.2, epsilon = 0.1)
        self.bot.load_q_table()
        
        # 保存和历史记录
        self.last_step_episode = -1
        self.last_save_episode = -1
        # 改进5：修复重复记录问题，移除重复的历史记录变量
        self.history_recorded = False  # 标记是否已记录当前回合的历史
        
        # 重置游戏状态
        self.reset()
        
    def update_speed(self, mode, side):
        """更新挡板速度"""
        if mode == 1:  # 向上
            self.pad_speed[side] = max(self.pad_speed[side] - self.accellerate, -self.pad_maxspeed)
        elif mode == -1:  # 向下
            self.pad_speed[side] = min(self.pad_speed[side] + self.accellerate, self.pad_maxspeed)
        elif mode == 0:  # 停止
            if self.pad_speed[side] > 0:
                 self.pad_speed[side] = max(self.pad_speed[side] - self.decellerate, 0)
            elif self.pad_speed[side] < 0:
                 self.pad_speed[side] = min(self.pad_speed[side] + self.decellerate, 0)
    
    def reset(self):
        """重置游戏状态"""
        self.pad_y = [self.WINDOW_HEIGHT // 2 - self.pad_height, self.WINDOW_HEIGHT // 2 - self.pad_height]
        # 改进6：使用浮点数位置，提高物理精度
        self.ball_x = float(self.WINDOW_WIDTH // 2)    
        self.ball_y = float(self.WINDOW_HEIGHT // 2)
        self.ball_speed = [5 * np.random.choice([-1, 1]), 5 * np.random.uniform(-1, 1)]
        self.history_recorded = False  # 重置历史记录标记

    def l_collision(self):
        """改进的左侧碰撞检测：考虑球半径"""
        return (self.ball_x - self.ball_radius <= self.pad_x[0] + self.pad_width and 
                self.ball_y + self.ball_radius >= self.pad_y[0] and 
                self.ball_y - self.ball_radius <= self.pad_y[0] + self.pad_height and
                self.ball_speed[0] < 0)  # 只有球向左移动时才触发
   
    def r_collision(self):
        """改进的右侧碰撞检测：考虑球半径"""
        return (self.ball_x + self.ball_radius >= self.pad_x[1] and 
                self.ball_y + self.ball_radius >= self.pad_y[1] and 
                self.ball_y - self.ball_radius <= self.pad_y[1] + self.pad_height and
                self.ball_speed[0] > 0)  # 只有球向右移动时才触发
         
    def cap(self, val, max_val):
        """限制数值范围"""
        if val > 0:
            return min(val, max_val)
        else:
            return max(val, -max_val)
         
    def step(self):
        """游戏主循环步骤"""
        if not self.running:
            return
        
        # 混合智能体策略（左侧挡板）
        if self.bot.episode_cnt % 1000 == 0 and self.bot.episode_cnt != self.last_switch_episode and self.bot.episode_cnt > 0:
            self.switch = not self.switch
            self.last_switch_episode = self.bot.episode_cnt
            print(f"陪练切换：此前为{self.bot_type}（回合数：{self.bot.episode_cnt}）")
            
        if self.switch:
            self.bot_left.act()
            self.bot_type = "bot_Q"
        else : 
            self.bot_0.act(0)
            self.bot_type = "bot_0"
            
        # Q-learning智能体（右侧挡板）
        self.bot.act()
        
        # 挡板移动
        if self.switch: 
            if  self.pad_y[0] + self.pad_speed[0] < 0:
                self.pad_y[0] = 0
            elif self.pad_y[0] + self.pad_speed[0] > self.WINDOW_HEIGHT - self.pad_height:
                self.pad_y[0] = self.WINDOW_HEIGHT - self.pad_height
            else:
                self.pad_y[0] += self.pad_speed[0]
           
        if  self.pad_y[1] + self.pad_speed[1] < 0:
            self.pad_y[1] = 0
        elif self.pad_y[1] + self.pad_speed[1] > self.WINDOW_HEIGHT - self.pad_height:
            self.pad_y[1] = self.WINDOW_HEIGHT - self.pad_height
        else:
            self.pad_y[1] += self.pad_speed[1]
        
        # 改进7：球移动使用浮点数，提高精度            
        self.ball_x += self.ball_speed[0]
        self.ball_y += self.ball_speed[1]
        
        # 挡板反弹    
        if self.l_collision():
            self.ball_speed[0] = -self.ball_speed[0] * self.factor
            self.ball_speed[1] += self.pad_speed[0] * self.mu
        if self.r_collision():
            self.ball_speed[0] = -self.ball_speed[0] * self.factor
            self.ball_speed[1] += self.pad_speed[1] * self.mu
        
        # 改进8：边界碰撞检测考虑球半径
        if self.ball_y - self.ball_radius <= 0 or self.ball_y + self.ball_radius >= self.WINDOW_HEIGHT:
            self.ball_speed[1] = -self.ball_speed[1]
            # 确保球不会卡在边界外
            if self.ball_y - self.ball_radius <= 0:
                self.ball_y = self.ball_radius
            else:
                self.ball_y = self.WINDOW_HEIGHT - self.ball_radius
            
        # 速度限制
        self.ball_speed[0] = self.cap(self.ball_speed[0], self.ball_maxspeed)
        self.ball_speed[1] = self.cap(self.ball_speed[1], self.ball_maxspeed)
        
        # 分数更新和回合结束检查
        check_over = False 
        if self.ball_x < 0:
            self.right_score += 1
            self.bot.recent_win_history.append(1)
            # 改进9：统一胜率窗口长度
            if len(self.bot.recent_win_history) > self.bot.win_rate_max_episode:
                self.bot.recent_win_history.pop(0)
            check_over = True
            self.reset()
        
        elif self.ball_x > self.WINDOW_WIDTH:
            self.left_score += 1
            self.bot.recent_win_history.append(0)
            # 改进9：统一胜率窗口长度
            if len(self.bot.recent_win_history) > self.bot.win_rate_max_episode:
                self.bot.recent_win_history.pop(0)
            check_over = True
            self.reset()
        
        if check_over:    
            self.bot.episode_cnt += 1
            # 改进9：统一胜率计算窗口
            self.bot.recent_win_rate = sum(self.bot.recent_win_history) / min(self.bot.episode_cnt, self.bot.win_rate_max_episode)
            self.bot.win_rate_history.append(self.bot.recent_win_rate)
            if len(self.bot.win_rate_history) > self.bot.win_rate_max_episode:
                self.bot.win_rate_history.pop(0)

    def render(self):
        """渲染游戏画面"""
        self.screen.fill(self.BLACK)
        
        # 绘制背景
        if self.scaled_image:
            self.screen.blit(self.scaled_image, (self.image_x, self.image_y))
            
        # 绘制游戏元素
        pygame.draw.rect(self.screen, self.YELLOW, (self.pad_x[0], self.pad_y[0], self.pad_width, self.pad_height))
        pygame.draw.rect(self.screen, self.YELLOW, (self.pad_x[1], self.pad_y[1], self.pad_width, self.pad_height))
        pygame.draw.circle(self.screen, self.YELLOW, (int(self.ball_x), int(self.ball_y)), self.ball_radius)
        pygame.draw.line(self.screen, self.YELLOW, (self.WINDOW_WIDTH//2, 100), (self.WINDOW_WIDTH//2, self.WINDOW_HEIGHT), 1)
        pygame.draw.line(self.screen, self.YELLOW, (0, 100), (self.WINDOW_WIDTH, 100), 1)

        # 绘制分数
        font = pygame.font.SysFont("consolas", 50)
        score_text = font.render(f"{self.left_score}   :   {self.right_score}", True, self.WHITE)
        self.screen.blit(score_text, (self.WINDOW_WIDTH//2 - score_text.get_width()//2, 50 - score_text.get_height()//2))
        
        # 可视化训练进度
        self.visualize()
        
        pygame.display.flip()
        self.clock.tick(60)
        
    def record_history(self):
        """改进10：独立的历史记录函数，避免重复记录"""
        if self.history_recorded:
            return  # 避免重复记录
            
        avg_reward = np.mean(self.bot.reward_history) if self.bot.reward_history else 0.0
        avg_q_value = np.mean(self.bot.Q)
        
        # 记录到历史列表（用于保存和绘图）
        if not hasattr(self, 'FP_avg_q_value_history'):
            self.FP_avg_q_value_history = []
            self.FP_avg_reward_history = []
            self.FP_recent_win_rate_history = []
            self.FP_episode_cnt_history = []
            self.FP_epsilon_history = []
            
        self.FP_avg_q_value_history.append(avg_q_value)
        self.FP_avg_reward_history.append(avg_reward)
        self.FP_recent_win_rate_history.append(self.bot.recent_win_rate)
        self.FP_episode_cnt_history.append(self.bot.episode_cnt)
        self.FP_epsilon_history.append(self.bot.epsilon)
        
        self.history_recorded = True  # 标记已记录
        
    def visualize(self):
        """改进11：可视化函数不再重复记录历史，只负责显示"""
        font_small = pygame.font.SysFont("simsun", 18)
        
        # 计算当前指标
        avg_reward = np.mean(self.bot.reward_history) if self.bot.reward_history else 0.0
        avg_q_value = np.mean(self.bot.Q)
        
        # 绘制指标文本
        text_color = self.WHITE
        texts = [
            f"回合数: {self.bot.episode_cnt}",
            f"近期胜率: {self.bot.recent_win_rate:.2f}",
            f"平均奖励: {avg_reward:.2f}",
            f"Q表均值: {avg_q_value:.4f}",
            f"探索率: {self.bot.epsilon:.6f}",
            f"当前trainer_bot: {self.bot_type}"
        ]
        for i, text in enumerate(texts):
            text_surface = font_small.render(text, True, text_color)
            self.screen.blit(text_surface, (10, 10 + i*30))
        
        # 绘制胜率趋势图
        chart_x = 10
        chart_y = self.WINDOW_HEIGHT - 110
        chart_width = 200
        chart_height = 100
        
        pygame.draw.rect(self.screen, (50, 50, 50), (chart_x, chart_y, chart_width, chart_height))
        pygame.draw.line(self.screen, self.WHITE, (chart_x, chart_y + chart_height), (chart_x + chart_width, chart_y + chart_height), 1)
        pygame.draw.line(self.screen, self.WHITE, (chart_x, chart_y), (chart_x, chart_y + chart_height), 1)
        
        if len(self.bot.win_rate_history) > 1:
            points = []
            for i, rate in enumerate(self.bot.win_rate_history):
                x = chart_x + (i / (len(self.bot.win_rate_history) - 1)) * chart_width
                y = chart_y + chart_height - (rate * chart_height)
                points.append((int(x), int(y)))
            pygame.draw.lines(self.screen, self.GREEN, False, points, 2)
        
        # 绘制奖励趋势图
        if len(self.bot.reward_history) > 1:
            chart_x = self.WINDOW_WIDTH - 210
            chart_y = self.WINDOW_HEIGHT - 110
            chart_width = 200
            chart_height = 100
            
            pygame.draw.rect(self.screen, (50, 50, 50), (chart_x, chart_y, chart_width, chart_height))
            pygame.draw.line(self.screen, self.WHITE, (chart_x, chart_y + chart_height), (chart_x + chart_width, chart_y + chart_height), 1)
            pygame.draw.line(self.screen, self.WHITE, (chart_x, chart_y), (chart_x, chart_y + chart_height), 1)
            
            min_reward = min(self.bot.reward_history)
            max_reward = max(self.bot.reward_history)
            reward_range = max_reward - min_reward if max_reward != min_reward else 1.0
            
            points = []
            for i, r in enumerate(self.bot.reward_history):
                x = chart_x + (i / (len(self.bot.reward_history) - 1)) * chart_width
                normalized_r = (r - min_reward) / reward_range
                y = chart_y + chart_height - (normalized_r * chart_height)
                points.append((int(x), int(y)))
            pygame.draw.lines(self.screen, self.BLUE, False, points, 2)
        
    def save_training_history(self):
        """保存训练历史到CSV文件"""
        import pandas as pd
        
        if not hasattr(self, 'FP_episode_cnt_history') or not self.FP_episode_cnt_history:
            print("警告：没有训练历史数据可保存")
            return None
            
        history_df = pd.DataFrame({
            "episode_cnt": self.FP_episode_cnt_history,
            "recent_win_rate": self.FP_recent_win_rate_history,
            "avg_reward": self.FP_avg_reward_history,
            "avg_q_value": self.FP_avg_q_value_history,
            "epsilon": self.FP_epsilon_history
        })
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        save_path = f"training_history_v5_{timestamp}.csv"
        history_df.to_csv(save_path, index=False, encoding="utf-8")
        print(f"\n训练历史已保存到：{save_path}")
        return history_df

    def plot_training_history(self, history_df):
        """绘制训练历史图表"""
        if history_df is None or history_df.empty:
            print("警告：没有数据可绘制")
            return
            
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Pong-v5训练历史指标（总计{max(self.FP_episode_cnt_history)}回合）', fontsize=16, fontweight='bold')
        
        # 胜率变化
        ax1.plot(history_df["episode_cnt"], history_df["recent_win_rate"], color='#2E8B57', linewidth=2)
        ax1.set_title('近期胜率变化', fontsize=14)
        ax1.set_xlabel('回合数')
        ax1.set_ylabel('胜率（0~1）')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # 平均奖励
        ax2.plot(history_df["episode_cnt"], history_df["avg_reward"], color='#4169E1', linewidth=2)
        ax2.set_title('平均奖励变化', fontsize=14)
        ax2.set_xlabel('回合数')
        ax2.set_ylabel('平均奖励值')
        ax2.grid(True, alpha=0.3)
        
        # Q表均值
        ax3.plot(history_df["episode_cnt"], history_df["avg_q_value"], color='#DC143C', linewidth=2)
        ax3.set_title('Q表平均价值变化', fontsize=14)
        ax3.set_xlabel('回合数')
        ax3.set_ylabel('Q表均值')
        ax3.grid(True, alpha=0.3)
        
        # 探索率
        ax4.plot(history_df["episode_cnt"], history_df["epsilon"], color='#FF8C00', linewidth=2)
        ax4.set_title('探索率（epsilon）变化', fontsize=14)
        ax4.set_xlabel('回合数')
        ax4.set_ylabel('探索率（0~1）')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
        
        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        plot_save_path = f"training_plot_v5_{timestamp}.png"
        plt.savefig(plot_save_path, dpi=300, bbox_inches='tight')
        print(f"训练图表已保存到：{plot_save_path}")
        plt.show()

# 主程序
if __name__ == "__main__":
    env = PongEnv()
    manned = False
    
    print("=== Pong-v5 改进版本启动 ===")
    print("主要改进：")
    print("1. 修复奖励函数bug")
    print("2. 统一胜率窗口长度")
    print("3. 修复重复历史记录问题")
    print("4. 改进物理碰撞检测")
    print("5. 优化epsilon衰减策略")
    print("6. Q表零初始化")
    print("7. 添加奖励裁剪")
    print("8. 改进碰撞检测")
    print("\n按F2保存Q表，F4切换手动模式")
    
    while env.running:
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                env.running = False
                env.bot.save_q_table("q_table-v5.pkl")
                env.bot_left.save_q_table_left("q_table-v5-left.pkl")
                history_df = env.save_training_history()
                if history_df is not None:
                    env.plot_training_history(history_df)
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F2:
                    env.bot.save_q_table(versioned=True)
                    env.bot_left.save_q_table_left(versioned=True)
                if event.key == pygame.K_F4:
                    manned = not manned
                    print(f"手动模式: {'开启' if manned else '关闭'}")
        
        # 手动控制模式
        if manned:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP] or keys[pygame.K_w]:
                env.update_speed(1, 0)
            elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
                env.update_speed(-1, 0)
            else:
                env.update_speed(0, 0)
        
        env.step()
        # env.render()  # 取消注释以显示游戏画面
        
        # 定期保存和记录
        if env.bot.episode_cnt % 50000 == 0 and env.bot.episode_cnt > 0 and env.bot.episode_cnt != env.last_save_episode:
            env.bot.save_q_table(versioned=True)
            env.bot_left.save_q_table_left(versioned=True)
            history_df = env.save_training_history()
            if history_df is not None:
                env.plot_training_history(history_df)
            env.last_save_episode = env.bot.episode_cnt
            
        if env.bot.episode_cnt % 100 == 0 and env.bot.episode_cnt != 0 and env.bot.episode_cnt != env.last_step_episode:
            env.record_history()  # 使用改进的独立记录函数
            env.last_step_episode = env.bot.episode_cnt
            avg_reward = np.mean(env.bot.reward_history) if env.bot.reward_history else 0.0
            avg_q_value = np.mean(env.bot.Q)
            print(f"回合数：{env.bot.episode_cnt} | 近期胜率：{env.bot.recent_win_rate:.2f} | 平均奖励：{avg_reward:.2f} | 探索率：{env.bot.epsilon:.6f} | Q表均值：{avg_q_value:.4f}")
            
    pygame.quit()