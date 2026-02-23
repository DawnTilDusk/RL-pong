# pongenv-battle.py - 对战版本
# 基于pongenv-v5.py，专门用于对战模式
# 支持三种模式：人机对战(人vs Bot_Q)、Bot对战(Bot_Q vs Bot_0)、双人对战(人vs人)
# 只使用已训练好的Q表，不进行训练和数据保存

import pygame
import numpy as np
import sys
import pickle
import os

class Bot_0:
    """简单AI对手 - 跟随球的移动"""
    def __init__(self, env):
        self.env = env
        self.speed = 5
            
    def act(self, side):
        """Bot_0的行动逻辑：简单跟随球"""
        self.ball_center = self.env.ball_y
        self.pad_center = self.env.pad_y[side] + self.env.pad_height // 2
        
        if self.ball_center > self.env.pad_y[side] and self.ball_center < self.env.pad_y[side] + self.env.pad_height:
            return
        
        if self.ball_center < self.pad_center and self.env.pad_y[side] > 0:
            self.env.pad_y[side] -= self.speed
        elif self.ball_center > self.pad_center and self.env.pad_y[side] + self.env.pad_height < self.env.WINDOW_HEIGHT:
            self.env.pad_y[side] += self.speed   

class Bot_Q:
    """Q-learning智能体 - 对战版本（只使用已训练的Q表）"""
    def __init__(self, env, side=1, q_table_path=None):
        self.env = env
        self.side = side
        
        # 初始化Q表
        self.Q = np.zeros((5, 5, 10, 10, 10, 10, 3))
        
        # 加载已训练的Q表
        if q_table_path:
            self.load_q_table(q_table_path)
        else:
            # 默认路径
            if side == 0:
                self.load_q_table("q_table-v5-left.pkl")
            else:
                self.load_q_table("q_table-v5.pkl")
        
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
        """贪心策略选择最优动作（不再探索）"""
        return np.argmax(self.Q[state])
    
    def act(self):
        """智能体行动（对战模式，不更新Q表）"""
        current_state = self.get_state()
        action = self.take_action(current_state)
        self.env.update_speed(action-1, self.side)
    
    def load_q_table(self, path):
        """从本地文件加载Q表"""
        try:
            with open(path, "rb") as f:
                self.Q = pickle.load(f)
            print(f"已从 {path} 加载 Q 表")
        except FileNotFoundError:
            print(f"警告：未找到 {path}，将使用随机策略")
            print("请确保已有训练好的Q表文件")

class PongBattle:
    """Pong对战游戏环境"""
    WINDOW_WIDTH, WINDOW_HEIGHT = 1070, 600
    
    def __init__(self):
        # 初始化Pygame
        pygame.init()
        pygame.display.set_caption("Pong Battle - 对战模式")
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
        self.YELLOW = (255, 235, 205)
        
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
        
        # 游戏模式
        # 0: 人机对战(人vs Bot_Q)
        # 1: Bot对战(Bot_Q vs Bot_0) 
        # 2: 双人对战(人vs人)
        self.game_mode = 0
        self.mode_names = ["人机对战(人vs Bot_Q)", "Bot对战(Bot_Q vs Bot_0)", "双人对战(人vs人)"]
        
        # 初始化智能体
        self.bot_0 = Bot_0(self)
        self.bot_q_left = Bot_Q(self, side=0)  # 左侧Bot_Q
        self.bot_q_right = Bot_Q(self, side=1)  # 右侧Bot_Q
        
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
        self.ball_x = float(self.WINDOW_WIDTH // 2)    
        self.ball_y = float(self.WINDOW_HEIGHT // 2)
        self.ball_speed = [5 * np.random.choice([-1, 1]), 5 * np.random.uniform(-1, 1)]

    def l_collision(self):
        """左侧碰撞检测"""
        return (self.ball_x - self.ball_radius <= self.pad_x[0] + self.pad_width and 
                self.ball_y + self.ball_radius >= self.pad_y[0] and 
                self.ball_y - self.ball_radius <= self.pad_y[0] + self.pad_height and
                self.ball_speed[0] < 0)
   
    def r_collision(self):
        """右侧碰撞检测"""
        return (self.ball_x + self.ball_radius >= self.pad_x[1] and 
                self.ball_y + self.ball_radius >= self.pad_y[1] and 
                self.ball_y - self.ball_radius <= self.pad_y[1] + self.pad_height and
                self.ball_speed[0] > 0)
         
    def cap(self, val, max_val):
        """限制数值范围"""
        if val > 0:
            return min(val, max_val)
        else:
            return max(val, -max_val)
    
    def switch_mode(self):
        """切换游戏模式"""
        self.game_mode = (self.game_mode + 1) % 3
        print(f"切换到模式：{self.mode_names[self.game_mode]}")
        self.reset()  # 重置游戏状态
         
    def step(self, left_action=None, right_action=None):
        """游戏主循环步骤"""
        if not self.running:
            return
        
        # 根据游戏模式控制挡板
        if self.game_mode == 0:  # 人机对战(人vs Bot_Q)
            # 左侧：人类控制（通过left_action参数）
            if left_action is not None:
                self.update_speed(left_action, 0)
            # 右侧：Bot_Q控制
            self.bot_q_right.act()
            
        elif self.game_mode == 1:  # Bot对战(Bot_Q vs Bot_0)
            # 左侧：Bot_Q控制
            self.bot_q_left.act()
            # 右侧：Bot_0控制
            self.bot_0.act(1)
            
        elif self.game_mode == 2:  # 双人对战(人vs人)
            # 左侧：人类控制（通过left_action参数）
            if left_action is not None:
                self.update_speed(left_action, 0)
            # 右侧：人类控制（通过right_action参数）
            if right_action is not None:
                self.update_speed(right_action, 1)
        
        # 挡板移动
        for side in [0, 1]:
            if self.pad_y[side] + self.pad_speed[side] < 0:
                self.pad_y[side] = 0
            elif self.pad_y[side] + self.pad_speed[side] > self.WINDOW_HEIGHT - self.pad_height:
                self.pad_y[side] = self.WINDOW_HEIGHT - self.pad_height
            else:
                self.pad_y[side] += self.pad_speed[side]
        
        # 球移动            
        self.ball_x += self.ball_speed[0]
        self.ball_y += self.ball_speed[1]
        
        # 挡板反弹    
        if self.l_collision():
            self.ball_speed[0] = -self.ball_speed[0] * self.factor
            self.ball_speed[1] += self.pad_speed[0] * self.mu
        if self.r_collision():
            self.ball_speed[0] = -self.ball_speed[0] * self.factor
            self.ball_speed[1] += self.pad_speed[1] * self.mu
        
        # 边界碰撞检测
        if self.ball_y - self.ball_radius <= 0 or self.ball_y + self.ball_radius >= self.WINDOW_HEIGHT:
            self.ball_speed[1] = -self.ball_speed[1]
            if self.ball_y - self.ball_radius <= 0:
                self.ball_y = self.ball_radius
            else:
                self.ball_y = self.WINDOW_HEIGHT - self.ball_radius
            
        # 速度限制
        self.ball_speed[0] = self.cap(self.ball_speed[0], self.ball_maxspeed)
        self.ball_speed[1] = self.cap(self.ball_speed[1], self.ball_maxspeed)
        
        # 分数更新
        if self.ball_x < 0:
            self.right_score += 1
            self.reset()
        elif self.ball_x > self.WINDOW_WIDTH:
            self.left_score += 1
            self.reset()

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
        
        # 显示当前游戏模式和控制说明
        font_small = pygame.font.SysFont("simsun", 20)
        mode_text = font_small.render(f"当前模式: {self.mode_names[self.game_mode]}", True, self.WHITE)
        self.screen.blit(mode_text, (10, 10))
        
        # 控制说明
        controls = [
            "按键说明:",
            "M - 切换游戏模式",
            "R - 重置游戏",
            "ESC - 退出游戏"
        ]
        
        if self.game_mode == 0:  # 人机对战
            controls.extend([
                "左侧(人类): W/S 或 ↑/↓",
                "右侧(Bot_Q): 自动控制"
            ])
        elif self.game_mode == 1:  # Bot对战
            controls.extend([
                "左侧(Bot_Q): 自动控制",
                "右侧(Bot_0): 自动控制"
            ])
        elif self.game_mode == 2:  # 双人对战
            controls.extend([
                "左侧(玩家1): W/S",
                "右侧(玩家2): ↑/↓"
            ])
        
        for i, text in enumerate(controls):
            control_text = font_small.render(text, True, self.WHITE)
            self.screen.blit(control_text, (10, 40 + i*25))
        
        pygame.display.flip()
        self.clock.tick(60)

# 主程序
if __name__ == "__main__":
    game = PongBattle()
    
    print("=== Pong Battle 对战模式启动 ===")
    print("支持三种游戏模式：")
    print("0: 人机对战(人vs Bot_Q)")
    print("1: Bot对战(Bot_Q vs Bot_0)")
    print("2: 双人对战(人vs人)")
    print("\n按M键切换模式，按R键重置游戏")
    print("确保已有训练好的Q表文件：q_table-v5.pkl 和 q_table-v5-left.pkl")
    
    while game.running:
        left_action = None
        right_action = None
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game.running = False
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    game.switch_mode()
                elif event.key == pygame.K_r:
                    game.reset()
                    print("游戏已重置")
                elif event.key == pygame.K_ESCAPE:
                    game.running = False
                    sys.exit()
        
        # 处理持续按键
        keys = pygame.key.get_pressed()
        
        # 左侧控制（在人机对战和双人对战模式下）
        if game.game_mode in [0, 2]:
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                left_action = 1  # 向上
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                left_action = -1  # 向下
            else:
                left_action = 0  # 停止
        
        # 右侧控制（仅在双人对战模式下）
        if game.game_mode == 2:
            if keys[pygame.K_UP]:
                right_action = 1  # 向上
            elif keys[pygame.K_DOWN]:
                right_action = -1  # 向下
            else:
                right_action = 0  # 停止
        
        game.step(left_action, right_action)
        game.render()
            
    pygame.quit()