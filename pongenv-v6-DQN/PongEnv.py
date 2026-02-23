import pygame
import numpy as np 
from datetime import datetime
import sys

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

class PongEnv:
    WINDOW_WIDTH, WINDOW_HEIGHT = 1070, 600
    
    def __init__(self):
        # 初始化Pygame
        pygame.init()
        pygame.display.set_caption("pong-DQN")
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
        self.switch = False
        self.bot0 = Bot_0(self)
        
        # 保存和历史记录
        self.last_step_episode = -1
        self.last_save_episode = -1
        # 改进5：修复重复记录问题，移除重复的历史记录变量
        self.history_recorded = False  # 标记是否已记录当前回合的历史
        
        # 重置游戏状态
        self.check_over = False
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
        
        # 挡板移动
        if self.switch: 
            if  self.pad_y[0] + self.pad_speed[0] < 0:
                self.pad_y[0] = 0
            elif self.pad_y[0] + self.pad_speed[0] > self.WINDOW_HEIGHT - self.pad_height:
                self.pad_y[0] = self.WINDOW_HEIGHT - self.pad_height
            else:
                self.pad_y[0] += self.pad_speed[0]
        else: 
            self.bot0.act(0)
           
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
        self.check_over = False 
        if self.ball_x < 0:
            self.right_score += 1
            self.check_over = True
        
        elif self.ball_x > self.WINDOW_WIDTH:
            self.left_score += 1
            self.check_over = True
            
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
        
        font = pygame.font.SysFont("consolas", 50)
        score_text = font.render(f"{self.left_score}   :   {self.right_score}", True, self.WHITE)
        self.screen.blit(score_text, (self.WINDOW_WIDTH//2 - score_text.get_width()//2, 50 - score_text.get_height()//2))
        
        # 在左上角绘制HUD文本（如果存在）
        if hasattr(self, "hud_lines") and self.hud_lines:
            try:
                font = pygame.font.SysFont(None, 22)
            except Exception:
                pygame.font.init()
                font = pygame.font.SysFont(None, 22)
            y = 10
            for line in self.hud_lines[:6]:
                text_surface = font.render(str(line), True, self.WHITE)
                self.screen.blit(text_surface, (10, y))
                y += 22
        
        pygame.display.flip()
        self.clock.tick(60)