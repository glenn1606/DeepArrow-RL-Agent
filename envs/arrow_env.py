"""
ArrowShooting Deep Reinforcement Learning Environment
A Gymnasium-compatible environment for training RL agents to shoot arrows at moving targets.

Đặc biệt: Environment này KHÔNG có reward function built-in.
Sinh viên cần tự định nghĩa reward bằng cách sử dụng RewardWrapper hoặc custom wrapper.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any
import pygame
from dataclasses import dataclass, field


@dataclass
class Vector2:
    """2D Vector for position and velocity"""
    x: float
    y: float
    
    def __add__(self, other):
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __mul__(self, scalar):
        return Vector2(self.x * scalar, self.y * scalar)
    
    def distance_to(self, other):
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)


@dataclass
class Target:
    """Moving target that the agent must hit"""
    id: int
    pos: Vector2
    vel: Vector2
    active: bool = True
    score_value: int = 100
    radius: float = 20.0


@dataclass
class Arrow:
    """Arrow projectile with physics"""
    pos: Vector2
    vel: Vector2
    trajectory: list = field(default_factory=list)
    active: bool = True


class ArrowEnv(gym.Env):
    """
    Arrow Shooting Environment for Deep Reinforcement Learning
    
    MÔI TRƯỜNG HỌC TẬP:
    - Bắn mũi tên để trúng targets di chuyển
    - Quản lý mana để bắn
    - Chịu ảnh hưởng của gió và trọng lực
    
    Action Space:
        Box(3): [angle, power, shoot]
        - angle: -1 to 1 (maps to 0-90 degrees)
        - power: -1 to 1 (maps to 20-50 units)
        - shoot: >0 to shoot arrow
    
    Observation Space:
        Dictionary chứa:
        - player: {x, y}
        - wind: {x, y}
        - resources: {mana, time_left, arrows_left}
        - targets: [{pos: {x, y}, vel: {x, y}}, ...]
    
    QUAN TRỌNG:
        Environment này KHÔNG return reward!
        Sinh viên phải tự định nghĩa reward function bằng wrapper.
        Xem RewardWrapper trong file reward_wrappers.py
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    
    # ========== CONSTANTS ==========
    WORLD_WIDTH = 1800
    WORLD_HEIGHT = 900
    MAX_TARGETS = 5
    TARGET_SCORE = 100
    ARROW_LEFT_BONUS = 20
    TARGET_RADIUS = 20
    
    MANA_REGEN_RATE = 0.3
    SHOOT_COST = 30
    MAX_ARROWS = 20
    MAX_MANA = 100
    
    GRAVITY = 0.3
    WIND_MAX_STRENGTH = 0.2
    
    MAX_EPISODE_STEPS = 3000
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        Khởi tạo environment
        
        Args:
            render_mode: "human" để hiển thị, "rgb_array" để lấy pixels, None để không render
        """
        super().__init__()
        
        self.render_mode = render_mode
        
        # Define action space
        self.action_space = spaces.Box(
            low=np.array([0, 10, 0]),
            high=np.array([90, 50, 1]),
            dtype=np.float64
        )
        
        # Pygame rendering
        self.window = None
        self.clock = None
        self.font = None
        
        # Game state
        self.player_pos = Vector2(50, self.WORLD_HEIGHT - 50)
        self.wind = Vector2(0, 0)
        self.mana = self.MAX_MANA
        self.time_left = self.MAX_EPISODE_STEPS
        self.arrows_left = self.MAX_ARROWS
        self.targets = []
        self.arrows = []
        self.score = 0
        self.steps = 0
        self.terminated = False
        self.truncated = False
        
        # Stats để tracking
        self.episode_stats = {
            'targets_hit': 0,
            'arrows_missed': 0,
            'total_shots': 0
        }
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset environment về trạng thái ban đầu
        
        Returns:
            observation: Dict chứa state hiện tại
            info: Dict chứa metadata
        """
        super().reset(seed=seed)
        
        # Reset game state
        self.player_pos = Vector2(50, self.WORLD_HEIGHT - 50)
        self.wind = Vector2(
            (self.np_random.random() - 0.5) * self.WIND_MAX_STRENGTH * 2,
            (self.np_random.random() - 0.5) * self.WIND_MAX_STRENGTH * 2
        )
        self.mana = self.MAX_MANA
        self.time_left = self.MAX_EPISODE_STEPS
        self.arrows_left = self.MAX_ARROWS
        self.score = 0
        self.steps = 0
        self.terminated = False
        self.truncated = False
        self.arrows = []
        
        # Reset stats
        self.episode_stats = {
            'targets_hit': 0,
            'arrows_missed': 0,
            'total_shots': 0
        }
        
        # Create targets
        self.targets = []
        for i in range(self.MAX_TARGETS):
            target = Target(
                id=i,
                pos=Vector2(
                    600 + self.np_random.random() * 300,
                    100 + self.np_random.random() * 400
                ),
                vel=Vector2(0, (self.np_random.random() - 0.5) * 4),
                active=True
            )
            self.targets.append(target)
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], bool, bool, Dict[str, Any]]:
        """
        Thực hiện 1 bước trong environment
        
        Args:
            action: [angle, power, shoot]
        
        Returns:
            observation: Dict chứa state mới
            terminated: True nếu hoàn thành mục tiêu (all targets destroyed)
            truncated: True nếu hết thời gian hoặc hết arrows
            info: Dict chứa các thông tin để tính reward
        
        QUAN TRỌNG:
            Hàm này KHÔNG return reward!
            Tất cả thông tin cần thiết để tính reward nằm trong 'info'
        """
        # Check action validity
        assert self.action_space.contains(action), f"Action {action} không hợp lệ. Kiểm tra action_space định nghĩa trong init()."
        if self.terminated or self.truncated:
            raise RuntimeError("Episode đã kết thúc. Không thể step() tiếp. Vui lòng gọi reset() để bắt đầu lại.")

        # Track previous state for reward calculation
        prev_score = self.score
        prev_active_targets = sum(1 for t in self.targets if t.active)
        
        self.steps += 1
        self.time_left -= 1
        
        # 1. Regenerate mana
        self.mana = min(self.MAX_MANA, self.mana + self.MANA_REGEN_RATE)
        
        # 2. Handle shooting
        angle_action, power_action, shoot_action = action
        shot_fired = False
        
        if shoot_action > 0 and self.arrows_left > 0 and self.mana >= self.SHOOT_COST:
            self.mana -= self.SHOOT_COST
            self.arrows_left -= 1
            self.episode_stats['total_shots'] += 1
            shot_fired = True
            
            # Convert action to physics parameters
            angle_deg = angle_action  # 0 to 90 degrees
            angle_rad = np.deg2rad(angle_deg)
            power = power_action # 20 to 50 units
            
            # Create arrow
            new_arrow = Arrow(
                pos=Vector2(self.player_pos.x, self.player_pos.y),
                vel=Vector2(
                    np.cos(angle_rad) * power,
                    -np.sin(angle_rad) * power
                ),
                trajectory=[Vector2(self.player_pos.x, self.player_pos.y)]
            )
            self.arrows.append(new_arrow)
        # Update wind physics
        self.wind.x += (self.np_random.random() - 0.5) * 0.014
        self.wind.y += (self.np_random.random() - 0.5) * 0.013
        self.wind.x = np.clip(self.wind.x, -self.WIND_MAX_STRENGTH, self.WIND_MAX_STRENGTH)
        self.wind.y = np.clip(self.wind.y, -self.WIND_MAX_STRENGTH, self.WIND_MAX_STRENGTH)
        # 3. Update arrow physics
        arrows_went_out = 0
        for arrow in self.arrows:
            if not arrow.active:
                continue
            
            # Apply forces
            arrow.vel.x += self.wind.x
            arrow.vel.y += self.wind.y + self.GRAVITY
            
            # Update position
            arrow.pos.x += arrow.vel.x
            arrow.pos.y += arrow.vel.y
            arrow.trajectory.append(Vector2(arrow.pos.x, arrow.pos.y))
            
            # Check bounds
            if (arrow.pos.x < 0 or 
                arrow.pos.y > self.WORLD_HEIGHT ):
                arrow.active = False
                arrows_went_out += 1
        
        # 4. Update targets and check collisions
        targets_hit_this_step = 0
        
        for target in self.targets:
            if not target.active:
                continue
            
            # Move target
            target.pos.y += target.vel.y
            
            # Bounce off boundaries
            if target.pos.y < 50 or target.pos.y > self.WORLD_HEIGHT - 50:
                target.vel.y *= -1
            
            # Check collision with arrows
            for arrow in self.arrows:
                if not arrow.active:
                    continue
                
                dist = arrow.pos.distance_to(target.pos)
                if dist < target.radius:
                    target.active = False
                    arrow.active = False
                    self.score += self.TARGET_SCORE
                    self.episode_stats['targets_hit'] += 1
                    targets_hit_this_step += 1
                    break
        
        # Track missed arrows
        if arrows_went_out > 0:
            self.episode_stats['arrows_missed'] += arrows_went_out
        
        # 5. Cleanup old arrows (keep last 20)
        if len(self.arrows) > 20:
            self.arrows = self.arrows[-20:]
        
        # 6. Check termination conditions
        all_targets_down = all(not t.active for t in self.targets)
        if all_targets_down:
            self.terminated = True
            self.score += self.arrows_left * self.ARROW_LEFT_BONUS
        
        if self.time_left <= 0 or (self.episode_stats['total_shots'] == self.episode_stats['targets_hit'] + self.episode_stats['arrows_missed'] and self.arrows_left == 0):
            self.truncated = True
        
        
        # 7. Get observation and info
        observation = self._get_obs()
        info = self._get_info()

        if self.terminated or self.truncated:
            print(f"\n{'='*70}")
            print("EPISODE FINISHED!")
            print(f"{'='*70}")
            print(f"Final Score: {info['score']}")
            print(f"Targets Hit: {info['targets_hit']}/{info['total_targets']}")
            print(f"Accuracy: {info['targets_hit']}/{info['total_shots']} = "
                  f"{info['targets_hit']/max(1,info['total_shots'])*100:.1f}%")
            
            if self.terminated:
                print("Reason of termination: 🎯 All targets destroyed")
            else:
                print("Reason of termination: ⏱️ Time/Arrows ran out")
            
            print(f"\nPress R to restart or ESC to quit")
            print(f"{'='*70}")
        
        # Add step-specific info for reward calculation
        info['step_info'] = {
            'shot_fired': shot_fired,
            'targets_hit': targets_hit_this_step,
            'arrows_went_out': arrows_went_out,
            'score_gained': self.score - prev_score,
            'active_targets_before': prev_active_targets,
            'active_targets_after': sum(1 for t in self.targets if t.active),
        }
        
        return observation, self.terminated, self.truncated, info
    
    def _get_obs(self) -> Dict[str, Any]:
        """
        Tạo observation dict
        
        Returns:
            Dict chứa toàn bộ state của game
        """
        obs = {
            "player": {
                "x": self.player_pos.x,
                "y": self.player_pos.y
            },
            "wind": {
                "x": self.wind.x,
                "y": self.wind.y
            },
            "resources": {
                "mana": self.mana,
                "time_left": self.time_left,
                "arrows_left": self.arrows_left
            },
            "arrows": [],
            "targets": []
        }
        # Add active arrows
        for arrow in self.arrows:
            if arrow.active:
                obs["arrows"].append({
                    "pos": {"x": arrow.pos.x, "y": arrow.pos.y},
                    "vel": {"x": arrow.vel.x, "y": arrow.vel.y}
                })

        # Add active targets
        for target in self.targets:
            if target.active:
                obs["targets"].append({
                    "pos": {"x": target.pos.x, "y": target.pos.y},
                    "vel": {"x": target.vel.x, "y": target.vel.y}
                })
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """
        Tạo info dict với đầy đủ thông tin để tính reward
        
        Returns:
            Dict chứa:
            - score: Điểm hiện tại
            - mana: Mana hiện tại
            - time_left: Thời gian còn lại
            - arrows_left: Số arrows còn lại
            - active_targets: Số targets còn sống
            - targets_hit: Tổng số targets đã trúng
            - arrows_missed: Tổng số arrows bắn trượt
            - total_shots: Tổng số lần bắn
        """
        active_targets = sum(1 for t in self.targets if t.active)
        
        return {
            "score": self.score,
            "mana": self.mana,
            "time_left": self.time_left,
            "arrows_left": self.arrows_left,
            "total_targets": self.MAX_TARGETS,
            "active_targets": active_targets,
            "targets_hit": self.episode_stats['targets_hit'],
            "arrows_missed": self.episode_stats['arrows_missed'],
            "total_shots": self.episode_stats['total_shots'],
            "arrows_active": sum(1 for a in self.arrows if a.active),
        }
    
    def render(self):
        """Render environment"""
        if self.render_mode is None:
            return
        return self._render_frame()
    
    def _render_frame(self):
        """Render a single frame using pygame"""
        # Initialize pygame if needed
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.WORLD_WIDTH, self.WORLD_HEIGHT))
            pygame.display.set_caption("Arrow Shooting DRL - Student Environment")
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        if self.font is None:
            self.font = pygame.font.Font(None, 24)
            self.small_font = pygame.font.Font(None, 18)

        pygame.display.flip()
        
        # Clear screen
        self.window.fill((15, 23, 42))
        
        # Draw grid
        for x in range(0, self.WORLD_WIDTH, 50):
            pygame.draw.line(self.window, (30, 41, 59), (x, 0), (x, self.WORLD_HEIGHT))
        for y in range(0, self.WORLD_HEIGHT, 50):
            pygame.draw.line(self.window, (30, 41, 59), (0, y), (self.WORLD_WIDTH, y))
        
        # Draw ground
        pygame.draw.rect(self.window, (30, 41, 59), 
                        (0, self.WORLD_HEIGHT - 20, self.WORLD_WIDTH, 20))
        
        # Draw targets
        for target in self.targets:
            if not target.active:
                pygame.draw.circle(self.window, (239, 68, 68, 25),
                                 (int(target.pos.x), int(target.pos.y)),
                                 int(target.radius), 1)
                continue
            
            pygame.draw.circle(self.window, (239, 68, 68),
                             (int(target.pos.x), int(target.pos.y)), int(target.radius))
            pygame.draw.circle(self.window, (255, 255, 255),
                             (int(target.pos.x), int(target.pos.y)), int(target.radius * 0.6))
            pygame.draw.circle(self.window, (239, 68, 68),
                             (int(target.pos.x), int(target.pos.y)), int(target.radius * 0.2))
        
        # Draw arrows
        for arrow in self.arrows:
            if len(arrow.trajectory) > 1:
                color = (56, 189, 248) if arrow.active else (148, 163, 184)
                points = [(int(p.x), int(p.y)) for p in arrow.trajectory]
                if len(points) > 1:
                    pygame.draw.lines(self.window, color, False, points, 2)
            
            if arrow.active:
                angle = np.arctan2(arrow.vel.y, arrow.vel.x)
                tip = (int(arrow.pos.x), int(arrow.pos.y))
                tail = (int(arrow.pos.x - 15 * np.cos(angle)),
                       int(arrow.pos.y - 15 * np.sin(angle)))
                pygame.draw.line(self.window, (248, 250, 252), tip, tail, 3)
        
        # Draw player
        pygame.draw.circle(self.window, (56, 189, 248),
                         (int(self.player_pos.x), int(self.player_pos.y)), 15)
        pygame.draw.circle(self.window, (255, 255, 255),
                         (int(self.player_pos.x), int(self.player_pos.y)), 15, 2)
        
        # Draw HUD
        self._draw_hud()

        
        # Draw Episode Finished overlay
        if self.terminated or self.truncated:
            overlay = pygame.Surface((self.WORLD_WIDTH, self.WORLD_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 180))
            self.window.blit(overlay, (0, 0))
            
            msg_font = pygame.font.Font(None, 48)
            msg_text = "EPISODE FINISHED!"
            msg_surface = msg_font.render(msg_text, True, (255, 255, 255))
            msg_rect = msg_surface.get_rect(center=(self.WORLD_WIDTH // 2, self.WORLD_HEIGHT // 2 - 50))
            self.window.blit(msg_surface, msg_rect)
            
            subfont = pygame.font.Font(None, 32)
            reason_text = "All targets destroyed!" if self.terminated else "Time/Arrows ran out!"
            reason_surface = subfont.render(f"Reason: {reason_text}", True, (255, 255, 255))
            reason_rect = reason_surface.get_rect(center=(self.WORLD_WIDTH // 2, self.WORLD_HEIGHT // 2))
            self.window.blit(reason_surface, reason_rect)
            
            prompt_surface = subfont.render("Press R to restart or ESC to quit", True, (255, 255, 255))
            prompt_rect = prompt_surface.get_rect(center=(self.WORLD_WIDTH // 2, self.WORLD_HEIGHT // 2 + 50))
            self.window.blit(prompt_surface, prompt_rect)
            
        # pygame.display.flip()
        pygame.event.pump()
        self.clock.tick(self.metadata["render_fps"])
        
    def _draw_hud(self):
        """Draw HUD overlay"""
        # Mana bar
        mana_ratio = self.mana / self.MAX_MANA
        pygame.draw.rect(self.window, (51, 65, 85), (10, 10, 200, 20))
        pygame.draw.rect(self.window, (59, 130, 246), (10, 10, int(200 * mana_ratio), 20))
        mana_text = self.small_font.render(f"Mana: {int(self.mana)}/{self.MAX_MANA}", 
                                          True, (255, 255, 255))
        self.window.blit(mana_text, (15, 12))
        
        # Score
        score_text = self.font.render(f"Score: {self.score}", True, (74, 222, 128))
        self.window.blit(score_text, (self.WORLD_WIDTH - 150, 10))
        
        # Time
        time_color = (239, 68, 68) if self.time_left < 100 else (255, 255, 255)
        time_text = self.font.render(f"Time: {int(self.time_left)}", True, time_color)
        self.window.blit(time_text, (self.WORLD_WIDTH - 150, 40))
        
        # Targets
        active_targets = sum(1 for t in self.targets if t.active)
        target_text = self.font.render(f"Targets: {active_targets}/{self.MAX_TARGETS}", 
                                      True, (255, 255, 255))
        self.window.blit(target_text, (220, 10))
        
        # Arrows
        arrows_text = self.font.render(f"Arrows: {self.arrows_left}", True, (255, 255, 255))
        self.window.blit(arrows_text, (220, 40))
        
        # Wind
        wind_text = self.small_font.render(f"Wind: ({self.wind.x:.2f}, {self.wind.y:.2f})", 
                                          True, (200, 200, 200))
        self.window.blit(wind_text, (10, 40))
    
    def close(self):
        """Clean up resources"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
            self.font = None