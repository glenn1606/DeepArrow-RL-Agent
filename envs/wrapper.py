"""
DeepArrow Environment Wrapper
Chuyển đổi Dict observation thành Vector và thêm Reward shaping
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Tuple, Any


class DeepArrowWrapper(gym.Wrapper):
    """
    Wrapper chính cho môi trường ArrowShooting
    - Chuyển Dict observation thành flat vector với physics-informed features
    - Thêm reward shaping dựa trên kết quả bắn
    - Rời rạc hóa action space cho DQN
    """
    
    def __init__(self, env, num_angle_actions=90, fixed_power=50.0):
        super().__init__(env)
        
        self.num_angle_actions = num_angle_actions
        self.fixed_power = fixed_power
        
        # Định nghĩa action space rời rạc (0, 1, 2, ..., num_angle_actions-1)
        self.action_space = gym.spaces.Discrete(num_angle_actions)
        
        # Observation space: vector phẳng với các features vật lý
        # Features: [rel_x, rel_y, wind_x, wind_y, target_vel_x, target_vel_y, 
        #            theoretical_angle, mana_ratio, time_ratio, arrows_left]
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10,),
            dtype=np.float32
        )
        
        # Tracking để tính reward shaping
        self.last_min_distance = None
        self.episode_hits = 0
        
    def reset(self, **kwargs):
        """Reset environment và tracking variables"""
        obs, info = self.env.reset(**kwargs)
        self.last_min_distance = None
        self.episode_hits = 0
        return self._transform_observation(obs), info
    
    def step(self, action: int):
        """
        Thực hiện action và tính reward
        
        Args:
            action: Index của góc bắn (0 -> num_angle_actions-1)
        
        Returns:
            obs, reward, terminated, truncated, info
        """
        # Chuyển action index thành [angle, power, shoot]
        angle = self._discrete_to_angle(action)
        continuous_action = [angle, self.fixed_power, 1.0]  # Luôn bắn
        
        # Thực hiện action trong môi trường gốc
        obs_dict, terminated, truncated, info = self.env.step(continuous_action)
        
        # Tính reward dựa trên kết quả
        reward = self._compute_reward(info)
        
        # Chuyển observation thành vector
        obs_vector = self._transform_observation(obs_dict)
        
        return obs_vector, reward, terminated, truncated, info
    
    def _discrete_to_angle(self, action: int) -> float:
        """Chuyển action index thành góc (0-90 độ)"""
        return (action / (self.num_angle_actions - 1)) * 90.0
    
    def _transform_observation(self, obs_dict: Dict) -> np.ndarray:
        """
        Chuyển Dict observation thành Vector với physics-informed features
        
        Features engineering:
        1. Relative position (rel_x, rel_y): Khoảng cách tới target
        2. Wind (wind_x, wind_y): Vận tốc gió
        3. Target velocity (target_vel_x, target_vel_y): Tốc độ mục tiêu
        4. Theoretical angle: Góc bắn lý thuyết không có gió
        5. Resource ratios: mana, time, arrows
        """
        player_pos = obs_dict['player']
        wind = obs_dict['wind']
        resources = obs_dict['resources']
        targets = obs_dict['targets']
        
        # Nếu không có target, trả về vector zero
        if len(targets) == 0:
            return np.zeros(10, dtype=np.float32)
        
        # Lấy target đầu tiên (hoặc target gần nhất)
        target = targets[0]
        target_pos = target['pos']
        target_vel = target['vel']
        
        # 1. Relative position
        rel_x = target_pos['x'] - player_pos['x']
        rel_y = target_pos['y'] - player_pos['y']
        
        # 2. Wind
        wind_x = wind['x']
        wind_y = wind['y']
        
        # 3. Target velocity
        target_vel_x = target_vel['x']
        target_vel_y = target_vel['y']
        
        # 4. Theoretical angle (công thức ném xiên)
        # Bỏ qua gió, chỉ tính với gravity
        theoretical_angle = self._compute_theoretical_angle(rel_x, rel_y)
        
        # 5. Resource ratios
        mana_ratio = resources['mana'] / 100.0  # Normalize 0-1
        time_ratio = resources['time_left'] / 600.0  # Giả sử max time = 600
        arrows_left = resources['arrows_left'] / 20.0  # Normalize
        
        # Tạo feature vector
        features = np.array([
            rel_x / 800.0,  # Normalize theo kích thước màn hình
            rel_y / 600.0,
            wind_x / 5.0,   # Normalize theo max wind
            wind_y / 5.0,
            target_vel_x / 5.0,
            target_vel_y / 5.0,
            theoretical_angle / 90.0,  # Normalize 0-1
            mana_ratio,
            time_ratio,
            arrows_left
        ], dtype=np.float32)
        
        return features
    
    def _compute_theoretical_angle(self, dx: float, dy: float) -> float:
        """
        Tính góc bắn lý thuyết (không có gió)
        Sử dụng công thức ném xiên cơ bản
        
        Công thức đơn giản hóa: angle ≈ arctan(dy/dx) với điều chỉnh
        """
        if dx <= 0:
            return 45.0  # Default angle nếu target ở phía sau
        
        # Góc đơn giản dựa trên khoảng cách
        distance = np.sqrt(dx**2 + dy**2)
        if distance < 1:
            return 45.0
        
        # Tính góc với điều chỉnh cho gravity
        # Đây là xấp xỉ, có thể tinh chỉnh thêm
        angle = np.degrees(np.arctan2(dy + distance * 0.1, dx))
        
        # Clamp trong khoảng [0, 90]
        angle = np.clip(angle, 0, 90)
        
        return angle
    
    def _compute_reward(self, info: Dict) -> float:
        """
        Reward shaping:
        - +100 cho mỗi target bắn trúng
        - -5 cho mỗi mũi tên miss (ra ngoài màn hình)
        - +reward shaping dựa trên khoảng cách gần nhất
        """
        reward = 0.0
        
        step_info = info.get('step_info', {})
        
        # Reward chính: Hit target
        targets_hit = step_info.get('targets_hit', 0)
        if targets_hit > 0:
            reward += 100.0 * targets_hit
            self.episode_hits += targets_hit
        
        # Penalty: Miss (arrow ra ngoài)
        arrows_went_out = step_info.get('arrows_went_out', 0)
        if arrows_went_out > 0:
            reward -= 5.0 * arrows_went_out
        
        # Reward shaping: Khuyến khích bắn gần target
        min_distance = step_info.get('min_distance_to_target', None)
        if min_distance is not None and min_distance < 100:  # Chỉ reward nếu gần
            # Thưởng điểm nhỏ nếu arrow đi gần target
            proximity_reward = (100 - min_distance) / 100.0  # 0 to 1
            reward += proximity_reward * 0.5  # Scale nhỏ để không át reward chính
        
        # Small negative reward mỗi step (khuyến khích giải quyết nhanh)
        reward -= 0.01
        
        return reward


class HybridControlWrapper(gym.Wrapper):
    """
    Wrapper để implement Hybrid Control Strategy
    AI quyết định góc bắn, Rule-based quyết định khi nào bắn
    """
    
    def __init__(self, env, min_mana_threshold=30):
        super().__init__(env)
        self.min_mana_threshold = min_mana_threshold
    
    def step(self, action):
        """
        Override step để thêm logic rule-based cho việc bắn
        
        Args:
            action: Góc bắn từ AI (hoặc action index nếu dùng với DeepArrowWrapper)
        """
        # Lấy observation hiện tại để check resources
        # Note: Cần modify để access current state
        
        # Logic: Chỉ bắn khi có đủ mana
        # Đây là placeholder, cần integrate với agent logic
        
        return self.env.step(action)


# Utility function để tạo wrapped environment
def make_deeparrow_env(render_mode=None, num_angles=90):
    """
    Factory function để tạo môi trường đã wrap
    
    Args:
        render_mode: 'human' hoặc None
        num_angles: Số lượng góc rời rạc (90 hoặc 180)
    
    Returns:
        Wrapped environment sẵn sàng để train
    """
    # Import environment (cần đảm bảo arrow_env.py trong path)
    from envs.arrow_env import ArrowShootingEnv
    
    # Tạo base environment
    env = ArrowShootingEnv(render_mode=render_mode)
    
    # Wrap với DeepArrowWrapper
    env = DeepArrowWrapper(env, num_angle_actions=num_angles)
    
    return env