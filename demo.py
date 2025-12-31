"""
Demo cho Arrow Shooting DRL Environment

Cho phép chơi manual hoặc xem agent hoạt động
"""

import numpy as np
import pygame
from envs.arrow_env import ArrowEnv
import time
from pprint import pprint, pformat


def manual_control_demo():
    """
    Manual control - chơi bằng keyboard

    """
    print("="*70)
    print("MANUAL CONTROL DEMO")
    print("="*70)
    print("\nControls:")
    print("  ↑/↓     : Adjust angle (góc bắn)")
    print("  ←/→     : Adjust power (lực bắn)")
    print("  SPACE   : Shoot arrow (bắn)")
    print("  R       : Reset environment")
    print("  ESC     : Quit")
    print("\nTips:")
    print("  - Góc từ 0° (ngang) đến 90° (thẳng đứng)")
    print("  - Power từ 10 đến 50")
    print("  - Cần đủ mana (≥30) để bắn")
    print("="*70)
    
    # Create environment
    env = ArrowEnv(render_mode="human")
    
    observation, info = env.reset()
    env.render()
    
    # Control state
    angle = 45  # degrees
    power = 10  # 
    
    episode_count = 0
    print(f"Episode {episode_count} started!")
    print(f"{'='*70}")
    clock = pygame.time.Clock()
    
    running = True
    while running:
        # Slow down a bit for playability
        time.sleep(0.016)  # ~60 FPS
        
        # Handle events
        shoot = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break
                
                if event.key == pygame.K_SPACE:
                    shoot = 1
                
                if event.key == pygame.K_r:
                    observation, info = env.reset()
                    episode_count += 1
                    print(f"\n{'='*70}")
                    print(f"Episode {episode_count} started!")
                    print(f"{'='*70}")
        
        if not running:
            break
        
        # Get continuous key state
        keys = pygame.key.get_pressed()
        
        # Adjust angle
        if keys[pygame.K_UP]:
            angle = min(90, angle + 1)
        if keys[pygame.K_DOWN]:
            angle = max(0, angle - 1)
        
        # Adjust power
        if keys[pygame.K_RIGHT]:
            power = min(50, power + 1)
        if keys[pygame.K_LEFT]:
            power = max(10, power - 1)
        
        # Create action
        action = np.array([angle, power, shoot], dtype=np.float32)
        
        try:
            observation, terminated, truncated, info = env.step(action)
        except RuntimeError as e:
            print(e)


        # print("Observation:")
        # pprint(observation, sort_dicts=False, width=120)
        # print("Info:")
        # pprint(info, sort_dicts=False, width=120)
        
        # Render
        env.render()
        
        # Draw manual controls
        _draw_manual_controls(env, angle, power)
        
        pygame.display.flip()
        clock.tick(60)
        
        # Check termination

    
    env.close()


def _draw_manual_controls(env, angle, power):
    """Draw control indicators on screen"""
    if env.window is None or not hasattr(env, 'small_font'):
        return
    
    font = pygame.font.Font(None, 20)
    
    # Calculate actual values
    angle_deg = angle
    power_val = power
    
    # Display angle and power
    y_offset = 70
    angle_text = font.render(f"Angle: {angle_deg:.1f}°", True, (56, 189, 248))
    power_text = font.render(f"Power: {power_val:.1f}", True, (16, 185, 129))
    
    env.window.blit(angle_text, (10, y_offset))
    env.window.blit(power_text, (10, y_offset + 25))
    
    # Draw aim line
    
    angle_rad = np.deg2rad(angle_deg)
    line_length = power_val * 4
    end_x = env.player_pos.x + np.cos(angle_rad) * line_length
    end_y = env.player_pos.y - np.sin(angle_rad) * line_length
    

    # Draw dashed aim line
    if env.mana >= env.SHOOT_COST and env.arrows_left > 0:
        pygame.draw.line(
            env.window,
            (255, 100, 100, 200),
            (int(env.player_pos.x), int(env.player_pos.y)),
            (int(end_x), int(end_y)),
            3
        )
    else: 
        pygame.draw.line(
            env.window,
            (150, 150, 150, 100),
            (int(env.player_pos.x), int(env.player_pos.y)),
            (int(end_x), int(end_y)),
            3
        )

def main():
    """Main menu"""
    print("\n" + "="*70)
    print("ARROW SHOOTING DRL ENVIRONMENT - DEMO")
    print("="*70)

    manual_control_demo()
    
if __name__ == "__main__":
    main()