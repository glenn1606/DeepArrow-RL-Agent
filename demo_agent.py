import time
import torch
import numpy as np
import os
import argparse
import sys
import pygame

# Thêm thư mục hiện tại vào path
sys.path.append('.')

from envs.wrapper import make_deeparrow_env
from agents.ddqn_agent import DDQNAgent, get_device

def watch(model_path, num_episodes=5, fps=60):
    """
    Load model và xem agent chơi (Visual Mode)
    """
    
    # 1. Kiểm tra model
    if not os.path.exists(model_path):
        print(f"Lỗi: Không tìm thấy file model tại '{model_path}'")
        print(f"Vui lòng kiểm tra lại đường dẫn. File mặc định là 'checkpoints/model_best.pth'")
        return

    print(f"Đang chuẩn bị môi trường...")
    print(f"Loading model: {model_path}")

    # 2. Cấu hình (Phải khớp với CONFIG trong train.py)
    NUM_ANGLES = 90  
    STATE_DIM = 10   
    
    # 3. Khởi tạo môi trường với render_mode='human' (QUAN TRỌNG)
    env = make_deeparrow_env(render_mode='human', num_angles=NUM_ANGLES)
    
    # 4. Khởi tạo Agent
    device = get_device()
    agent = DDQNAgent(
        state_dim=STATE_DIM,
        action_dim=NUM_ANGLES,
        device=device
    )
    
    # 5. Load weights
    try:
        agent.load(model_path)
        agent.policy_net.eval() # Chuyển sang chế độ đánh giá (tắt dropout/batchnorm nếu có)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Lỗi khi load model (Format file có đúng không?): {e}")
        env.close()
        return

    print("\n BẮT ĐẦU GIẢ LẬP!")
    print("Nhấn ESC trên cửa sổ game để thoát ngay lập tức.")
    print("=" * 60)

    # 6. Vòng lặp chơi
    for ep in range(num_episodes):
        obs, info = env.reset()

        env.render()

        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # --- Xử lý sự kiện Pygame (Chống treo cửa sổ) ---
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print(" Người dùng đã hủy bỏ.")
                        env.close()
                        return

            # --- Agent hành động ---
            # eval_mode=True: Agent chọn action tốt nhất (không random)
            action = agent.select_action(obs, eval_mode=True)
            
            # Step môi trường
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Render frame mới
            env.render()
            
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
            # --- Điều chỉnh tốc độ (FPS) ---
            # Nếu chạy quá nhanh, mắt không nhìn kịp
            if hasattr(env.unwrapped, 'clock'):
                env.unwrapped.clock.tick(fps)
            else:
                time.sleep(1/fps)

        # In kết quả sau mỗi episode
        print(f"Episode {ep + 1}/{num_episodes}")
        print(f"  Score: {info.get('score', 0)}")
        print(f"  Targets Hit: {info.get('targets_hit', 0)}/{info.get('total_targets', 5)}")
        print(f"  Total Reward: {total_reward:.2f}")
        print("-" * 40)
        
        time.sleep(1) # Nghỉ 1 giây giữa các ván

    print("✅ Đã hoàn thành tất cả các lượt chơi.")
    env.close()

if __name__ == "__main__":
    # Thiết lập tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Watch DeepArrow Agent")
    parser.add_argument('--model', type=str, default='checkpoints/model_best.pth', 
                        help='Đường dẫn file .pth (Mặc định: checkpoints/model_best.pth)')
    parser.add_argument('--episodes', type=int, default=5, 
                        help='Số ván chơi (Mặc định: 5)')
    parser.add_argument('--fps', type=int, default=60, 
                        help='Tốc độ khung hình (Mặc định: 60)')
    
    args = parser.parse_args()
    
    watch(args.model, args.episodes, args.fps)