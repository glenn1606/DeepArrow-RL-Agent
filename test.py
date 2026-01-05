"""
Test/Demo Script cho trained DDQN Agent
Visualize agent performance với Pygame
"""
import torch
import numpy as np
import argparse
import time
import gymnasium as gym

class AgentTester:
    """
    Test trained agent với visualization
    """
    
    def __init__(self, env, agent, render_delay=0.03):
        self.env = env
        self.agent = agent
        self.render_delay = render_delay
        
        # Statistics
        self.episode_stats = []
    
    def test_episodes(self, num_episodes=10, verbose=True):
        """
        Test agent trên nhiều episodes
        
        Args:
            num_episodes: Số episode để test
            verbose: In chi tiết
        
        Returns:
            Dict chứa statistics
        """
        print(f"\n{'='*60}")
        print(f"🎯 Testing Agent on {num_episodes} Episodes")
        print(f"{'='*60}\n")
        
        all_rewards = []
        all_hits = []
        all_lengths = []
        
        for ep in range(num_episodes):
            reward, length, hits, info = self._run_episode()
            
            all_rewards.append(reward)
            all_hits.append(hits)
            all_lengths.append(length)
            
            if verbose:
                print(f"Episode {ep+1:3d}: Reward={reward:7.2f}, Hits={hits:2d}, Length={length:3d}")
        
        # Calculate statistics
        stats = {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'mean_hits': np.mean(all_hits),
            'mean_length': np.mean(all_lengths),
            'total_hits': np.sum(all_hits)
        }
        
        print(f"\n{'='*60}")
        print("📊 Test Results:")
        print(f"{'='*60}")
        print(f"Mean Reward:  {stats['mean_reward']:7.2f} ± {stats['std_reward']:.2f}")
        print(f"Mean Hits:    {stats['mean_hits']:7.2f}")
        print(f"Total Hits:   {stats['total_hits']:7.0f}")
        print(f"Mean Length:  {stats['mean_length']:7.2f}")
        print(f"{'='*60}\n")
        
        return stats
    
    def _run_episode(self):
        """
        Chạy 1 episode với visualization
        
        Returns:
            total_reward, episode_length, total_hits, info
        """
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        total_hits = 0
        
        while not done:
            # Select action (greedy)
            action = self.agent.select_action(state, eval_mode=True)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Render if environment supports it
            if hasattr(self.env.unwrapped, 'render'):
                self.env.unwrapped.render()
                time.sleep(self.render_delay)
            
            # Update
            state = next_state
            total_reward += reward
            episode_length += 1
            
            # Track hits
            step_info = info.get('step_info', {})
            total_hits += step_info.get('targets_hit', 0)
        
        return total_reward, episode_length, total_hits, info
    
    def demo_interactive(self):
        """
        Demo interactive: người dùng có thể reset episode
        """
        print("\n🎮 Interactive Demo Mode")
        print("Press 'R' to reset episode, 'ESC' to quit")
        print("-" * 60)
        
        running = True
        
        while running:
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_hits = 0
            
            while not done and running:
                # Select action
                action = self.agent.select_action(state, eval_mode=True)
                
                # Take step
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                # Render
                if hasattr(self.env.unwrapped, 'render'):
                    self.env.unwrapped.render()
                
                # Update
                state = next_state
                episode_reward += reward
                
                step_info = info.get('step_info', {})
                episode_hits += step_info.get('targets_hit', 0)
                
                # Check for quit (này cần handle từ pygame events)
                # Placeholder logic
                time.sleep(self.render_delay)
            
            if done:
                print(f"Episode finished: Reward={episode_reward:.2f}, Hits={episode_hits}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test DeepArrow DDQN Agent')
    parser.add_argument('--model', type=str, default='checkpoints/model_best.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--episodes', type=int, default=10,
                       help='Number of test episodes')
    parser.add_argument('--render', action='store_true',
                       help='Enable rendering (Pygame visualization)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive demo mode')
    parser.add_argument('--num_angles', type=int, default=90,
                       help='Number of discrete angles (must match training)')
    
    args = parser.parse_args()
    
    # Import dependencies
    import sys
    sys.path.append('.')
    
    from agents.ddqn_agent import DDQNAgent, get_device
    from envs.wrapper import make_deeparrow_env
    
    # Setup
    device = get_device()
    render_mode = 'human' if args.render else None
    
    # Create environment
    print("Creating environment...")
    env = make_deeparrow_env(render_mode=render_mode, num_angles=args.num_angles)
    
    # Create agent
    print("Creating agent...")
    agent = DDQNAgent(
        state_dim=10,
        action_dim=args.num_angles,
        device=device
    )
    
    # Load model
    print(f"Loading model from {args.model}...")
    agent.load(args.model)
    agent.policy_net.eval()  # Set to evaluation mode
    
    # Create tester
    tester = AgentTester(env, agent)
    
    # Run test
    if args.interactive:
        tester.demo_interactive()
    else:
        tester.test_episodes(num_episodes=args.episodes)
    
    print("\n✅ Testing complete!")


if __name__ == '__main__':
    main()