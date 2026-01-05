"""
Training Script cho DeepArrow DDQN Agent
Bao gồm Tensorboard logging và model checkpointing
"""
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from tqdm import tqdm


class Trainer:
    """
    Training Manager cho DDQN Agent
    
    Features:
    - Tensorboard logging
    - Model checkpointing (save best model)
    - Progress tracking với tqdm
    - Evaluation episodes
    """
    
    def __init__(
        self,
        env,
        agent,
        num_episodes=5000,
        eval_freq=100,
        eval_episodes=10,
        save_dir='./checkpoints',
        log_dir='./runs'
    ):
        self.env = env
        self.agent = agent
        self.num_episodes = num_episodes
        self.eval_freq = eval_freq
        self.eval_episodes = eval_episodes
        
        # Setup directories
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Tensorboard
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.writer = SummaryWriter(os.path.join(log_dir, f'ddqn_{timestamp}'))
        
        # Tracking
        self.best_eval_reward = -np.inf
        self.episode_rewards = []
        self.episode_lengths = []
        self.eval_rewards = []
    
    def train(self):
        """Main training loop"""
        print("=" * 60)
        print("🎯 Starting DeepArrow DDQN Training")
        print("=" * 60)
        print(f"Episodes: {self.num_episodes}")
        print(f"Device: {self.agent.device}")
        print(f"Action Space: {self.agent.action_dim} discrete angles")
        print("=" * 60)
        
        global_step = 0
        
        for episode in tqdm(range(self.num_episodes), desc="Training"):
            # Run one episode
            episode_reward, episode_length, episode_hits = self._run_episode(train=True)
            
            global_step += episode_length
            
            # Track metrics
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Log to tensorboard
            self.writer.add_scalar('Train/EpisodeReward', episode_reward, episode)
            self.writer.add_scalar('Train/EpisodeLength', episode_length, episode)
            self.writer.add_scalar('Train/TargetsHit', episode_hits, episode)
            self.writer.add_scalar('Train/Epsilon', self.agent.epsilon, episode)
            
            # Log average loss
            if len(self.agent.losses) > 0:
                avg_loss = np.mean(self.agent.losses[-episode_length:])
                self.writer.add_scalar('Train/Loss', avg_loss, episode)
            
            # Evaluation
            if (episode + 1) % self.eval_freq == 0:
                eval_reward, eval_hits = self._evaluate()
                self.eval_rewards.append(eval_reward)
                
                self.writer.add_scalar('Eval/MeanReward', eval_reward, episode)
                self.writer.add_scalar('Eval/MeanHits', eval_hits, episode)
                
                print(f"\n[Episode {episode + 1}] Eval Reward: {eval_reward:.2f}, Hits: {eval_hits:.2f}, Epsilon: {self.agent.epsilon:.3f}")
                
                # Save best model
                if eval_reward > self.best_eval_reward:
                    self.best_eval_reward = eval_reward
                    self.save_checkpoint('model_best.pth', episode, eval_reward)
                    print(f"✅ New best model! Eval reward: {eval_reward:.2f}")
            
            # Periodic checkpoint
            if (episode + 1) % 500 == 0:
                self.save_checkpoint(f'model_ep{episode + 1}.pth', episode, episode_reward)
        
        # Final save
        self.save_checkpoint('model_final.pth', self.num_episodes, self.episode_rewards[-1])
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print(f"Best Eval Reward: {self.best_eval_reward:.2f}")
        print(f"Final Epsilon: {self.agent.epsilon:.4f}")
        print("=" * 60)
        
        self.writer.close()
    
    def _run_episode(self, train=True):
        """
        Chạy 1 episode
        
        Args:
            train: Nếu True, thực hiện training steps
        
        Returns:
            total_reward, episode_length, total_hits
        """
        state, _ = self.env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        total_hits = 0
        
        while not done:
            # Select action
            action = self.agent.select_action(state, eval_mode=not train)
            
            # Take step
            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            
            # Store transition và train
            if train:
                self.agent.store_transition(state, action, reward, next_state, done)
                self.agent.train_step()
            
            # Update state
            state = next_state
            total_reward += reward
            episode_length += 1
            
            # Track hits
            step_info = info.get('step_info', {})
            total_hits += step_info.get('targets_hit', 0)
        
        return total_reward, episode_length, total_hits
    
    def _evaluate(self):
        """
        Evaluate agent trên nhiều episodes
        
        Returns:
            mean_reward, mean_hits
        """
        eval_rewards = []
        eval_hits = []
        
        for _ in range(self.eval_episodes):
            reward, _, hits = self._run_episode(train=False)
            eval_rewards.append(reward)
            eval_hits.append(hits)
        
        return np.mean(eval_rewards), np.mean(eval_hits)
    
    def save_checkpoint(self, filename, episode, reward):
        """Save model checkpoint"""
        path = os.path.join(self.save_dir, filename)
        self.agent.save(path)
        
        # Save training metadata
        metadata_path = path.replace('.pth', '_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write(f"Episode: {episode}\n")
            f.write(f"Reward: {reward:.2f}\n")
            f.write(f"Epsilon: {self.agent.epsilon:.4f}\n")
            f.write(f"Steps: {self.agent.steps}\n")


def main():
    """Main function để chạy training"""
    # Import dependencies
    import sys
    sys.path.append('.')  # Add current dir to path
    
    from agents.ddqn_agent import DDQNAgent, get_device
    from envs.wrapper import make_deeparrow_env
    
    # Hyperparameters
    CONFIG = {
        'num_episodes': 200,
        'num_angles': 90,  # 90 hoặc 180
        'lr': 1e-4,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'buffer_size': 100000,
        'batch_size': 64,
        'target_update_freq': 1000,
        'eval_freq': 100,
        'eval_episodes': 10
    }
    
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Create environment
    env = make_deeparrow_env(render_mode=None, num_angles=CONFIG['num_angles'])
    
    # Create agent
    device = get_device()
    agent = DDQNAgent(
        state_dim=10,  # From wrapper
        action_dim=CONFIG['num_angles'],
        lr=CONFIG['lr'],
        gamma=CONFIG['gamma'],
        epsilon_start=CONFIG['epsilon_start'],
        epsilon_end=CONFIG['epsilon_end'],
        epsilon_decay=CONFIG['epsilon_decay'],
        buffer_size=CONFIG['buffer_size'],
        batch_size=CONFIG['batch_size'],
        target_update_freq=CONFIG['target_update_freq'],
        device=device
    )
    
    # Create trainer
    trainer = Trainer(
        env=env,
        agent=agent,
        num_episodes=CONFIG['num_episodes'],
        eval_freq=CONFIG['eval_freq'],
        eval_episodes=CONFIG['eval_episodes']
    )
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        trainer.save_checkpoint('model_interrupted.pth', len(trainer.episode_rewards), 
                              trainer.episode_rewards[-1] if trainer.episode_rewards else 0)
    
    print("\n🎉 Done!")


if __name__ == '__main__':
    main()