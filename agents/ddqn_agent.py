import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class QNetwork(nn.Module):
    """
    Neural Network để ước lượng Q-values
    Architecture: FC layers với ReLU activation
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(QNetwork, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, state):
        """Forward pass: state -> Q-values"""
        return self.network(state)


class ReplayBuffer:
    """
    Experience Replay Buffer để lưu trữ transitions
    Giúp break correlation giữa các sample liên tiếp
    """
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Thêm transition vào buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Lấy random batch từ buffer"""
        batch = random.sample(self.buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self):
        return len(self.buffer)


class DDQNAgent:
    """
    Double DQN Agent
    
    Features:
    - Target Network để stabilize training
    - Experience Replay
    - Epsilon-greedy exploration
    """
    
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=100000,
        batch_size=64,
        target_update_freq=1000,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        
        # Epsilon for exploration
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Policy Network (online network)
        self.policy_net = QNetwork(state_dim, action_dim).to(device)
        
        # Target Network (stabilize training)
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network không train
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Loss function
        self.criterion = nn.SmoothL1Loss()  # Huber loss, robust hơn MSE
        
        # Replay Buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Tracking
        self.steps = 0
        self.losses = []
    
    def select_action(self, state, eval_mode=False):
        """
        Chọn action với epsilon-greedy strategy
        
        Args:
            state: Current state
            eval_mode: Nếu True, không explore (dùng khi test)
        
        Returns:
            action: Index của action được chọn
        """
        # Evaluation mode: greedy
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t)
                action = q_values.argmax(dim=1).item()
        else:
            # Exploration: random action
            action = random.randrange(self.action_dim)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Lưu transition vào replay buffer"""
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Thực hiện 1 bước training
        
        Returns:
            loss: Loss value (để tracking)
        """
        # Chỉ train khi buffer đủ lớn
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch từ replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values: Q(s, a)
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Tách việc chọn action và đánh giá Q-value
        with torch.no_grad():
            # Policy network chọn best action cho next state
            next_actions = self.policy_net(next_states).argmax(dim=1)
            
            # Target network đánh giá Q-value của action đó
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Target Q-value: r + gamma * Q_target(s', a')
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping để tránh exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)
        
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        # Track loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def save(self, path):
        """Lưu model"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        print(f"Model loaded from {path}")


# Helper function
def get_device():
    """Kiểm tra GPU availability"""
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("Using CPU")
    return device