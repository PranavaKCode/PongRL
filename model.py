"""
Pong RL v2.7
"""

import sys, os
sys.path.insert(0, os.path.expanduser('~/.local/lib/python3.12/site-packages'))

import torch
import torch.nn as nn
from torch.distributions import Categorical
import time
from dataclasses import dataclass
from typing import Optional, List
from collections import deque
import matplotlib.pyplot as plt
from IPython.display import clear_output
import warnings
warnings.filterwarnings('ignore')

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"AMP: {AMP_AVAILABLE}")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Config:
    # Environment
    num_envs: int = 8192
    width: int = 84
    height: int = 84
    paddle_height: int = 12
    paddle_width: int = 4
    ball_size: int = 3
    paddle_speed: float = 1.5
    ball_speed: float = 1.5
    max_ball_speed: float = 4.0
    
    # PPO Hyperparameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    # Training
    total_timesteps: int = 100_000_000
    rollout_steps: int = 128
    num_minibatches: int = 32
    update_epochs: int = 4
    
    # Self-play
    opponent_update_freq: int = 10
    latest_opponent_prob: float = 0.5
    max_opponent_pool: int = 10
    
    # Reward shaping
    distance_reward_scale: float = 2.0
    hit_reward: float = 1.0
    rally_reward_scale: float = 0.002
    score_reward: float = 5.0
    win_bonus: float = 10.0
    
    # Optimization
    use_mixed_precision: bool = True
    
    # Logging
    plot_interval: int = 10
    save_interval: int = 50
    
    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_steps
    
    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches


# ============================================================================
# NEURAL NETWORK
# ============================================================================

class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = 6, act_dim: int = 3, hidden: int = 256):
        super().__init__()
        
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)
        
        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
    
    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h).squeeze(-1)
    
    def get_action_and_value(self, x, action=None):
        logits, value = self(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ============================================================================
# PONG ENVIRONMENT WITH DISTANCE REWARD
# ============================================================================

class PongEnv:
    """
    Vectorized Pong environment with distance-based reward shaping.
    """
    
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.n = cfg.num_envs
        
        # Dimensions
        self.W = float(cfg.width)
        self.H = float(cfg.height)
        self.ph = float(cfg.paddle_height)
        self.pw = float(cfg.paddle_width)
        self.bs = float(cfg.ball_size)
        
        # Ball state
        self.ball_x = torch.zeros(self.n, device=device)
        self.ball_y = torch.zeros(self.n, device=device)
        self.ball_vx = torch.zeros(self.n, device=device)
        self.ball_vy = torch.zeros(self.n, device=device)
        
        # Paddle positions (y-coordinate of center)
        self.left_y = torch.zeros(self.n, device=device)
        self.right_y = torch.zeros(self.n, device=device)
        
        # For distance reward
        self.prev_dist = torch.zeros(self.n, device=device)
        
        # Game state
        self.rally = torch.zeros(self.n, device=device, dtype=torch.int32)
        self.score_left = torch.zeros(self.n, device=device, dtype=torch.int32)
        self.score_right = torch.zeros(self.n, device=device, dtype=torch.int32)
        self.ep_reward = torch.zeros(self.n, device=device)
        self.ep_length = torch.zeros(self.n, device=device, dtype=torch.int32)
        
        # Opponent model
        self.opponent = None
        
        # Initialize
        self.reset()
    
    def set_opponent(self, model: nn.Module):
        """Set the opponent model for self-play."""
        self.opponent = model
    
    def _reset_ball(self, idx: torch.Tensor, direction: Optional[torch.Tensor] = None):
        """Reset ball position and velocity for given environment indices."""
        m = idx.shape[0]
        if m == 0:
            return
        
        # Center position
        self.ball_x[idx] = self.W / 2
        self.ball_y[idx] = self.H / 2
        
        # Random angle (-45 to +45 degrees)
        angle = (torch.rand(m, device=self.device) - 0.5) * 1.5
        
        # Random or specified direction
        if direction is None:
            direction = torch.sign(torch.rand(m, device=self.device) - 0.5)
            direction = torch.where(direction == 0, torch.ones_like(direction), direction)
        
        speed = self.cfg.ball_speed
        self.ball_vx[idx] = direction * speed * torch.cos(angle).abs().clamp(min=0.5)
        self.ball_vy[idx] = speed * torch.sin(angle) * 0.5
        
        self.rally[idx] = 0
    
    def reset(self, idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset environments. If idx is None, reset all."""
        if idx is None:
            idx = torch.arange(self.n, device=self.device)
        
        self._reset_ball(idx)
        
        # Center paddles
        self.left_y[idx] = self.H / 2
        self.right_y[idx] = self.H / 2
        
        # Reset distance tracking
        self.prev_dist[idx] = torch.abs(self.right_y[idx] - self.ball_y[idx])
        
        # Reset scores
        self.score_left[idx] = 0
        self.score_right[idx] = 0
        self.ep_reward[idx] = 0
        self.ep_length[idx] = 0
        
        return self._get_obs()
    
    def _get_obs(self, flip: bool = False) -> torch.Tensor:
        """
        Get observation tensor.
        
        State: [ball_x, ball_y, ball_vx, ball_vy, opponent_y, self_y]
        All normalized to roughly [-1, 1]
        
        If flip=True, return from left paddle's perspective (for opponent).
        """
        if flip:
            # Opponent's view (left paddle) - flip x coordinates
            return torch.stack([
                (self.W - self.ball_x) / self.W * 2 - 1,  # Flipped ball x
                self.ball_y / self.H * 2 - 1,
                -self.ball_vx / self.cfg.max_ball_speed,  # Flipped velocity
                self.ball_vy / self.cfg.max_ball_speed,
                self.right_y / self.H * 2 - 1,            # Agent is opponent's opponent
                self.left_y / self.H * 2 - 1,             # Self
            ], dim=1)
        else:
            # Agent's view (right paddle)
            return torch.stack([
                self.ball_x / self.W * 2 - 1,
                self.ball_y / self.H * 2 - 1,
                self.ball_vx / self.cfg.max_ball_speed,
                self.ball_vy / self.cfg.max_ball_speed,
                self.left_y / self.H * 2 - 1,             # Opponent
                self.right_y / self.H * 2 - 1,            # Self
            ], dim=1)
    
    @torch.no_grad()
    def step(self, action: torch.Tensor):
        """
        Execute one step for all environments.
        
        Action: 0 = stay, 1 = up, 2 = down
        
        Returns: obs, reward, done, info
        """
        # Store previous distance for reward shaping
        self.prev_dist = torch.abs(self.right_y - self.ball_y)
        
        ps = self.cfg.paddle_speed
        
        # ===== OPPONENT (LEFT PADDLE) =====
        if self.opponent is not None:
            opp_obs = self._get_obs(flip=True)
            opp_logits, _ = self.opponent(opp_obs)
            opp_action = opp_logits.argmax(dim=1)
        else:
            # Simple tracking AI as fallback
            diff = self.ball_y - self.left_y
            opp_action = torch.zeros(self.n, device=self.device, dtype=torch.long)
            opp_action = torch.where(diff < -2, torch.ones_like(opp_action), opp_action)
            opp_action = torch.where(diff > 2, torch.full_like(opp_action, 2), opp_action)
        
        # Move opponent paddle
        opp_move = torch.zeros(self.n, device=self.device)
        opp_move = torch.where(opp_action == 1, -ps, opp_move)
        opp_move = torch.where(opp_action == 2, ps, opp_move)
        self.left_y = (self.left_y + opp_move).clamp(self.ph / 2, self.H - self.ph / 2)
        
        # ===== AGENT (RIGHT PADDLE) =====
        agent_move = torch.zeros(self.n, device=self.device)
        agent_move = torch.where(action == 1, -ps, agent_move)
        agent_move = torch.where(action == 2, ps, agent_move)
        self.right_y = (self.right_y + agent_move).clamp(self.ph / 2, self.H - self.ph / 2)
        
        # ===== BALL PHYSICS =====
        self.ball_x = self.ball_x + self.ball_vx
        self.ball_y = self.ball_y + self.ball_vy
        
        # Top/bottom wall collision
        hit_top = self.ball_y < self.bs
        hit_bottom = self.ball_y > self.H - self.bs
        hit_wall = hit_top | hit_bottom
        self.ball_vy = torch.where(hit_wall, -self.ball_vy, self.ball_vy)
        self.ball_y = self.ball_y.clamp(self.bs, self.H - self.bs)
        
        # ===== PADDLE COLLISIONS =====
        # Right paddle (agent)
        right_paddle_x = self.W - self.pw
        hit_right = (
            (self.ball_x >= right_paddle_x - self.bs) &
            (self.ball_vx > 0) &
            (torch.abs(self.ball_y - self.right_y) <= self.ph / 2 + self.bs)
        )
        
        # Left paddle (opponent)
        hit_left = (
            (self.ball_x <= self.pw + self.bs) &
            (self.ball_vx < 0) &
            (torch.abs(self.ball_y - self.left_y) <= self.ph / 2 + self.bs)
        )
        
        # Bounce off paddles
        hit_paddle = hit_right | hit_left
        self.ball_vx = torch.where(hit_paddle, -self.ball_vx * 1.02, self.ball_vx)
        
        # Add spin based on where ball hits paddle
        paddle_y = torch.where(hit_right, self.right_y, self.left_y)
        relative_hit = torch.where(
            hit_paddle,
            (self.ball_y - paddle_y) / (self.ph / 2),
            torch.zeros_like(self.ball_y)
        )
        self.ball_vy = self.ball_vy + relative_hit * 0.4
        
        # Keep ball in bounds after paddle hit (prevent tunneling)
        self.ball_x = torch.where(hit_right, right_paddle_x - self.bs - 0.5, self.ball_x)
        self.ball_x = torch.where(hit_left, self.pw + self.bs + 0.5, self.ball_x)
        
        # Clamp ball speed
        speed = torch.sqrt(self.ball_vx ** 2 + self.ball_vy ** 2)
        too_fast = speed > self.cfg.max_ball_speed
        scale = self.cfg.max_ball_speed / (speed + 1e-8)
        self.ball_vx = torch.where(too_fast, self.ball_vx * scale, self.ball_vx)
        self.ball_vy = torch.where(too_fast, self.ball_vy * scale, self.ball_vy)
        
        # Update rally counter
        self.rally = self.rally + 1
        
        # ===== REWARD CALCULATION =====
        # 1. Distance reward (from your simple version!)
        new_dist = torch.abs(self.right_y - self.ball_y)
        dist_reward = (self.prev_dist - new_dist) * self.cfg.distance_reward_scale
        
        reward = dist_reward
        
        # 2. Hit bonus + rally bonus
        rally_bonus = self.rally.float() * self.cfg.rally_reward_scale
        reward = torch.where(hit_right, reward + self.cfg.hit_reward + rally_bonus, reward)
        
        # 3. Scoring
        scored_left = self.ball_x < 0      # Opponent scored (agent missed)
        scored_right = self.ball_x > self.W  # Agent scored
        
        reward = torch.where(scored_right, reward + self.cfg.score_reward, reward)
        reward = torch.where(scored_left, reward - self.cfg.score_reward, reward)
        
        # Update scores
        self.score_left = self.score_left + scored_left.int()
        self.score_right = self.score_right + scored_right.int()
        
        # 4. Game over at 11 points
        done = (self.score_left >= 11) | (self.score_right >= 11)
        
        # Win/loss bonus
        reward = torch.where(
            done & (self.score_right > self.score_left),
            reward + self.cfg.win_bonus,
            reward
        )
        reward = torch.where(
            done & (self.score_left > self.score_right),
            reward - self.cfg.win_bonus,
            reward
        )
        
        # Track episode stats
        self.ep_reward = self.ep_reward + reward
        self.ep_length = self.ep_length + 1
        
        # ===== COLLECT INFO =====
        info = {
            'ep_rewards': self.ep_reward[done].tolist() if done.any() else [],
            'ep_lengths': self.ep_length[done].tolist() if done.any() else [],
            'wins': int((self.score_right[done] > self.score_left[done]).sum()) if done.any() else 0,
            'losses': int((self.score_left[done] > self.score_right[done]).sum()) if done.any() else 0,
            'avg_rally': float(self.rally.float().mean()),
            'avg_dist': float(new_dist.mean()),
        }
        
        # ===== RESET BALL AFTER SCORING =====
        scored = scored_left | scored_right
        if scored.any():
            scored_idx = torch.where(scored)[0]
            # Ball goes toward whoever just got scored on
            direction = torch.where(
                scored_right[scored_idx],
                -torch.ones(scored_idx.shape[0], device=self.device),
                torch.ones(scored_idx.shape[0], device=self.device)
            )
            self._reset_ball(scored_idx, direction)
        
        # ===== FULL RESET FOR FINISHED GAMES =====
        if done.any():
            self.reset(torch.where(done)[0])
        
        return self._get_obs(), reward, done, info
    
    def render(self, idx: int = 0):
        """Render a single environment to numpy array."""
        h, w = int(self.H), int(self.W)
        img = torch.zeros((h, w, 3), dtype=torch.uint8, device='cpu')
        
        # Center line
        for y in range(0, h, 4):
            img[y:min(y+2, h), w//2-1:w//2+1] = 50
        
        # Ball (white)
        bx = int(self.ball_x[idx].item())
        by = int(self.ball_y[idx].item())
        b = int(self.bs)
        y1, y2 = max(0, by - b), min(h, by + b)
        x1, x2 = max(0, bx - b), min(w, bx + b)
        img[y1:y2, x1:x2] = torch.tensor([255, 255, 255], dtype=torch.uint8)
        
        # Left paddle (blue)
        ly = int(self.left_y[idx].item())
        p = int(self.ph // 2)
        pw = int(self.pw)
        y1, y2 = max(0, ly - p), min(h, ly + p)
        img[y1:y2, 0:pw] = torch.tensor([100, 150, 255], dtype=torch.uint8)
        
        # Right paddle (green)
        ry = int(self.right_y[idx].item())
        y1, y2 = max(0, ry - p), min(h, ry + p)
        img[y1:y2, w - pw:w] = torch.tensor([100, 255, 150], dtype=torch.uint8)
        
        return img.numpy()


# ============================================================================
# PPO TRAINER
# ============================================================================

class PPOTrainer:
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = device
        
        print("Initializing trainer...")
        
        # Environment
        self.env = PongEnv(cfg, device)
        
        # Agent model
        self.model = ActorCritic().to(device)
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Opponent model (copy of agent)
        self.opponent = ActorCritic().to(device)
        self.opponent.load_state_dict(self.model.state_dict())
        self.opponent.eval()
        self.env.set_opponent(self.opponent)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.learning_rate)
        
        # Mixed precision
        self.use_amp = AMP_AVAILABLE and cfg.use_mixed_precision
        self.scaler = GradScaler() if self.use_amp else None
        print(f"  Mixed precision: {self.use_amp}")
        
        # Rollout buffers (pre-allocated)
        T, N = cfg.rollout_steps, cfg.num_envs
        self.buf_obs = torch.zeros((T, N, 6), device=device)
        self.buf_act = torch.zeros((T, N), device=device, dtype=torch.long)
        self.buf_rew = torch.zeros((T, N), device=device)
        self.buf_done = torch.zeros((T, N), device=device)
        self.buf_val = torch.zeros((T, N), device=device)
        self.buf_logp = torch.zeros((T, N), device=device)
        
        # Training stats
        self.global_step = 0
        self.num_updates = 0
        self.ep_rewards = deque(maxlen=500)
        self.win_history = deque(maxlen=50)
        
        # Logging
        self.log = {
            'step': [],
            'reward': [],
            'winrate': [],
            'rally': [],
            'dist': [],
            'entropy': [],
            'fps': [],
        }
        
        # Opponent pool for league training
        self.opponent_pool: List[dict] = []
        
        print("  Initialization complete!")
    
    def update_opponent(self):
        """Update opponent from current or historical policy."""
        # Save current policy to pool
        state_dict = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        if len(self.opponent_pool) < self.cfg.max_opponent_pool:
            self.opponent_pool.append(state_dict)
        else:
            # Replace oldest
            idx = self.num_updates % self.cfg.max_opponent_pool
            self.opponent_pool[idx] = state_dict
        
        # Select opponent: latest vs random from pool
        if torch.rand(1).item() < self.cfg.latest_opponent_prob or len(self.opponent_pool) < 3:
            # Use latest policy
            self.opponent.load_state_dict(self.model.state_dict())
        else:
            # Use random historical policy
            idx = torch.randint(0, len(self.opponent_pool), (1,)).item()
            self.opponent.load_state_dict(self.opponent_pool[idx])
        
        print(f"      🔄 Opponent updated (pool size: {len(self.opponent_pool)})")
    
    @torch.no_grad()
    def collect_rollout(self):
        """Collect experience by running policy in environment."""
        self.model.eval()
        obs = self.env._get_obs()
        
        wins, losses = 0, 0
        rallies = []
        distances = []
        
        for t in range(self.cfg.rollout_steps):
            self.buf_obs[t] = obs
            
            # Get action from policy
            if self.use_amp:
                with autocast():
                    action, log_prob, _, value = self.model.get_action_and_value(obs)
            else:
                action, log_prob, _, value = self.model.get_action_and_value(obs)
            
            self.buf_act[t] = action
            self.buf_logp[t] = log_prob
            self.buf_val[t] = value
            
            # Step environment
            obs, reward, done, info = self.env.step(action)
            
            self.buf_rew[t] = reward
            self.buf_done[t] = done.float()
            
            # Track stats
            self.ep_rewards.extend(info['ep_rewards'])
            wins += info['wins']
            losses += info['losses']
            rallies.append(info['avg_rally'])
            distances.append(info['avg_dist'])
            
            self.global_step += self.cfg.num_envs
        
        # Compute GAE
        if self.use_amp:
            with autocast():
                _, _, _, next_value = self.model.get_action_and_value(obs)
        else:
            _, _, _, next_value = self.model.get_action_and_value(obs)
        
        advantages = torch.zeros_like(self.buf_rew)
        last_gae = 0
        
        for t in reversed(range(self.cfg.rollout_steps)):
            if t == self.cfg.rollout_steps - 1:
                next_val = next_value
            else:
                next_val = self.buf_val[t + 1]
            
            non_terminal = 1.0 - self.buf_done[t]
            delta = self.buf_rew[t] + self.cfg.gamma * next_val * non_terminal - self.buf_val[t]
            last_gae = delta + self.cfg.gamma * self.cfg.gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        
        returns = advantages + self.buf_val
        
        # Track win rate
        if wins + losses > 0:
            self.win_history.append(wins / (wins + losses))
        
        return {
            'advantages': advantages,
            'returns': returns,
            'avg_rally': sum(rallies) / len(rallies),
            'avg_dist': sum(distances) / len(distances),
            'wins': wins,
            'losses': losses,
        }
    
    def update(self, advantages: torch.Tensor, returns: torch.Tensor):
        """Perform PPO update."""
        self.model.train()
        
        # Flatten buffers
        B = self.cfg.batch_size
        b_obs = self.buf_obs.reshape(B, 6)
        b_act = self.buf_act.reshape(B)
        b_logp = self.buf_logp.reshape(B)
        b_adv = advantages.reshape(B)
        b_ret = returns.reshape(B)
        
        # Normalize advantages
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        
        total_entropy = 0
        count = 0
        
        for epoch in range(self.cfg.update_epochs):
            # Shuffle indices
            perm = torch.randperm(B, device=self.device)
            
            for start in range(0, B, self.cfg.minibatch_size):
                idx = perm[start:start + self.cfg.minibatch_size]
                
                if self.use_amp:
                    with autocast():
                        _, new_logp, entropy, new_val = self.model.get_action_and_value(
                            b_obs[idx], b_act[idx]
                        )
                        
                        # Policy loss
                        ratio = (new_logp - b_logp[idx]).exp()
                        pg_loss1 = -b_adv[idx] * ratio
                        pg_loss2 = -b_adv[idx] * ratio.clamp(
                            1 - self.cfg.clip_epsilon,
                            1 + self.cfg.clip_epsilon
                        )
                        pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                        
                        # Value loss
                        val_loss = 0.5 * ((new_val - b_ret[idx]) ** 2).mean()
                        
                        # Entropy bonus
                        ent_loss = entropy.mean()
                        
                        # Total loss
                        loss = pg_loss + self.cfg.value_coef * val_loss - self.cfg.entropy_coef * ent_loss
                    
                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    _, new_logp, entropy, new_val = self.model.get_action_and_value(
                        b_obs[idx], b_act[idx]
                    )
                    
                    ratio = (new_logp - b_logp[idx]).exp()
                    pg_loss1 = -b_adv[idx] * ratio
                    pg_loss2 = -b_adv[idx] * ratio.clamp(
                        1 - self.cfg.clip_epsilon,
                        1 + self.cfg.clip_epsilon
                    )
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    
                    val_loss = 0.5 * ((new_val - b_ret[idx]) ** 2).mean()
                    ent_loss = entropy.mean()
                    
                    loss = pg_loss + self.cfg.value_coef * val_loss - self.cfg.entropy_coef * ent_loss
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                    self.optimizer.step()
                
                total_entropy += ent_loss.item()
                count += 1
        
        self.num_updates += 1
        return total_entropy / count
    
    def train(self):
        """Main training loop."""
        total_updates = self.cfg.total_timesteps // self.cfg.batch_size
        
        print(f"\n{'='*65}")
        print(f"🎮 Pong Self-Play Training (with Distance Reward)")
        print(f"{'='*65}")
        print(f"  Device:          {self.device}")
        print(f"  Environments:    {self.cfg.num_envs:,}")
        print(f"  Batch size:      {self.cfg.batch_size:,}")
        print(f"  Total updates:   {total_updates}")
        print(f"  Total steps:     {self.cfg.total_timesteps:,}")
        print(f"  Opponent update: every {self.cfg.opponent_update_freq} updates")
        print(f"  Distance reward: {self.cfg.distance_reward_scale}")
        print(f"  Rally bonus:     {self.cfg.rally_reward_scale}")
        print(f"{'='*65}\n")
        
        start_time = time.time()
        
        for update in range(1, total_updates + 1):
            update_start = time.time()
            
            # Collect experience
            rollout = self.collect_rollout()
            
            # Update policy
            entropy = self.update(rollout['advantages'], rollout['returns'])
            
            # Update opponent periodically
            if update % self.cfg.opponent_update_freq == 0:
                self.update_opponent()
            
            # Calculate stats
            update_time = time.time() - update_start
            fps = self.cfg.batch_size / update_time
            
            avg_reward = sum(self.ep_rewards) / max(1, len(self.ep_rewards))
            winrate = sum(self.win_history) / max(1, len(self.win_history)) * 100
            
            # Log
            self.log['step'].append(self.global_step)
            self.log['reward'].append(avg_reward)
            self.log['winrate'].append(winrate)
            self.log['rally'].append(rollout['avg_rally'])
            self.log['dist'].append(rollout['avg_dist'])
            self.log['entropy'].append(entropy)
            self.log['fps'].append(fps)
            
            # Print progress
            elapsed = time.time() - start_time
            eta = elapsed / update * (total_updates - update)
            
            print(f"[{update:3d}/{total_updates}] "
                  f"Step {self.global_step:>10,} | "
                  f"FPS {fps:>9,.0f} | "
                  f"Rew {avg_reward:>7.2f} | "
                  f"Win {winrate:>5.1f}% | "
                  f"Rally {rollout['avg_rally']:>5.1f} | "
                  f"Dist {rollout['avg_dist']:>4.1f} | "
                  f"Ent {entropy:.3f} | "
                  f"ETA {eta/60:.1f}m")
            
            # Plot
            if update % self.cfg.plot_interval == 0:
                clear_output(wait=True)
                self._plot()
                print(f"\n📊 Update {update}/{total_updates} | Step {self.global_step:,}")
            
            # Save
            if update % self.cfg.save_interval == 0:
                self._save(f"pong_checkpoint_{self.global_step}.pt")
        
        # Final save
        total_time = time.time() - start_time
        print(f"\n{'='*65}")
        print(f"✅ Training Complete!")
        print(f"   Total time:  {total_time/60:.1f} minutes")
        print(f"   Avg FPS:     {sum(self.log['fps'])/len(self.log['fps']):,.0f}")
        print(f"   Final rally: {self.log['rally'][-1]:.1f}")
        print(f"{'='*65}")
        
        self._save("pong_final.pt")
    
    def _plot(self):
        """Plot training progress."""
        if len(self.log['reward']) < 2:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Reward
        axes[0, 0].plot(self.log['step'], self.log['reward'], 'b-', alpha=0.7)
        axes[0, 0].set_title('Episode Reward')
        axes[0, 0].set_xlabel('Steps')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win rate
        axes[0, 1].plot(self.log['step'], self.log['winrate'], 'g-', alpha=0.7)
        axes[0, 1].axhline(50, color='r', linestyle='--', alpha=0.5, label='50%')
        axes[0, 1].set_title('Win Rate vs Past Self')
        axes[0, 1].set_xlabel('Steps')
        axes[0, 1].set_ylim(0, 100)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Rally length
        axes[0, 2].plot(self.log['step'], self.log['rally'], 'm-', alpha=0.7)
        axes[0, 2].set_title('Avg Rally Length')
        axes[0, 2].set_xlabel('Steps')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Distance to ball
        axes[1, 0].plot(self.log['step'], self.log['dist'], 'c-', alpha=0.7)
        axes[1, 0].set_title('Avg Distance to Ball')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Entropy
        axes[1, 1].plot(self.log['step'], self.log['entropy'], 'orange', alpha=0.7)
        axes[1, 1].set_title('Policy Entropy')
        axes[1, 1].set_xlabel('Steps')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Game visualization
        try:
            img = self.env.render(0)
            axes[1, 2].imshow(img)
            sl = int(self.env.score_left[0].item())
            sr = int(self.env.score_right[0].item())
            axes[1, 2].set_title(f'Game: Opponent {sl} - {sr} Agent')
        except Exception as e:
            axes[1, 2].text(0.5, 0.5, f'Render error:\n{e}', ha='center', va='center')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def _save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.global_step,
            'updates': self.num_updates,
            'config': self.cfg,
        }, path)
        print(f"  💾 Saved: {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        self.global_step = ckpt.get('step', 0)
        self.num_updates = ckpt.get('updates', 0)
        print(f"  📂 Loaded: {path}")


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate(trainer: PPOTrainer, num_games: int = 100):
    """Evaluate agent against simple tracking AI."""
    print(f"\n Evaluating against simple AI ({num_games} games)...")
    
    trainer.model.eval()
    
    # Temporarily use simple AI as opponent
    old_opponent = trainer.env.opponent
    trainer.env.opponent = None
    
    trainer.env.reset()
    wins, losses = 0, 0
    
    with torch.no_grad():
        while wins + losses < num_games:
            obs = trainer.env._get_obs()
            action, _, _, _ = trainer.model.get_action_and_value(obs)
            _, _, _, info = trainer.env.step(action)
            wins += info['wins']
            losses += info['losses']
    
    # Restore self-play opponent
    trainer.env.opponent = old_opponent
    
    winrate = wins / (wins + losses) * 100
    print(f"  Results: {wins}W - {losses}L ({winrate:.1f}% win rate)")
    
    return winrate


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        print(f"   GPU: {props.name}")
        print(f"   Memory: {props.total_memory / 1e9:.1f} GB")
    
    # Configuration
    cfg = Config(
        # Environment
        num_envs=8192,
        
        # Training
        total_timesteps=100_000_000,  # 100M
        rollout_steps=128,
        
        # Self-play
        opponent_update_freq=10,      # Frequent updates
        latest_opponent_prob=0.5,     # 50/50 latest vs pool
        
        # Reward shaping (key settings!)
        distance_reward_scale=2.0,    # From your simple version
        hit_reward=1.0,
        rally_reward_scale=0.002,
        score_reward=5.0,
        win_bonus=10.0,
        
        # Logging
        plot_interval=10,
        save_interval=50,
    )
    
    # Quick benchmark
    print("\n⏱️  Quick benchmark...")
    temp_env = PongEnv(Config(num_envs=4096), device)
    temp_model = ActorCritic().to(device)
    
    obs = temp_env.reset()
    start = time.time()
    for _ in range(50):
        with torch.no_grad():
            action, _, _, _ = temp_model.get_action_and_value(obs)
        obs, _, _, _ = temp_env.step(action)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    fps = 50 * 4096 / elapsed
    print(f"   Estimated FPS: {fps:,.0f}")
    print(f"   Est. time for 100M steps: {100_000_000 / fps / 60:.1f} minutes")
    
    del temp_env, temp_model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Create trainer
    trainer = PPOTrainer(cfg, device)
    
    # Train!
    trainer.train()
    
    # Final evaluation
    evaluate(trainer, num_games=100)
    
    return trainer


# Run
if __name__ == "__main__":
    trainer = main()
