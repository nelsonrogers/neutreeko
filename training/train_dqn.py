"""
Self-play DQN training for Neutreeko.

Usage:
    # Start fresh
    python -m training.train_dqn

    # Resume from a checkpoint (adds more episodes on top)
    python -m training.train_dqn --resume training/checkpoints/dqn_20000.pth

    # Resume and specify how many additional episodes to run
    python -m training.train_dqn --resume training/checkpoints/dqn_20000.pth --episodes 20000

Produces:  training/checkpoints/dqn_<episode>.pth
           assets/weights.npz  (auto-exported at the end)

Checkpoint format
-----------------
Each .pth file is a dict with keys:
    policy_state   – policy network weights
    target_state   – target network weights
    optimizer_state – Adam optimizer state (preserves momentum)
    total_steps    – global step counter (keeps ε schedule correct)
    episode        – episode number at time of save
"""
import argparse
import sys
import os
import random
import math
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board
from game.constants import NUM_ACTIONS
from training.model import DQN
from training.export_weights import export_weights

# ------------------------------------------------------------------
# Hyper-parameters
# ------------------------------------------------------------------
EPISODES         = 20_000
BATCH_SIZE       = 64
GAMMA            = 0.99
LR               = 1e-3
REPLAY_CAPACITY  = 50_000
EPS_START        = 1.0
EPS_END          = 0.05
EPS_DECAY        = 10_000        # steps to decay epsilon
TARGET_SYNC      = 500           # steps between target network updates
CHECKPOINT_EVERY = 5_000         # episodes
MAX_STEPS        = 200           # max moves per game (prevent infinite loops)

# Reward shaping
# A small per-step reward for improving own piece proximity and alignment.
# Kept small relative to the terminal +1/-1 so the agent still optimises
# for winning rather than just clustering pieces indefinitely.
SHAPE_SCALE      = 0.06   # max shaped reward per step ≈ 0.06 * 16 = 0.96 < 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# Replay buffer
# ------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buf.append((state, action, reward, next_state, done))

    def sample(self, n):
        batch = random.sample(self.buf, n)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states),      dtype=torch.float32, device=DEVICE),
            torch.tensor(actions,                dtype=torch.long,    device=DEVICE),
            torch.tensor(rewards,                dtype=torch.float32, device=DEVICE),
            torch.tensor(np.array(next_states),  dtype=torch.float32, device=DEVICE),
            torch.tensor(dones,                  dtype=torch.float32, device=DEVICE),
        )

    def __len__(self):
        return len(self.buf)


# ------------------------------------------------------------------
# Epsilon schedule
# ------------------------------------------------------------------
def epsilon(step):
    return EPS_END + (EPS_START - EPS_END) * math.exp(-step / EPS_DECAY)


# ------------------------------------------------------------------
# Action selection
# ------------------------------------------------------------------
def select_action(board, player, policy_net, step, greedy=False):
    """Epsilon-greedy action selection. Returns action index (0-23)."""
    mask = board.legal_action_mask(player)
    if not mask.any():
        return None

    eps = 0.0 if greedy else epsilon(step)
    if random.random() < eps:
        legal_indices = np.where(mask)[0]
        return int(random.choice(legal_indices))

    state = torch.tensor(board.encode(player), dtype=torch.float32, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        q = policy_net(state).squeeze(0).cpu().numpy()
    q[~mask] = -np.inf
    return int(np.argmax(q))


# ------------------------------------------------------------------
# Training step
# ------------------------------------------------------------------
def optimize(policy_net, target_net, optimizer, replay):
    if len(replay) < BATCH_SIZE:
        return
    states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)

    # Current Q values
    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Target Q values (Double DQN)
    with torch.no_grad():
        next_actions = policy_net(next_states).argmax(dim=1)
        next_q = target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
        targets = rewards + GAMMA * next_q * (1 - dones)

    loss = nn.SmoothL1Loss()(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    return loss.item()


# ------------------------------------------------------------------
# Checkpoint helpers
# ------------------------------------------------------------------

def save_checkpoint(path, policy_net, target_net, optimizer, total_steps, episode):
    torch.save({
        "policy_state":    policy_net.state_dict(),
        "target_state":    target_net.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "total_steps":     total_steps,
        "episode":         episode,
    }, path)


def load_checkpoint(path, policy_net, target_net, optimizer):
    """
    Load a checkpoint. Supports both the new dict format and the old format
    (plain state_dict saved by earlier versions of this script).
    Returns (total_steps, episode_offset) so the training loop can resume.
    """
    ckpt = torch.load(path, map_location=DEVICE)

    if isinstance(ckpt, dict) and "policy_state" in ckpt:
        # New format
        policy_net.load_state_dict(ckpt["policy_state"])
        target_net.load_state_dict(ckpt["target_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        total_steps    = ckpt["total_steps"]
        episode_offset = ckpt["episode"]
    else:
        # Old format: plain state_dict (no optimizer / step info)
        policy_net.load_state_dict(ckpt)
        target_net.load_state_dict(ckpt)
        total_steps    = 0
        episode_offset = 0
        print("  Warning: old-format checkpoint — epsilon schedule restarts from 0.")

    return total_steps, episode_offset


# ------------------------------------------------------------------
# Reward shaping helper
# ------------------------------------------------------------------

def _pairwise_dist(positions):
    return sum(
        abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
        for i in range(3) for j in range(i + 1, 3)
    )


def _alignment(positions):
    return sum(
        1
        for i in range(3) for j in range(i + 1, 3)
        if (positions[i][0] == positions[j][0]
            or positions[i][1] == positions[j][1]
            or abs(positions[i][0] - positions[j][0])
               == abs(positions[i][1] - positions[j][1]))
    )


def board_score(board, player):
    """
    Relative heuristic score for player (higher = better).

    Three signals:
      1. Proximity: own pieces closer together is better; opponent closer is worse
      2. Alignment: own pairs sharing row/col/diagonal score higher; opponent's hurts
      3. Threat penalty: -10 if the opponent can win on their very next move,
         making threat-blocking the agent's top priority
    """
    opponent = 3 - player
    own_pos  = board.pawn_positions(player)
    opp_pos  = board.pawn_positions(opponent)

    base = (
        (-_pairwise_dist(own_pos) + 3 * _alignment(own_pos))
        - (-_pairwise_dist(opp_pos) + 3 * _alignment(opp_pos))
    )

    threat = any(
        board.apply_move(pid, d).check_winner() == opponent
        for pid, d, _, _ in board.get_legal_moves(opponent)
    )

    return base - (10 if threat else 0)


# ------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------

def train(resume_path=None, extra_episodes=None):
    policy_net = DQN().to(DEVICE)
    target_net = DQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_CAPACITY)

    total_steps    = 0
    episode_offset = 0

    if resume_path:
        print(f"Resuming from: {resume_path}")
        total_steps, episode_offset = load_checkpoint(
            resume_path, policy_net, target_net, optimizer
        )
        eps_now = epsilon(total_steps)
        print(f"  Restored total_steps={total_steps}, last episode={episode_offset}, ε={eps_now:.4f}")

    num_episodes = extra_episodes if extra_episodes is not None else EPISODES
    wins = [0, 0]   # wins[0]=p1, wins[1]=p2

    for episode in range(1, num_episodes + 1):
        board = Board()
        current_player = 1
        episode_steps = 0
        done = False

        prev_state  = [None, None]
        prev_action = [None, None]

        while not done and episode_steps < MAX_STEPS:
            pi = current_player - 1   # 0 or 1
            state = board.encode(current_player)

            action = select_action(board, current_player, policy_net, total_steps)
            if action is None:
                break

            pawn_id, direction = Board.action_to_move(action, current_player)
            next_board = board.apply_move(pawn_id, direction)
            winner = next_board.check_winner()
            done = winner != 0

            # Reward signal
            if winner == current_player:
                reward = 1.0
                wins[pi] += 1
            elif winner != 0:
                reward = -1.0
            else:
                # Shaped reward: improvement in own piece configuration
                # relative to the previous board state.
                reward = SHAPE_SCALE * (
                    board_score(next_board, current_player)
                    - board_score(board, current_player)
                )

            next_state = next_board.encode(current_player)

            # Store transition
            replay.push(state, action, reward, next_state, float(done))

            # Give the *opponent* a negative reward if current player just won
            opponent = 3 - current_player
            op = opponent - 1
            if prev_state[op] is not None and winner == current_player:
                replay.push(prev_state[op], prev_action[op], -1.0, next_state, 1.0)

            prev_state[pi]  = state
            prev_action[pi] = action

            # Optimize every 4 steps
            if total_steps % 4 == 0:
                optimize(policy_net, target_net, optimizer, replay)

            # Sync target network
            if total_steps % TARGET_SYNC == 0:
                target_net.load_state_dict(policy_net.state_dict())

            board = next_board
            current_player = 3 - current_player
            total_steps += 1
            episode_steps += 1

        # Logging — show absolute episode number
        abs_episode = episode_offset + episode
        if episode % 1000 == 0:
            eps_val = epsilon(total_steps)
            print(f"Episode {abs_episode:6d} | steps {total_steps:7d} | "
                  f"ε={eps_val:.3f} | wins p1={wins[0]} p2={wins[1]}")
            wins = [0, 0]

        # Checkpoint — filename uses absolute episode number
        if episode % CHECKPOINT_EVERY == 0:
            ckpt = os.path.join(CKPT_DIR, f"dqn_{abs_episode}.pth")
            save_checkpoint(ckpt, policy_net, target_net, optimizer, total_steps, abs_episode)
            print(f"  → saved checkpoint: {ckpt}")

    # Final save + export
    abs_final = episode_offset + num_episodes
    final_path = os.path.join(CKPT_DIR, f"dqn_{abs_final}.pth")
    save_checkpoint(final_path, policy_net, target_net, optimizer, total_steps, abs_final)
    print(f"\nTraining complete. Weights saved to {final_path}")

    assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    export_weights(final_path, os.path.join(assets_dir, "weights.npz"))
    print("Weights exported to assets/weights.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Neutreeko DQN agent.")
    parser.add_argument(
        "--resume", metavar="PATH",
        help="Path to a checkpoint .pth file to resume training from."
    )
    parser.add_argument(
        "--episodes", type=int, default=None,
        help="Number of additional episodes to run (default: EPISODES constant = %(default)s)."
    )
    args = parser.parse_args()
    train(resume_path=args.resume, extra_episodes=args.episodes)
