"""
Head-to-head evaluation between two DQN checkpoints.

Usage:
    python -m training.evaluate <checkpoint_a> <checkpoint_b> [--games N]

Example:
    python -m training.evaluate \\
        training/checkpoints/dqn_20000.pth \\
        training/checkpoints/dqn_60000.pth \\
        --games 200

Each agent always plays greedily (ε=0). Games alternate which agent moves
first to cancel out any first-mover advantage in the win-rate totals.
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board
from game.constants import NUM_ACTIONS
from training.model import DQN

MAX_STEPS = 200   # draw after this many moves (same cap as training)


# ------------------------------------------------------------------
# Greedy agent (PyTorch, no epsilon)
# ------------------------------------------------------------------

class EvalAgent:
    """Loads a checkpoint and plays greedily — no exploration."""

    def __init__(self, path):
        self.path = path
        self.net = DQN()
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt["policy_state"] if "policy_state" in ckpt else ckpt
        self.net.load_state_dict(state_dict)
        self.net.eval()

    def get_move(self, board, player):
        state = torch.tensor(board.encode(player), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.net(state).squeeze(0).numpy()
        mask = board.legal_action_mask(player)
        q[~mask] = -np.inf
        action = int(np.argmax(q))
        return Board.action_to_move(action, player)


# ------------------------------------------------------------------
# Single game
# ------------------------------------------------------------------

def play_game(agent_p1, agent_p2):
    """
    Play one game. agent_p1 controls player 1, agent_p2 controls player 2.
    Returns (winner, num_moves): winner is 1, 2, or 0 for draw.
    """
    board = Board()
    agents = {1: agent_p1, 2: agent_p2}
    current = 1

    for move_num in range(MAX_STEPS):
        pawn_id, direction = agents[current].get_move(board, current)
        board = board.apply_move(pawn_id, direction)
        winner = board.check_winner()
        if winner:
            return winner, move_num + 1
        current = 3 - current

    return 0, MAX_STEPS


# ------------------------------------------------------------------
# Evaluation loop
# ------------------------------------------------------------------

def evaluate(path_a, path_b, num_games):
    agent_a = EvalAgent(path_a)
    agent_b = EvalAgent(path_b)

    label_a = os.path.basename(path_a)
    label_b = os.path.basename(path_b)

    wins_a = 0
    wins_b = 0
    draws  = 0
    total_moves = 0

    print(f"\nAgent A: {label_a}")
    print(f"Agent B: {label_b}")
    print(f"Games:   {num_games} ({num_games // 2} as P1, {num_games // 2} as P2)\n")

    for i in range(num_games):
        # Alternate who moves first each game
        if i % 2 == 0:
            first, second = agent_a, agent_b
            a_is_p1 = True
        else:
            first, second = agent_b, agent_a
            a_is_p1 = False

        winner, moves = play_game(first, second)
        total_moves += moves

        if winner == 0:
            draws += 1
        elif (winner == 1 and a_is_p1) or (winner == 2 and not a_is_p1):
            wins_a += 1
        else:
            wins_b += 1

        # Progress every 10%
        if (i + 1) % max(1, num_games // 10) == 0:
            pct = (i + 1) / num_games * 100
            print(f"  {i + 1:4d}/{num_games}  ({pct:.0f}%)  "
                  f"A={wins_a}  B={wins_b}  draws={draws}")

    decisive = num_games - draws
    avg_moves = total_moves / num_games

    print(f"\n{'─' * 44}")
    print(f"{'Agent A':<20} {wins_a:4d} wins  "
          f"({wins_a / num_games * 100:.1f}%)")
    print(f"{'Agent B':<20} {wins_b:4d} wins  "
          f"({wins_b / num_games * 100:.1f}%)")
    if draws:
        print(f"{'Draws':<20} {draws:4d}       "
              f"({draws / num_games * 100:.1f}%)")
    print(f"{'Avg game length':<20} {avg_moves:.1f} moves")
    print(f"{'─' * 44}")

    if decisive == 0:
        print("\nAll games were draws — inconclusive.")
    elif wins_a > wins_b:
        delta = (wins_a - wins_b) / num_games * 100
        print(f"\n→ Agent A is stronger  (+{delta:.1f} pp)")
    elif wins_b > wins_a:
        delta = (wins_b - wins_a) / num_games * 100
        print(f"\n→ Agent B is stronger  (+{delta:.1f} pp)")
    else:
        print("\n→ Even match.")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Head-to-head evaluation between two DQN checkpoints."
    )
    parser.add_argument("checkpoint_a", help="Path to first checkpoint (.pth)")
    parser.add_argument("checkpoint_b", help="Path to second checkpoint (.pth)")
    parser.add_argument(
        "--games", type=int, default=200,
        help="Number of games to play (must be even, default: 200)"
    )
    args = parser.parse_args()

    if args.games % 2 != 0:
        parser.error("--games must be even so each agent plays the same number as P1 and P2.")

    evaluate(args.checkpoint_a, args.checkpoint_b, args.games)
