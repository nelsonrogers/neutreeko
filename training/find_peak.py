"""
Find the peak checkpoint by evaluating each one against its neighbour.

Usage:
    python -m training.find_peak [--games N] [--dir PATH]

For each consecutive pair (5000 vs 10000, 10000 vs 15000, …) it plays N games
and records the winner. A checkpoint's cumulative score is its wins minus losses
across all matches it played. The highest score is the peak.
"""
import argparse
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.board import Board
from training.model import DQN

MAX_STEPS = 200


# ------------------------------------------------------------------
# Minimal greedy agent (identical to evaluate.py)
# ------------------------------------------------------------------

class Agent:
    def __init__(self, path):
        self.label = os.path.basename(path)
        self.net = DQN()
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt["policy_state"] if "policy_state" in ckpt else ckpt
        self.net.load_state_dict(sd)
        self.net.eval()

    def get_move(self, board, player):
        state = torch.tensor(board.encode(player), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q = self.net(state).squeeze(0).numpy()
        mask = board.legal_action_mask(player)
        q[~mask] = -np.inf
        return Board.action_to_move(int(np.argmax(q)), player)


def play_game(agent_p1, agent_p2):
    board = Board()
    agents = {1: agent_p1, 2: agent_p2}
    current = 1
    for _ in range(MAX_STEPS):
        pawn_id, direction = agents[current].get_move(board, current)
        board = board.apply_move(pawn_id, direction)
        winner = board.check_winner()
        if winner:
            return winner
        current = 3 - current
    return 0


def matchup(agent_a, agent_b, num_games):
    """Returns (wins_a, wins_b, draws)."""
    wins_a = wins_b = draws = 0
    for i in range(num_games):
        if i % 2 == 0:
            result = play_game(agent_a, agent_b)
            if result == 1:   wins_a += 1
            elif result == 2: wins_b += 1
            else:             draws  += 1
        else:
            result = play_game(agent_b, agent_a)
            if result == 2:   wins_a += 1
            elif result == 1: wins_b += 1
            else:             draws  += 1
    return wins_a, wins_b, draws


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def episode_number(path):
    m = re.search(r"(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 0


def find_peak(ckpt_dir, num_games):
    paths = sorted(
        [os.path.join(ckpt_dir, f)
         for f in os.listdir(ckpt_dir)
         if f.endswith(".pth") and f != "dqn_final.pth"],
        key=episode_number,
    )

    if len(paths) < 2:
        print("Need at least 2 checkpoints.")
        return

    print(f"Found {len(paths)} checkpoints — running {len(paths)-1} matchups "
          f"({num_games} games each)\n")
    print(f"{'Matchup':<40} {'A wins':>7} {'B wins':>7} {'Draws':>7}  Result")
    print("─" * 72)

    # score[i] = wins - losses across all matchups checkpoint i played
    scores = {p: 0 for p in paths}

    for i in range(len(paths) - 1):
        a, b = Agent(paths[i]), Agent(paths[i + 1])
        wa, wb, wd = matchup(a, b, num_games)

        label = f"{a.label}  vs  {b.label}"
        if wa > wb:
            result = f"← {a.label}"
        elif wb > wa:
            result = f"→ {b.label}"
        else:
            result = "draw"

        print(f"{label:<40} {wa:>7} {wb:>7} {wd:>7}  {result}")

        scores[paths[i]]     += wa - wb
        scores[paths[i + 1]] += wb - wa

    # Rank by cumulative score
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    print("\n" + "─" * 72)
    print(f"\n{'Rank':<6} {'Checkpoint':<35} {'Score':>7}")
    print("─" * 50)
    for rank, (path, score) in enumerate(ranked, 1):
        label = os.path.basename(path)
        marker = "  ← PEAK" if rank == 1 else ""
        print(f"{rank:<6} {label:<35} {score:>+7}{marker}")

    peak_path = ranked[0][0]
    print(f"\nBest checkpoint: {peak_path}")
    print(f"\nTo use it:\n"
          f"  python3 -m training.export_weights {peak_path} assets/weights.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--games", type=int, default=50,
        help="Games per matchup (must be even, default: 50)"
    )
    parser.add_argument(
        "--dir", default="training/checkpoints",
        help="Checkpoint directory (default: training/checkpoints)"
    )
    args = parser.parse_args()

    if args.games % 2 != 0:
        parser.error("--games must be even.")

    find_peak(args.dir, args.games)
