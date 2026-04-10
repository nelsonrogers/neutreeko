"""
Alpha-beta minimax AI for Neutreeko.
Used for Easy (depth 1) and Medium (depth 4) difficulty.
"""
import math
import random
from game.board import Board
from game.constants import DIR_NAMES


# ------------------------------------------------------------------
# Heuristic evaluation
# ------------------------------------------------------------------

def _min_pairwise_distance(positions):
    """Sum of Manhattan distances between all pairs of 3 positions."""
    total = 0
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            r0, c0 = positions[i]
            r1, c1 = positions[j]
            total += abs(r0 - r1) + abs(c0 - c1)
    return total


def _alignment_score(positions):
    """
    Score how 'close to winning' 3 positions are.
    Check each possible winning direction; reward collinearity.
    """
    score = 0
    # Check if any two pawns share a row, col, or diagonal
    (r0, c0), (r1, c1), (r2, c2) = sorted(positions)
    pairs = [(r0, c0, r1, c1), (r0, c0, r2, c2), (r1, c1, r2, c2)]
    for ra, ca, rb, cb in pairs:
        dr = rb - ra
        dc = cb - ca
        if dr == 0:           score += 3  # same row
        elif dc == 0:         score += 3  # same column
        elif abs(dr) == abs(dc): score += 2  # same diagonal
    return score


def evaluate(board, player):
    """
    Positive = good for `player`.
    Combines pawn proximity (lower = better) and alignment score.
    """
    opponent = 3 - player
    own_pos   = board.pawn_positions(player)
    opp_pos   = board.pawn_positions(opponent)

    own_dist  = _min_pairwise_distance(own_pos)
    opp_dist  = _min_pairwise_distance(opp_pos)
    own_align = _alignment_score(own_pos)
    opp_align = _alignment_score(opp_pos)

    return (opp_dist - own_dist) * 2 + (own_align - opp_align) * 3


# ------------------------------------------------------------------
# Alpha-beta search
# ------------------------------------------------------------------

def _alphabeta(board, depth, alpha, beta, maximising_player, root_player):
    winner = board.check_winner()
    if winner == root_player:
        return 10000 + depth  # win sooner is better
    if winner != 0:
        return -10000 - depth  # lose later is better

    if depth == 0:
        return evaluate(board, root_player)

    current_player = root_player if maximising_player else (3 - root_player)
    moves = board.get_legal_moves(current_player)

    if not moves:
        return evaluate(board, root_player)

    if maximising_player:
        value = -math.inf
        for pawn_id, direction, _, _ in moves:
            child = board.apply_move(pawn_id, direction)
            value = max(value, _alphabeta(child, depth - 1, alpha, beta, False, root_player))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        return value
    else:
        value = math.inf
        for pawn_id, direction, _, _ in moves:
            child = board.apply_move(pawn_id, direction)
            value = min(value, _alphabeta(child, depth - 1, alpha, beta, True, root_player))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value


def get_best_move(board, player, depth):
    """
    Return (pawn_id, direction) for the best move found via alpha-beta.
    depth=1 → Easy (win check only), depth=4 → Medium.
    """
    moves = board.get_legal_moves(player)
    if not moves:
        return None

    best_value = -math.inf
    best_moves = []

    for pawn_id, direction, _, _ in moves:
        child = board.apply_move(pawn_id, direction)
        value = _alphabeta(child, depth - 1, -math.inf, math.inf, False, player)
        if value > best_value:
            best_value = value
            best_moves = [(pawn_id, direction)]
        elif value == best_value:
            best_moves.append((pawn_id, direction))

    return random.choice(best_moves)
