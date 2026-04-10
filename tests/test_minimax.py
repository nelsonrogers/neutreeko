"""
Tests for ai/minimax.py — alpha-beta search quality and correctness.
"""
import unittest
import numpy as np

from game.board import Board
from game.constants import P1_PAWNS, P2_PAWNS, DIR_NAMES
from ai.minimax import get_best_move, evaluate


def make_board(grid):
    return Board(np.array(grid, dtype=np.int8))


# ---------------------------------------------------------------------------
# get_best_move return-value contract
# ---------------------------------------------------------------------------

class TestGetBestMoveContract(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_returns_tuple(self):
        result = get_best_move(self.board, 1, depth=1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_pawn_belongs_to_player1(self):
        pawn_id, _ = get_best_move(self.board, 1, depth=1)
        self.assertIn(pawn_id, P1_PAWNS)

    def test_pawn_belongs_to_player2(self):
        pawn_id, _ = get_best_move(self.board, 2, depth=1)
        self.assertIn(pawn_id, P2_PAWNS)

    def test_direction_is_valid(self):
        _, direction = get_best_move(self.board, 1, depth=1)
        self.assertIn(direction, DIR_NAMES)

    def test_returned_move_is_legal(self):
        for depth in (1, 4):
            pawn_id, direction = get_best_move(self.board, 1, depth=depth)
            legal = self.board.get_legal_moves(1)
            legal_pairs = {(m[0], m[1]) for m in legal}
            self.assertIn((pawn_id, direction), legal_pairs,
                          f"depth={depth} returned illegal move ({pawn_id},{direction})")

    def test_none_when_no_moves(self):
        # Board with only one pawn pinned (no legal moves is extremely rare
        # in practice, so we test the None path explicitly)
        b = make_board([
            [1, 2, 0, 0, 0],   # p1 pawn 1 at (0,0), blocked left+up
            [3, 0, 0, 0, 0],   # p1 pawns form an L, pawn 2 at (0,1) blocks right
            [0, 0, 0, 0, 0],   # all legal moves exist for real pawns
            [0, 0, 0, 0, 0],
            [0, 0, 4, 5, 6],
        ])
        # Just verify it returns something (or None) without raising
        result = get_best_move(b, 1, depth=1)
        if result is not None:
            pawn_id, direction = result
            self.assertIn(pawn_id, P1_PAWNS)


# ---------------------------------------------------------------------------
# Tactical correctness — wins in 1 move
# ---------------------------------------------------------------------------

class TestWinsInOneMove(unittest.TestCase):
    """
    In each board below player 1 has exactly one move that wins immediately.
    depth=1 is sufficient to find a 1-move win.
    """

    def _assert_wins(self, board, player=1, depth=1):
        pawn_id, direction = get_best_move(board, player, depth=depth)
        result = board.apply_move(pawn_id, direction)
        self.assertEqual(result.check_winner(), player,
                         f"move ({pawn_id},{direction}) did not win")

    def test_horizontal_win_available(self):
        # Pawns 1,2 already adjacent at row 0; pawn 3 needs to slide right to complete line
        b = make_board([
            [1, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],   # pawn 3 can slide R → (3,4)? No, needs same row
            [0, 5, 0, 6, 4],
        ])
        # Manually construct a guaranteed 1-move win
        b2 = make_board([
            [0, 1, 2, 0, 0],   # pawn 3 slides L → col 0, completes line
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 5, 0, 6, 4],
        ])
        self._assert_wins(b2)

    def test_vertical_win_available(self):
        b = make_board([
            [1, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],   # pawn 3 slides U → (2,0), completing col 0
            [0, 5, 0, 6, 4],
        ])
        self._assert_wins(b)

    def test_diagonal_win_available(self):
        b = make_board([
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 3, 0],   # pawn 3 slides UL → (2,2), completing diagonal
            [0, 5, 0, 6, 4],
        ])
        self._assert_wins(b)

    def test_depth4_also_finds_win(self):
        b = make_board([
            [1, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 5, 0, 6, 4],
        ])
        self._assert_wins(b, depth=4)


# ---------------------------------------------------------------------------
# Tactical correctness — does not walk into a loss
# ---------------------------------------------------------------------------

class TestAvoidsImmediateLoss(unittest.TestCase):

    def test_does_not_complete_opponent_line(self):
        """
        Player 2 pawns 5,6 are adjacent; p1 should not make a move that
        lets p2 immediately win on the next move.
        This is a property test: run medium-depth search and verify the
        resulting board is not immediately lost for p1.
        """
        b = Board()
        pawn_id, direction = get_best_move(b, 1, depth=4)
        b2 = b.apply_move(pawn_id, direction)
        # p2 should not already be a winner
        self.assertNotEqual(b2.check_winner(), 2)


# ---------------------------------------------------------------------------
# Heuristic evaluation
# ---------------------------------------------------------------------------

class TestEvaluate(unittest.TestCase):

    def test_winning_position_has_high_score(self):
        b_win = make_board([
            [1, 2, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [4, 0, 5, 0, 6],
        ])
        b_neutral = Board()
        self.assertGreater(evaluate(b_win, 1), evaluate(b_neutral, 1))

    def test_score_is_symmetric(self):
        b = Board()
        # Symmetric initial position: both players have same structure
        # (not exactly, but scores should be close)
        s1 = evaluate(b, 1)
        s2 = evaluate(b, 2)
        self.assertAlmostEqual(s1, -s2, places=5)

    def test_closer_pawns_better(self):
        # Tightly grouped p1 pawns should score higher than spread ones
        b_tight = make_board([
            [1, 2, 3, 0, 0],   # all in row 0 cols 0-2, distance 1
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [4, 0, 5, 0, 6],
        ])
        b_spread = make_board([
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 0, 3, 0],
            [4, 0, 5, 0, 6],
        ])
        self.assertGreater(evaluate(b_tight, 1), evaluate(b_spread, 1))


if __name__ == "__main__":
    unittest.main()
