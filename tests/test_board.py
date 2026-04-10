"""
Tests for game/board.py — Board state, move generation, win detection,
state encoding, and action indexing.
"""
import unittest
import numpy as np

from game.board import Board
from game.constants import INITIAL_BOARD, DIR_NAMES, P1_PAWNS, P2_PAWNS, NUM_ACTIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_board(grid):
    """Shorthand to build a Board from a 5×5 list-of-lists."""
    return Board(np.array(grid, dtype=np.int8))


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestBoardInit(unittest.TestCase):

    def test_default_grid_matches_initial(self):
        b = Board()
        np.testing.assert_array_equal(b.grid, INITIAL_BOARD)

    def test_custom_grid(self):
        grid = np.zeros((5, 5), dtype=np.int8)
        grid[0, 0] = 1
        b = Board(grid)
        self.assertEqual(b.grid[0, 0], 1)

    def test_copy_is_independent(self):
        b = Board()
        c = b.copy()
        c.grid[0, 0] = 99
        self.assertNotEqual(b.grid[0, 0], 99)

    def test_grid_dtype(self):
        b = Board()
        self.assertEqual(b.grid.dtype, np.int8)


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------

class TestFindPawn(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_finds_all_six_pawns(self):
        for pawn_id in range(1, 7):
            pos = self.board.find_pawn(pawn_id)
            self.assertIsNotNone(pos, f"pawn {pawn_id} not found")

    def test_correct_positions(self):
        # Initial layout: [[0,1,0,2,0],[0,0,4,0,0],[0,0,0,0,0],[0,0,3,0,0],[0,5,0,6,0]]
        self.assertEqual(self.board.find_pawn(1), (0, 1))
        self.assertEqual(self.board.find_pawn(2), (0, 3))
        self.assertEqual(self.board.find_pawn(3), (3, 2))
        self.assertEqual(self.board.find_pawn(4), (1, 2))
        self.assertEqual(self.board.find_pawn(5), (4, 1))
        self.assertEqual(self.board.find_pawn(6), (4, 3))

    def test_missing_pawn_returns_none(self):
        b = Board(np.zeros((5, 5), dtype=np.int8))
        self.assertIsNone(b.find_pawn(1))

    def test_pawns_of_player1(self):
        self.assertEqual(self.board.pawns_of(1), list(P1_PAWNS))

    def test_pawns_of_player2(self):
        self.assertEqual(self.board.pawns_of(2), list(P2_PAWNS))


# ---------------------------------------------------------------------------
# Slide destination
# ---------------------------------------------------------------------------

class TestSlideDestination(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_slides_to_wall_right(self):
        # pawn 2 is at (0,3); sliding R should reach (0,4)
        dest = self.board.slide_destination(0, 3, "R")
        self.assertEqual(dest, (0, 4))

    def test_slides_to_wall_left(self):
        # pawn 1 at (0,1); L → (0,0)
        dest = self.board.slide_destination(0, 1, "L")
        self.assertEqual(dest, (0, 0))

    def test_blocked_by_piece(self):
        # pawn 1 at (0,1); R is blocked by pawn 2 at (0,3), so stops at (0,2)
        dest = self.board.slide_destination(0, 1, "R")
        self.assertEqual(dest, (0, 2))

    def test_no_movement_when_immediately_blocked(self):
        # Place a piece directly to the right of (0,0)
        b = make_board([
            [1, 2, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        dest = b.slide_destination(0, 0, "R")
        self.assertEqual(dest, (0, 0))  # no movement

    def test_diagonal_slide(self):
        # From (2,2) DR should slide to (4,4) if clear
        b = make_board([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        dest = b.slide_destination(2, 2, "DR")
        self.assertEqual(dest, (4, 4))


# ---------------------------------------------------------------------------
# Legal move generation
# ---------------------------------------------------------------------------

class TestLegalMoves(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_initial_move_count_player1(self):
        moves = self.board.get_legal_moves(1)
        self.assertEqual(len(moves), 14)

    def test_initial_move_count_player2(self):
        moves = self.board.get_legal_moves(2)
        self.assertEqual(len(moves), 14)

    def test_move_format(self):
        for pawn_id, direction, dr, dc in self.board.get_legal_moves(1):
            self.assertIn(pawn_id, P1_PAWNS)
            self.assertIn(direction, DIR_NAMES)
            self.assertIn(dr, range(5))
            self.assertIn(dc, range(5))

    def test_destination_is_empty(self):
        for _, _, dr, dc in self.board.get_legal_moves(1):
            self.assertEqual(self.board.grid[dr, dc], 0,
                             f"destination ({dr},{dc}) is not empty")

    def test_destination_differs_from_source(self):
        for pawn_id, _, dr, dc in self.board.get_legal_moves(1):
            src = self.board.find_pawn(pawn_id)
            self.assertNotEqual(src, (dr, dc))

    def test_only_own_pawns_in_moves(self):
        for pawn_id, _, _, _ in self.board.get_legal_moves(2):
            self.assertIn(pawn_id, P2_PAWNS)


# ---------------------------------------------------------------------------
# Apply move
# ---------------------------------------------------------------------------

class TestApplyMove(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_pawn_moves_to_destination(self):
        b2 = self.board.apply_move(1, "D")
        new_pos = b2.find_pawn(1)
        self.assertIsNotNone(new_pos)
        self.assertNotEqual(new_pos, (0, 1))

    def test_source_is_cleared(self):
        b2 = self.board.apply_move(1, "D")
        self.assertEqual(b2.grid[0, 1], 0)

    def test_original_board_unchanged(self):
        original = self.board.grid.copy()
        self.board.apply_move(1, "D")
        np.testing.assert_array_equal(self.board.grid, original)

    def test_pawn_id_preserved_at_destination(self):
        b2 = self.board.apply_move(1, "D")
        dest = b2.find_pawn(1)
        self.assertEqual(b2.grid[dest], 1)

    def test_invalid_direction_leaves_board_unchanged(self):
        # Pawn 2 is at (0,3); moving R should reach (0,4) — valid
        # Moving further R from (0,4) would be blocked (edge): try L from edge
        b = make_board([
            [1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        original = b.grid.copy()
        b2 = b.apply_move(1, "L")   # already at col 0, can't go left
        np.testing.assert_array_equal(b2.grid, original)


# ---------------------------------------------------------------------------
# Win detection
# ---------------------------------------------------------------------------

class TestCheckWinner(unittest.TestCase):

    def test_no_winner_initial(self):
        self.assertEqual(Board().check_winner(), 0)

    def test_player1_wins_horizontal(self):
        b = make_board([
            [1, 2, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [4, 0, 5, 0, 6],
        ])
        self.assertEqual(b.check_winner(), 1)

    def test_player1_wins_vertical(self):
        b = make_board([
            [1, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [4, 0, 5, 0, 6],
        ])
        self.assertEqual(b.check_winner(), 1)

    def test_player1_wins_diagonal(self):
        b = make_board([
            [1, 0, 0, 0, 0],
            [0, 2, 0, 0, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
            [4, 0, 5, 0, 6],
        ])
        self.assertEqual(b.check_winner(), 1)

    def test_player2_wins_horizontal(self):
        b = make_board([
            [1, 0, 2, 0, 3],
            [0, 0, 0, 0, 0],
            [4, 5, 6, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        self.assertEqual(b.check_winner(), 2)

    def test_player2_wins_anti_diagonal(self):
        b = make_board([
            [1, 0, 2, 0, 3],
            [0, 0, 0, 0, 0],
            [0, 0, 4, 0, 0],
            [0, 5, 0, 0, 0],
            [6, 0, 0, 0, 0],
        ])
        self.assertEqual(b.check_winner(), 2)

    def test_two_pawns_adjacent_is_not_a_win(self):
        b = make_board([
            [1, 2, 0, 3, 0],   # 1 and 2 adjacent but 3 not
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [4, 0, 5, 0, 6],
        ])
        self.assertEqual(b.check_winner(), 0)

    def test_spaced_pieces_are_not_a_win(self):
        b = make_board([
            [1, 0, 2, 0, 3],   # spaced 2 apart
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [4, 0, 5, 0, 6],
        ])
        self.assertEqual(b.check_winner(), 0)


# ---------------------------------------------------------------------------
# State encoding
# ---------------------------------------------------------------------------

class TestEncode(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_shape(self):
        self.assertEqual(self.board.encode(1).shape, (75,))

    def test_dtype(self):
        self.assertEqual(self.board.encode(1).dtype, np.float32)

    def test_sum_equals_board_cells(self):
        # Every cell maps to exactly one channel → sum = 25
        self.assertAlmostEqual(float(self.board.encode(1).sum()), 25.0)

    def test_values_are_binary(self):
        enc = self.board.encode(1)
        self.assertTrue(np.all((enc == 0) | (enc == 1)))

    def test_own_channel_count(self):
        enc = self.board.encode(1).reshape(5, 5, 3)
        self.assertEqual(int(enc[:, :, 1].sum()), 3)  # 3 own pawns

    def test_opponent_channel_count(self):
        enc = self.board.encode(1).reshape(5, 5, 3)
        self.assertEqual(int(enc[:, :, 2].sum()), 3)  # 3 opponent pawns

    def test_empty_channel_count(self):
        enc = self.board.encode(1).reshape(5, 5, 3)
        self.assertEqual(int(enc[:, :, 0].sum()), 19)  # 25 - 6 pieces

    def test_perspective_swaps_channels(self):
        enc1 = self.board.encode(1).reshape(5, 5, 3)
        enc2 = self.board.encode(2).reshape(5, 5, 3)
        # Own channel for p1 == opponent channel for p2 and vice-versa
        np.testing.assert_array_equal(enc1[:, :, 1], enc2[:, :, 2])
        np.testing.assert_array_equal(enc1[:, :, 2], enc2[:, :, 1])


# ---------------------------------------------------------------------------
# Action indexing
# ---------------------------------------------------------------------------

class TestActionIndexing(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def test_action_to_move_p1_first(self):
        pawn_id, direction = Board.action_to_move(0, 1)
        self.assertEqual(pawn_id, P1_PAWNS[0])
        self.assertEqual(direction, DIR_NAMES[0])

    def test_action_to_move_p1_last(self):
        pawn_id, direction = Board.action_to_move(23, 1)
        self.assertEqual(pawn_id, P1_PAWNS[2])
        self.assertEqual(direction, DIR_NAMES[7])

    def test_action_to_move_p2(self):
        pawn_id, _ = Board.action_to_move(0, 2)
        self.assertEqual(pawn_id, P2_PAWNS[0])

    def test_legal_action_mask_length(self):
        mask = self.board.legal_action_mask(1)
        self.assertEqual(len(mask), NUM_ACTIONS)

    def test_legal_action_mask_count_matches_moves(self):
        legal_count = len(self.board.get_legal_moves(1))
        mask_count = int(self.board.legal_action_mask(1).sum())
        self.assertEqual(mask_count, legal_count)

    def test_masked_actions_correspond_to_legal_moves(self):
        moves = self.board.get_legal_moves(1)
        mask = self.board.legal_action_mask(1)
        for pawn_id, direction, _, _ in moves:
            pawn_idx = list(P1_PAWNS).index(pawn_id)
            dir_idx = DIR_NAMES.index(direction)
            action = pawn_idx * 8 + dir_idx
            self.assertTrue(mask[action],
                            f"action {action} ({pawn_id},{direction}) not in mask")


if __name__ == "__main__":
    unittest.main()
