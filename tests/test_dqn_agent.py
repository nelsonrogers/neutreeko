"""
Tests for ai/dqn_agent.py — DQN inference agent.

These tests do not require a trained weights file.
They verify the agent's interface, fallback behaviour, and numpy forward pass.
"""
import os
import tempfile
import unittest
import numpy as np

from game.board import Board
from game.constants import P1_PAWNS, P2_PAWNS, DIR_NAMES, NUM_ACTIONS
from ai.dqn_agent import DQNAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_board(grid):
    return Board(np.array(grid, dtype=np.int8))


def _make_random_weights():
    """Return a dict of small random numpy arrays matching the network shape."""
    rng = np.random.default_rng(42)
    return {
        "W1": rng.standard_normal((128, 75)).astype(np.float32),
        "b1": rng.standard_normal((128,)).astype(np.float32),
        "W2": rng.standard_normal((64, 128)).astype(np.float32),
        "b2": rng.standard_normal((64,)).astype(np.float32),
        "W3": rng.standard_normal((24, 64)).astype(np.float32),
        "b3": rng.standard_normal((24,)).astype(np.float32),
    }


def _save_weights(path):
    np.savez(path, **_make_random_weights())


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

class TestDQNAgentInit(unittest.TestCase):

    def test_loaded_false_without_file(self):
        agent = DQNAgent(weights_path="/nonexistent/path/weights.npz")
        self.assertFalse(agent.loaded)

    def test_loaded_true_with_valid_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.npz")
            _save_weights(path)
            agent = DQNAgent(weights_path=path)
            self.assertTrue(agent.loaded)

    def test_weight_shapes_after_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.npz")
            _save_weights(path)
            agent = DQNAgent(weights_path=path)
            self.assertEqual(agent.W1.shape, (128, 75))
            self.assertEqual(agent.b1.shape, (128,))
            self.assertEqual(agent.W2.shape, (64, 128))
            self.assertEqual(agent.b2.shape, (64,))
            self.assertEqual(agent.W3.shape, (24, 64))
            self.assertEqual(agent.b3.shape, (24,))

    def test_corrupted_file_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "bad.npz")
            # Write a valid npz with wrong keys
            np.savez(path, garbage=np.zeros(5))
            agent = DQNAgent(weights_path=path)
            self.assertFalse(agent.loaded)


# ---------------------------------------------------------------------------
# Forward pass (numpy inference)
# ---------------------------------------------------------------------------

class TestForwardPass(unittest.TestCase):

    def setUp(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "weights.npz")
            _save_weights(path)
            self.agent = DQNAgent(weights_path=path)
        # Keep a reference to tmpdir open; weights already in memory

    def test_output_shape(self):
        state = np.zeros(75, dtype=np.float32)
        q = self.agent.forward(state)
        self.assertEqual(q.shape, (NUM_ACTIONS,))

    def test_output_dtype(self):
        state = np.zeros(75, dtype=np.float32)
        q = self.agent.forward(state)
        self.assertEqual(q.dtype, np.float32)

    def test_relu_zeros_negatives(self):
        x = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
        result = DQNAgent._relu(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0])

    def test_forward_is_deterministic(self):
        state = Board().encode(1)
        q1 = self.agent.forward(state)
        q2 = self.agent.forward(state)
        np.testing.assert_array_equal(q1, q2)

    def test_different_states_give_different_q_values(self):
        b = Board()
        s1 = b.encode(1)
        # Move a piece so the board changes
        b2 = b.apply_move(1, "D")
        s2 = b2.encode(1)
        q1 = self.agent.forward(s1)
        q2 = self.agent.forward(s2)
        self.assertFalse(np.allclose(q1, q2))


# ---------------------------------------------------------------------------
# get_move — interface and legality
# ---------------------------------------------------------------------------

class TestGetMove(unittest.TestCase):

    def setUp(self):
        self.board = Board()

    def _make_agent(self, with_weights=False):
        if with_weights:
            tmpdir = tempfile.mkdtemp()
            path = os.path.join(tmpdir, "weights.npz")
            _save_weights(path)
            return DQNAgent(weights_path=path)
        return DQNAgent(weights_path="/nonexistent/weights.npz")

    def test_returns_tuple_without_weights(self):
        agent = self._make_agent(with_weights=False)
        result = agent.get_move(self.board, 1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def test_returns_tuple_with_weights(self):
        agent = self._make_agent(with_weights=True)
        result = agent.get_move(self.board, 1)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)

    def _assert_move_is_legal(self, player, with_weights):
        agent = self._make_agent(with_weights=with_weights)
        for _ in range(20):   # run several times to cover random fallback
            pawn_id, direction = agent.get_move(self.board, player)
            legal = {(m[0], m[1]) for m in self.board.get_legal_moves(player)}
            self.assertIn((pawn_id, direction), legal,
                          f"({pawn_id},{direction}) is not a legal move for player {player}")

    def test_move_is_legal_p1_no_weights(self):
        self._assert_move_is_legal(1, with_weights=False)

    def test_move_is_legal_p2_no_weights(self):
        self._assert_move_is_legal(2, with_weights=False)

    def test_move_is_legal_p1_with_weights(self):
        self._assert_move_is_legal(1, with_weights=True)

    def test_move_is_legal_p2_with_weights(self):
        self._assert_move_is_legal(2, with_weights=True)

    def test_pawn_belongs_to_player(self):
        agent = self._make_agent(with_weights=True)
        for player, expected_pawns in ((1, P1_PAWNS), (2, P2_PAWNS)):
            pawn_id, _ = agent.get_move(self.board, player)
            self.assertIn(pawn_id, expected_pawns)

    def test_direction_is_valid(self):
        agent = self._make_agent(with_weights=True)
        _, direction = agent.get_move(self.board, 1)
        self.assertIn(direction, DIR_NAMES)

    def test_action_masking_never_picks_illegal_action(self):
        """After masking, argmax should never land on a masked-out action."""
        agent = self._make_agent(with_weights=True)
        for _ in range(50):
            pawn_id, direction = agent.get_move(self.board, 1)
            b2 = self.board.apply_move(pawn_id, direction)
            # If the move was illegal, the board would be unchanged
            self.assertFalse(
                np.array_equal(b2.grid, self.board.grid),
                f"({pawn_id},{direction}) produced no board change (likely illegal)"
            )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_near_win_state_does_not_crash(self):
        """Board where player can win in one move — agent should not crash."""
        b = make_board([
            [1, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 5, 0, 6, 4],
        ])
        agent = DQNAgent(weights_path="/nonexistent/weights.npz")
        result = agent.get_move(b, 1)
        self.assertIsNotNone(result)

    def test_late_game_board_does_not_crash(self):
        """Board close to end-game with few moves available."""
        b = make_board([
            [1, 2, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [4, 5, 6, 0, 0],
        ])
        agent = DQNAgent(weights_path="/nonexistent/weights.npz")
        result = agent.get_move(b, 1)
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
