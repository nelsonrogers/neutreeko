# Neutreeko

A Python implementation of the [Neutreeko](https://en.wikipedia.org/wiki/Neutreeko) board game. You play against an AI opponent: the hardest level is a **Deep Q-Network (DQN)** trained through self-play, and the easier levels use alpha-beta minimax search.

The game runs as a native Pygame application and can also be compiled to WebAssembly with [Pygbag](https://pygame-web.github.io/) so it can be played in a browser.

---

## Background

This started as a school programming project during my *classes préparatoires* (CPGE), the two-year program that prepares students for the competitive entrance exams to French engineering schools. Building this game is what really taught me how to code.

The original version is still in the repo as [NeutreekoV1.py](NeutreekoV1.py) (written in French, with matplotlib for graphics). Its AI was a set of rules I came up with myself:

- **Level 1:** play the winning move if there is one, otherwise move at random.
- **Level 2:** try random moves, recursively re-rolling (up to 400 times) until one doesn't let the player win on the next turn.
- **Level 3:** same as level 2, but also prefer moves that put two of its pieces side by side.

[NeutreekoV2.py](NeutreekoV2.py) is a later rewrite of the same game in English, using classes.

Since then I've learned a lot more about programming and got interested in AI, especially deep learning. Coming back to this game felt like a good way to put that into practice: rebuild it with a cleaner structure, replace the hand-written rules with proper search, and see how strong an opponent I can get with reinforcement learning. It's an ongoing project that I use to sharpen my skills as an AI engineer.

---

## Game rules

Neutreeko is a two-player abstract strategy game played on a 5×5 board. Each player controls **3 pieces**.

- **Objective:** Be the first to align your 3 pieces in a straight line — horizontal, vertical, or diagonal — with no gaps between them.
- **Movement:** On your turn, choose one of your pieces and a direction (8 possible: 4 orthogonal + 4 diagonal). The piece **slides** in that direction until it hits the board edge or another piece, then stops.
- There are no captures. The game ends the moment a player's 3 pieces are adjacent in a line.

**Initial layout:**

```
. 1 . 2 .
. . 4 . .
. . . . .
. . 3 . .
. 5 . 6 .
```

`1,2,3` = AI / Player 1 (red).  `4,5,6` = Human / Player 2 (cyan).

---

## Project structure

```
neutreeko/
├── game/
│   ├── board.py          # Board class: state, move gen, win detection, encoding
│   └── constants.py      # Board size, directions, colors, window dimensions
│
├── ai/
│   ├── minimax.py        # Alpha-beta search (Easy depth=1, Medium depth=4)
│   └── dqn_agent.py      # Numpy-only DQN inference — loads assets/weights.npz
│
├── training/
│   ├── model.py          # PyTorch DQN model definition (75 → 128 → 64 → 24)
│   ├── train_dqn.py      # Self-play Double DQN training loop, 20 k episodes
│   └── export_weights.py # Converts .pth checkpoint → assets/weights.npz
│
├── ui/
│   ├── renderer.py       # Board, pieces, selection glow, move hints, animation
│   └── screens.py        # Menu screen, win/lose overlay, Button widget
│
├── assets/
│   └── weights.npz       # Pre-trained DQN weights (numpy format, WASM-safe)
│
├── tests/
│   ├── test_board.py     # 47 unit tests for Board (moves, wins, encoding, …)
│   ├── test_minimax.py   # 14 tests for alpha-beta correctness and heuristic
│   └── test_dqn_agent.py # 20 tests for DQN init, forward pass, and move legality
│
├── main.py               # Async game loop — Pygbag/WebAssembly compatible
├── requirements.txt
├── NeutreekoV1.py        # Original CPGE version (matplotlib, French), kept for reference
└── NeutreekoV2.py        # Later rewrite (matplotlib, English OOP), kept for reference
```

---

## Installation

**Python 3.9+** is required.

```bash
pip install pygame numpy          # run the game
pip install torch                 # only needed to train the DQN
pip install pygbag                # only needed to build for the web
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

---

## Running the game locally

```bash
python main.py
```

A window opens showing the difficulty menu. Select Easy, Medium, or Hard. The AI plays as the red pieces; you play as cyan. Click a piece to select it — valid destinations are shown as yellow dots — then click a destination to move.

---

## Difficulty levels

| Level  | AI implementation               | Strength |
|--------|---------------------------------|----------|
| Easy   | Alpha-beta minimax, depth 1     | Wins immediately if possible; otherwise random |
| Medium | Alpha-beta minimax, depth 4     | Looks 4 plies ahead; uses proximity + alignment heuristic |
| Hard   | Deep Q-Network (pre-trained)    | Trained via 20 k episodes of self-play |

If `assets/weights.npz` is not present, Hard falls back to random moves until you train the model (see below).

---

## Training the DQN

### 1. Train

```bash
# Start fresh (20 000 episodes by default)
python -m training.train_dqn

# Resume from a checkpoint and run 20 000 more episodes
python -m training.train_dqn --resume training/checkpoints/dqn_20000.pth

# Resume and specify exactly how many additional episodes to run
python -m training.train_dqn --resume training/checkpoints/dqn_20000.pth --episodes 40000
```

Runs self-play Double DQN with experience replay. Training takes roughly **15–30 minutes on CPU** per 20 000 episodes; significantly less on GPU.

Checkpoints are saved every 5 000 episodes to `training/checkpoints/` and are named by **absolute** episode count, so resumed runs produce new files rather than overwriting old ones (e.g. `dqn_20000.pth` → `dqn_25000.pth` → …).

**What is saved in each checkpoint:** network weights, target network weights, Adam optimizer momentum, and the global step counter (`total_steps`). This means the epsilon schedule and optimizer state resume exactly where they left off. The replay buffer is not persisted (it is in-memory only) but refills within the first few hundred steps — no meaningful impact on training quality.

**Resuming from an old checkpoint** (saved before this resume feature was added): the old format is detected automatically. The weights load correctly; epsilon and optimizer state restart from zero, so expect heavier random exploration for the first ~2 000 steps before epsilon decays back down.

Key hyper-parameters (edit at the top of [training/train_dqn.py](training/train_dqn.py)):

| Parameter        | Default | Description |
|------------------|---------|-------------|
| `EPISODES`       | 20 000  | Total self-play games |
| `BATCH_SIZE`     | 64      | SGD mini-batch size |
| `GAMMA`          | 0.99    | Discount factor |
| `LR`             | 1e-3    | Adam learning rate |
| `REPLAY_CAPACITY`| 50 000  | Experience replay buffer size |
| `EPS_START/END`  | 1.0/0.05| Epsilon-greedy exploration range |
| `EPS_DECAY`      | 10 000  | Steps to decay epsilon (exponential) |
| `TARGET_SYNC`    | 500     | Steps between target-network syncs |

### 2. Export weights

At the end of training `assets/weights.npz` is exported automatically. To export a specific checkpoint manually:

```bash
python -m training.export_weights training/checkpoints/dqn_10000.pth assets/weights.npz
```

### Network architecture

```
Input: 75 floats
  └─ 5×5 board encoded as 3 one-hot channels:
       channel 0 = empty cells
       channel 1 = current player's pieces
       channel 2 = opponent's pieces

Linear(75 → 128) → ReLU
Linear(128 → 64) → ReLU
Linear(64 → 24)

Output: Q-value for each of 24 actions
  └─ 3 pieces × 8 directions
     Illegal actions masked to -∞ before argmax
```

---

## Building for the web (WebAssembly)

The game loop uses `await asyncio.sleep(0)` so it is fully compatible with [Pygbag](https://pygame-web.github.io/).

```bash
# Serve locally (opens browser at localhost:8000)
python -m pygbag main.py

# Build a static bundle
python -m pygbag --build main.py
```

The build outputs to `web-build/`. Copy the directory to your static site and embed it with:

```html
<iframe
  src="/neutreeko/index.html"
  width="520"
  height="640"
  style="border:none;">
</iframe>
```

---

## Running the tests

```bash
python -m unittest discover -s tests -v
```

Expected output: **81 tests, 0 failures**.

### Test coverage summary

**`tests/test_board.py`** (47 tests)
- `TestBoardInit` — default grid, custom grid, copy independence, dtype
- `TestFindPawn` — locates all 6 pieces, correct initial positions, missing piece returns `None`
- `TestSlideDestination` — slides to wall, stops at piece, no-move when blocked, diagonal
- `TestLegalMoves` — initial count (14 per player), format, destination is empty, only own pieces
- `TestApplyMove` — destination set, source cleared, original board unchanged, pawn ID preserved
- `TestCheckWinner` — horizontal / vertical / diagonal / anti-diagonal wins for each player; two-adjacent and spaced pieces are not wins
- `TestEncode` — shape (75,), dtype float32, binary values, channel sums, perspective symmetry
- `TestActionIndexing` — `action_to_move` round-trip, mask length, mask count matches `get_legal_moves`, every legal move is in mask

**`tests/test_minimax.py`** (14 tests)
- `TestGetBestMoveContract` — return type, pawn ownership, direction validity, legality at depth 1 and 4
- `TestWinsInOneMove` — horizontal, vertical, diagonal 1-move wins found at depth 1 and depth 4
- `TestAvoidsImmediateLoss` — depth-4 search does not leave player 2 in a winning position
- `TestEvaluate` — winning position scores higher than neutral; score is antisymmetric; tighter pawns score higher

**`tests/test_dqn_agent.py`** (20 tests)
- `TestDQNAgentInit` — `loaded=False` without file, `loaded=True` with valid file, weight shapes, corrupted file handled gracefully
- `TestForwardPass` — output shape (24,), dtype float32, ReLU correctness, determinism, different inputs produce different outputs
- `TestGetMove` — returns tuple, pawn in correct set, direction valid, legal at depth with and without weights (run 20× for random coverage), action masking never picks an illegal action
- `TestEdgeCases` — near-win board and late-game board do not crash

---

## Tech stack

| Layer      | Technology |
|------------|------------|
| UI / game loop | [Pygame](https://www.pygame.org/) 2.x |
| WebAssembly build | [Pygbag](https://pygame-web.github.io/) |
| Numerical core | [NumPy](https://numpy.org/) |
| DQN training | [PyTorch](https://pytorch.org/) |
| In-game inference | Pure NumPy (no PyTorch dependency at runtime) |
| Tests | Python `unittest` (stdlib) |
