"""
Neutreeko — main entry point.
Compatible with Pygbag (WebAssembly) via asyncio game loop.

Run locally:  python main.py
Build WASM:   python -m pygbag --build main.py
"""
import asyncio
import time
import pygame

from game.board import Board
from game.constants import (
    WINDOW_W, WINDOW_H, BOARD_OFFSET_X, BOARD_OFFSET_Y,
    CELL_SIZE, BOARD_SIZE, BG_COLOR, TEXT_COLOR,
    P1_COLOR, P2_COLOR,
)
from ui.renderer import draw_board, draw_status_bar, cell_rect
from ui.screens import (
    draw_menu, draw_win_screen,
    build_menu_buttons, build_postgame_buttons,
)
from ai.minimax import get_best_move
from ai.dqn_agent import DQNAgent

# ------------------------------------------------------------------
# Difficulty → AI config
# ------------------------------------------------------------------
DIFFICULTY = {
    "Easy":   {"type": "minimax", "depth": 1},
    "Medium": {"type": "minimax", "depth": 4},
    "Hard":   {"type": "dqn"},
}

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def get_cell_at(px, py):
    """Convert pixel (px, py) to board (row, col), or None if outside."""
    col = (px - BOARD_OFFSET_X) // CELL_SIZE
    row = (py - BOARD_OFFSET_Y) // CELL_SIZE
    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row, col
    return None


class GameState:
    """All mutable game state in one place."""

    def __init__(self, difficulty_name, dqn_agent):
        self.board = Board()
        self.difficulty = difficulty_name
        self.dqn_agent = dqn_agent
        self.current_player = 2   # human is player 2, AI is player 1
        self.selected_pawn = None
        self.legal_moves = []     # moves for selected pawn
        self.winner = 0
        self.anim = None          # animation dict or None
        self.anim_start = 0.0
        self.anim_duration = 0.3  # seconds

    @property
    def is_human_turn(self):
        return self.current_player == 2

    def select_pawn(self, row, col):
        pawn_id = int(self.board.grid[row, col])
        if pawn_id in (4, 5, 6):  # player 2's pieces
            self.selected_pawn = pawn_id
            all_moves = self.board.get_legal_moves(2)
            self.legal_moves = [m for m in all_moves if m[0] == pawn_id]
        else:
            self.selected_pawn = None
            self.legal_moves = []

    def try_move(self, row, col):
        """Try to move selected pawn to (row, col). Returns True if move made."""
        for pawn_id, direction, dr, dc in self.legal_moves:
            if dr == row and dc == col:
                src = self.board.find_pawn(pawn_id)
                self._start_anim(pawn_id, src, (row, col))
                self.board = self.board.apply_move(pawn_id, direction)
                self.selected_pawn = None
                self.legal_moves = []
                self.winner = self.board.check_winner()
                if not self.winner:
                    self.current_player = 1
                return True
        return False

    def ai_move(self):
        """Compute and apply the AI's move (player 1)."""
        cfg = DIFFICULTY[self.difficulty]
        if cfg["type"] == "dqn":
            result = self.dqn_agent.get_move(self.board, 1)
        else:
            result = get_best_move(self.board, 1, cfg["depth"])

        if result is None:
            self.current_player = 2
            return

        pawn_id, direction = result
        src = self.board.find_pawn(pawn_id)
        dest = self.board.slide_destination(*src, direction) if src else None

        if dest:
            self._start_anim(pawn_id, src, dest)

        self.board = self.board.apply_move(pawn_id, direction)
        self.winner = self.board.check_winner()
        if not self.winner:
            self.current_player = 2

    def _start_anim(self, pawn_id, from_pos, to_pos):
        self.anim = {
            "pawn_id": pawn_id,
            "from_pos": from_pos,
            "to_pos": to_pos,
            "progress": 0.0,
        }
        self.anim_start = time.monotonic()

    def update_anim(self):
        if self.anim is None:
            return
        elapsed = time.monotonic() - self.anim_start
        self.anim["progress"] = min(1.0, elapsed / self.anim_duration)
        if self.anim["progress"] >= 1.0:
            self.anim = None


# ------------------------------------------------------------------
# Async main loop (Pygbag compatible)
# ------------------------------------------------------------------

async def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("Neutreeko")

    # Fonts
    font_large  = pygame.font.SysFont("Arial", 48, bold=True)
    font_medium = pygame.font.SysFont("Arial", 28)
    font_small  = pygame.font.SysFont("Arial", 18)

    # Preload DQN weights (may silently fail if weights.npz missing)
    dqn_agent = DQNAgent()

    # Build menu buttons
    menu_buttons = build_menu_buttons()

    # State machine
    phase = "menu"        # "menu" | "playing" | "ai_turn" | "win"
    game = None
    play_again_btn, menu_btn = build_postgame_buttons()
    difficulty_name = "Easy"

    clock = pygame.time.Clock()
    pygame.event.clear()  # discard any events that queued during initialisation

    while True:
        mouse_pos = pygame.mouse.get_pos()
        events = pygame.event.get()

        for event in events:
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            # ---- MENU ----
            if phase == "menu":
                for btn in menu_buttons:
                    if btn.is_clicked(event):
                        difficulty_name = btn.label
                        game = GameState(difficulty_name, dqn_agent)
                        phase = "ai_turn" if not game.is_human_turn else "playing"

            # ---- PLAYING (human turn) ----
            elif phase == "playing":
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    cell = get_cell_at(*event.pos)
                    if cell:
                        row, col = cell
                        if game.selected_pawn is not None:
                            # Try move first
                            moved = game.try_move(row, col)
                            if not moved:
                                # Re-select or deselect
                                game.select_pawn(row, col)
                        else:
                            game.select_pawn(row, col)

                    if game.winner:
                        phase = "win"
                    elif not game.is_human_turn:
                        phase = "ai_turn"

            # ---- WIN SCREEN ----
            elif phase == "win":
                if play_again_btn.is_clicked(event):
                    game = GameState(difficulty_name, dqn_agent)
                    phase = "ai_turn" if not game.is_human_turn else "playing"
                if menu_btn.is_clicked(event):
                    phase = "menu"
                    game = None

        # ---- AI TURN (non-blocking: runs once per frame) ----
        if phase == "ai_turn" and game is not None:
            if game.anim is None:  # wait for animation to finish
                game.ai_move()
                pygame.event.clear()  # discard clicks that landed during AI computation
                if game.winner:
                    phase = "win"
                else:
                    phase = "playing"

        # ---- UPDATE ANIMATION ----
        if game:
            game.update_anim()

        # ---- DRAW ----
        if phase == "menu":
            draw_menu(screen, font_large, font_medium, font_small,
                      menu_buttons, mouse_pos)

        elif phase in ("playing", "ai_turn") and game:
            draw_board(
                screen, game.board,
                selected_pawn=game.selected_pawn,
                legal_moves=game.legal_moves if game.selected_pawn else None,
                anim=game.anim,
            )
            # Status bar
            if game.is_human_turn or phase == "playing":
                msg = "Your turn"
                color = P2_COLOR
            else:
                msg = f"AI thinking...  [{difficulty_name}]"
                color = P1_COLOR
            draw_status_bar(screen, font_medium, msg, color)

        elif phase == "win" and game:
            draw_board(screen, game.board, anim=None)
            if game.winner == 1:
                label = "AI WINS"
                wcolor = P1_COLOR
            else:
                label = "YOU WIN!"
                wcolor = P2_COLOR
            draw_win_screen(screen, font_large, font_medium, font_small,
                            label, wcolor,
                            play_again_btn, menu_btn, mouse_pos)

        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)  # yield to browser event loop (Pygbag)


if __name__ == "__main__":
    asyncio.run(main())
