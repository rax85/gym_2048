"""A Gym environment for playing the game 2048."""
# pylint: disable=too-many-instance-attributes, duplicate-code
import random

from typing import Any, Tuple, Dict, Optional

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np
import numpy.typing as npt
import numba

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

GRID_SIZE = (4, 4)
PADDING_PX = 8
SQUARE_PX = 64

RECT_COLORS = {
    0:    (142, 142, 147),
    2:    (0,   113, 164),
    4:    (178, 80,  0),
    8:    (0,   122, 255),
    16:   (52,  199, 89),
    32:   (88,  86,  214),
    64:   (255, 149, 0),
    128:  (255, 45,  85),
    256:  (175, 82,  222),
    512:  (255, 59,  48),
    1024: (90,  200, 250),
    2048: (255, 204, 0)
}

CANVAS_SIZE = (GRID_SIZE[0] * (SQUARE_PX + PADDING_PX),
               GRID_SIZE[1] * (SQUARE_PX + PADDING_PX))


@numba.jit(nopython=True)
def _pack_jit(a: npt.NDArray[np.int32]) -> Tuple[npt.NDArray[np.int32], int]:
    non_zeros = a[a != 0]
    out = np.zeros(a.shape, dtype=a.dtype)
    reward = 0
    idx = 0
    i = 0
    length = len(non_zeros)
    while i < length:
        val = non_zeros[i]
        if i + 1 < length and val == non_zeros[i+1]:
            merged_val = val * 2
            out[idx] = merged_val
            reward += merged_val
            i += 2
        else:
            out[idx] = val
            i += 1
        idx += 1
    return out, reward


@numba.jit(nopython=True)
def _can_pack_or_slide_jit(a: npt.NDArray[np.int32]) -> bool:
    length = len(a)
    last_nonzero_val = -1
    for i in range(length):
        val = a[i]
        if val != 0:
            if last_nonzero_val == val:
                return True
            last_nonzero_val = val
    
    has_nonzero = False
    for i in range(length):
        val = a[i]
        if val != 0:
            has_nonzero = True
        else:
            if has_nonzero:
                return True
    return False


@numba.jit(nopython=True)
def _get_valid_moves_jit(grid: npt.NDArray[np.int32]) -> npt.NDArray[np.int32]:
    moves = np.zeros(4, dtype=np.int32)
    
    # UP (0)
    can_up = False
    for i in range(4):
        col = grid[i] # View of column i
        col_flipped = col[::-1]
        if _can_pack_or_slide_jit(col_flipped):
            can_up = True
            break
    moves[0] = 1 if can_up else 0
    
    # DOWN (1)
    can_down = False
    for i in range(4):
        col = grid[i]
        if _can_pack_or_slide_jit(col):
            can_down = True
            break
    moves[1] = 1 if can_down else 0
    
    # LEFT (2)
    can_left = False
    for i in range(4):
        row = grid[:, i]
        row_flipped = row[::-1]
        if _can_pack_or_slide_jit(row_flipped):
            can_left = True
            break
    moves[2] = 1 if can_left else 0
    
    # RIGHT (3)
    can_right = False
    for i in range(4):
        row = grid[:, i]
        if _can_pack_or_slide_jit(row):
            can_right = True
            break
    moves[3] = 1 if can_right else 0
    
    return moves


@numba.jit(nopython=True)
def _merge_jit(grid: npt.NDArray[np.int32], action: int) -> int:
    reward = 0
    if action == 0: # UP
        for i in range(4):
            col = grid[i]
            packed, r = _pack_jit(col)
            reward += r
            grid[i] = packed
    elif action == 1: # DOWN
        for i in range(4):
            col = grid[i]
            col_flipped = col[::-1]
            packed, r = _pack_jit(col_flipped)
            reward += r
            grid[i] = packed[::-1]
    elif action == 2: # LEFT
        for i in range(4):
            row = grid[:, i]
            packed, r = _pack_jit(row)
            reward += r
            grid[:, i] = packed
    elif action == 3: # RIGHT
        for i in range(4):
            row = grid[:, i]
            row_flipped = row[::-1]
            packed, r = _pack_jit(row_flipped)
            reward += r
            grid[:, i] = packed[::-1]
    return reward


class Gym2048Env(gym.Env):
    """A Gym environment for playing the game 2048."""
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, render_mode: Optional[str] = None) -> None:
        super().__init__()
        self.render_mode = render_mode
        font_properties = font_manager.FontProperties(family='monospace', weight='bold')
        font_file = font_manager.findfont(font_properties)
        logging.info('Loading font from %s', font_file)
        self._font = ImageFont.truetype(font_file, 24)

        self._render_cache = {}
        for val in RECT_COLORS:
            self._generate_tile(val)

        observation_shape = [CANVAS_SIZE[1], CANVAS_SIZE[0], 3]
        n_actions = 4 # up, down, left, right.
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=0, high=1.0, shape=observation_shape, dtype=np.float32),
            'valid_mask': spaces.Box(low=0, high=1, shape=[n_actions], dtype=np.int32)
        })
        
        # Pre-calculate slices for rendering
        self._grid_slices = []
        for x in range(GRID_SIZE[0]):
            for y in range(GRID_SIZE[1]):
                rx = int(x * (SQUARE_PX + PADDING_PX) + PADDING_PX / 2)
                ry = int(y * (SQUARE_PX + PADDING_PX) + PADDING_PX / 2)
                self._grid_slices.append((slice(ry, ry+SQUARE_PX), slice(rx, rx+SQUARE_PX)))
        
        # Pre-calculate positions for spawning
        self._all_positions = [(y, x) for y in range(GRID_SIZE[0]) for x in range(GRID_SIZE[1])]

        self._background = np.full((CANVAS_SIZE[1], CANVAS_SIZE[0], 3), 255, dtype=np.uint8).astype(np.float32) / 256.0
        self._current_observation = self._background.copy()
        self.reset()
        logging.info('Canvas size is %s', CANVAS_SIZE)

    def step(self, action: int) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        reward = 0
        valid_moves = _get_valid_moves_jit(self._grid)
        if valid_moves[action] == 1:
            reward = _merge_jit(self._grid, action)
            self._score += reward
            self._random_spawn()
            self._render()
        else:
            reward = -32
        observation, terminated = self._create_observation()
        truncated = False
        return observation, float(reward), terminated, truncated, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        super().reset(seed=seed)
        self._grid = np.zeros(GRID_SIZE, dtype=np.int32)
        self._score = 0
        self._current_observation = self._background.copy()
        if options is None or options != 'nospawn':
            self._random_spawn()
            self._random_spawn()
        self._render()
        observation, _ = self._create_observation()
        return observation, {}

    def _random_spawn(self) -> bool:
        candidates = [pos for pos in self._all_positions if self._grid[pos] == 0]
        if len(candidates) > 0:
            y, x = random.choice(candidates)
            self._grid[y, x] = random.choice([2, 4])
        return len(candidates) == 0

    def _generate_tile(self, val: int) -> None:
        color = RECT_COLORS[val]
        img = Image.new('RGB', (SQUARE_PX, SQUARE_PX), color=color)
        draw = ImageDraw.Draw(img)
        if val > 0:
            draw.text((8, 18), f'{val}', fill='black', font=self._font)
        self._render_cache[val] = np.array(img).astype(np.float32) / 256.0

    def _render(self) -> None:
        for val, (s_y, s_x) in zip(self._grid.flat, self._grid_slices):
            self._current_observation[s_y, s_x] = self._render_cache[val]

    def _create_observation(self, valid_moves: Optional[npt.NDArray[np.int32]] = None) -> Tuple[Dict[str, Any], bool]:
        if valid_moves is None:
            valid_moves = _get_valid_moves_jit(self._grid)
        done = np.count_nonzero(valid_moves) == 0
        return {
            'observation': self._current_observation.copy(),
            'valid_mask': valid_moves
        }, done

    def render(self) -> Optional[npt.NDArray[np.uint8]]:
        return (self._current_observation * 256).astype(np.uint8)

    def _pack(self, a: npt.ArrayLike) -> Tuple[npt.NDArray[np.int32], int]:
        return _pack_jit(np.asarray(a))

    def _can_pack_or_slide(self, a: npt.ArrayLike) -> bool:
        return _can_pack_or_slide_jit(np.asarray(a))

    def _can_move(self, action: int) -> bool:
        return _get_valid_moves_jit(self._grid)[action] == 1

    def close(self) -> None:
        pass