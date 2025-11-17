"""A Gym environment for playing the game 2048."""
# pylint: disable=too-many-instance-attributes, duplicate-code
import random

import gymnasium as gym
from absl import logging
from gymnasium import spaces
from matplotlib import font_manager
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy as np

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


class Gym2048Env(gym.Env):
    """A Gym environment for playing the game 2048."""
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        font_properties = font_manager.FontProperties(family='monospace', weight='bold')
        font_file = font_manager.findfont(font_properties)
        logging.info('Loading font from %s', font_file)
        self._font = ImageFont.truetype(font_file, 24)

        observation_shape = [CANVAS_SIZE[0], CANVAS_SIZE[1], 3]
        n_actions = 4 # up, down, left, right.
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Dict({
            'observation': spaces.Box(low=0, high=1.0, shape=observation_shape, dtype=np.float32),
            'valid_mask': spaces.Box(low=0, high=1, shape=[n_actions], dtype=np.int32)
        })
        self._canvas = Image.new(mode='RGB', size=CANVAS_SIZE, color='white')
        self._current_observation = np.array(self._canvas)
        self.reset()
        logging.info('Canvas size is %s', CANVAS_SIZE)

    def step(self, action):
        reward = 0
        valid_moves = self._valid_moves()
        if valid_moves[action] == 1:
            if action in (LEFT, RIGHT):
                reward = self._merge_left_right(action == LEFT)
            elif action in (UP, DOWN):
                reward = self._merge_up_down(action == UP)
            self._score += reward
            self._random_spawn()
            self._render()
        else:
            reward = -32
        observation, terminated = self._create_observation()
        truncated = False
        return observation, reward, terminated, truncated, {}

    def _can_pack_or_slide(self, a):
        a = np.array(a)
        packed = a[a != 0]
        delta_len = len(a) - len(packed)
        repadded = np.pad(packed, pad_width=(delta_len, 0))
        if delta_len > 0 and not np.array_equal(a, repadded):
            return True # Can slide
        for i in range(0, len(packed) - 1):
            if packed[i] == packed[i + 1]:
                return True
        return False

    def _can_move(self, direction):
        can_move = False
        if direction in (UP, DOWN):
            for y in range(GRID_SIZE[0]):
                cur_slice = self._grid[y] if (direction == DOWN) else np.flip(self._grid[y])
                can_move |= self._can_pack_or_slide(cur_slice)
        else:
            for x in range(GRID_SIZE[1]):
                cur_slice = self._grid[:, x] if (direction == RIGHT) else np.flip(self._grid[:, x])
                can_move |= self._can_pack_or_slide(cur_slice)
        return can_move

    def _valid_moves(self):
        valid_moves = [
            1 if self._can_move(UP) else 0,
            1 if self._can_move(DOWN) else 0,
            1 if self._can_move(LEFT) else 0,
            1 if self._can_move(RIGHT) else 0
        ]
        return valid_moves

    def _pack(self, vals):
        reward = 0.0
        # Remove zeros.
        a = np.array(vals)
        a = a[a != 0]
        # Add adjacent elements.
        length = len(a)
        for i in range(0, length):
            if i + 1 != length and a[i] == a[i + 1]:
                a[i] *= 2
                a[i + 1] = 0
                reward += a[i]
        # Remove zeros again and expand to the original size.
        a = a[a != 0]
        num_pad = len(vals) - len(a)
        return np.pad(a, pad_width=(0, num_pad)), reward

    def _merge_up_down(self, is_up):
        reward = 0
        for y in range(GRID_SIZE[0]):
            cur_slice = self._grid[y] if is_up else np.flip(self._grid[y])
            packed, cur_reward = self._pack(cur_slice)
            reward += cur_reward
            self._grid[y] = packed if is_up else np.flip(packed)
        return reward

    def _merge_left_right(self, is_left):
        reward = 0
        for x in range(GRID_SIZE[1]):
            cur_slice = self._grid[:, x] if is_left else np.flip(self._grid[:, x])
            packed, cur_reward = self._pack(cur_slice)
            reward += cur_reward
            self._grid[:, x] = packed if is_left else np.flip(packed)
        return reward

    def reset(self, seed=None, options=None): # pylint: disable=arguments-differ
        super().reset(seed=seed)
        self._grid = np.zeros(GRID_SIZE, dtype=np.int32)
        self._score = 0
        self._canvas = Image.new(mode='RGB', size=CANVAS_SIZE, color='white')
        self._current_observation = np.array(self._canvas)
        if options is None or options != 'nospawn':
            self._random_spawn()
            self._random_spawn()
        self._render()
        observation, _ = self._create_observation()
        return observation, {}

    def _random_spawn(self):
        candidates = []
        for y in range(GRID_SIZE[0]):
            for x in range(GRID_SIZE[1]):
                if self._grid[y, x] == 0:
                    candidates.append((y, x))
        if len(candidates) > 0:
            y, x = random.choice(candidates)
            self._grid[y, x] = random.choice([2, 4])
        return len(candidates) == 0

    def _render(self):
        xmax, ymax = GRID_SIZE
        draw = ImageDraw.Draw(self._canvas)
        for x in range(xmax):
            for y in range(ymax):
                rx = x * (SQUARE_PX + PADDING_PX) + PADDING_PX / 2
                ry = y * (SQUARE_PX + PADDING_PX) + PADDING_PX / 2
                val = self._grid[x, y]
                draw.rectangle(((rx, ry), (rx + SQUARE_PX, ry + SQUARE_PX)), fill=RECT_COLORS[val])
                if val > 0:
                    draw.text((rx + 8, ry + 18), f'{val}', fill='black', font=self._font)
        self._current_observation = np.array(self._canvas)

    def _create_observation(self):
        valid_moves = np.asarray(self._valid_moves(), dtype=np.int32)
        done = np.count_nonzero(valid_moves) == 0
        return {
            'observation': self._current_observation.astype(np.float32) / 256.0,
            'valid_mask': valid_moves
        }, done

    def render(self):
        return self._current_observation

    def close(self):
        self._canvas.close()
