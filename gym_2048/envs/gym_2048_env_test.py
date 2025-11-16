import copy
import unittest

import numpy as np

from PIL import Image

from . import gym_2048_env
from . import Gym2048Env

class TestGym2048Env(unittest.TestCase):
    def test_initial_state(self):
        env = Gym2048Env()
        env.reset()
        self.assertEqual(np.count_nonzero(env._grid), 2)

    def test_pack_nomerge(self):
        vals = [4, 0, 2, 0]
        env = Gym2048Env()
        packed, reward = env._pack(vals)
        self.assertTrue(np.array_equal(packed, [4, 2, 0, 0]))
        self.assertEqual(reward, 0)

    def test_pack_merge_single(self):
        vals = [2, 0, 2, 0]
        env = Gym2048Env()
        packed, reward = env._pack(vals)
        self.assertTrue(np.array_equal(packed, [4, 0, 0, 0]))
        self.assertEqual(reward, 4)

    def test_pack_merge_double(self):
        vals = [2, 2, 2, 2]
        env = Gym2048Env()
        packed, reward = env._pack(vals)
        self.assertTrue(np.array_equal(packed, [4, 4, 0, 0]))
        self.assertEqual(reward, 8)

    def test_pack_merge_slide(self):
        vals = [2, 2, 0, 2]
        env = Gym2048Env()
        packed, reward = env._pack(vals)
        self.assertTrue(np.array_equal(packed, [4, 2, 0, 0]))
        self.assertEqual(reward, 4)

    def _save(self, env, name):
        env._render()
        rgb_data = env.render()
        image = Image.fromarray(rgb_data)
        image.save('/tmp/test_%s.png' % name)

    def test_render(self):
        env = Gym2048Env()
        env._grid[0, 2] = 0
        env._grid[1, 2] = 64
        env._grid[2, 2] = 128
        env._grid[3, 2] = 2048
        self._save(env, 'render')

    def test_move_nomerge_l(self):
        env = Gym2048Env()
        env._grid[0, 2] = 0
        env._grid[1, 2] = 128
        env._grid[2, 2] = 0
        env._grid[3, 2] = 256
        self._save(env, 'move_nomerge_l0')
        _, reward, done, _, _ = env.step(gym_2048_env.LEFT)
        self._save(env, 'move_nomerge_l1')

        self.assertEqual(env._grid[0, 2], 128)
        self.assertEqual(env._grid[1, 2], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 0)

    def test_move_nomerge_r(self):
        env = Gym2048Env()
        env._grid[0, 2] = 128
        env._grid[1, 2] = 0
        env._grid[2, 2] = 256
        env._grid[3, 2] = 0
        self._save(env, 'test_move_nomerge_r0')
        _, reward, done, _, _ = env.step(gym_2048_env.RIGHT)
        self._save(env, 'test_move_nomerge_r1')

        self.assertEqual(env._grid[2, 2], 128)
        self.assertEqual(env._grid[3, 2], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 0)

    def test_move_nomerge_u(self):
        env = Gym2048Env()
        env._grid[1, 0] = 0
        env._grid[1, 1] = 128
        env._grid[1, 2] = 0
        env._grid[1, 3] = 256
        self._save(env, 'test_move_nomerge_u0')
        _, reward, done, _, _ = env.step(gym_2048_env.UP)
        self._save(env, 'test_move_nomerge_u1')

        self.assertEqual(env._grid[1, 0], 128)
        self.assertEqual(env._grid[1, 1], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 0)

    def test_move_nomerge_d(self):
        env = Gym2048Env()
        env._grid[1, 0] = 128
        env._grid[1, 1] = 0
        env._grid[1, 2] = 256
        env._grid[1, 3] = 0
        self._save(env, 'test_move_nomerge_d0')
        _, reward, done, _, _ = env.step(gym_2048_env.DOWN)
        self._save(env, 'test_move_nomerge_d1')

        self.assertEqual(env._grid[1, 2], 128)
        self.assertEqual(env._grid[1, 3], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 0)

    def test_move_merge_l(self):
        env = Gym2048Env()
        env._grid[0, 2] = 0
        env._grid[1, 2] = 128
        env._grid[2, 2] = 0
        env._grid[3, 2] = 128
        self._save(env, 'move_merge_l0')
        _, reward, done, _, _ = env.step(gym_2048_env.LEFT)
        self._save(env, 'move_merge_l1')

        self.assertEqual(env._grid[0, 2], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 256)

    def test_move_merge_r(self):
        env = Gym2048Env()
        env._grid[0, 2] = 128
        env._grid[1, 2] = 0
        env._grid[2, 2] = 128
        env._grid[3, 2] = 0
        self._save(env, 'move_merge_r0')
        _, reward, done, _, _ = env.step(gym_2048_env.RIGHT)
        self._save(env, 'move_merge_r1')

        self.assertEqual(env._grid[3, 2], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 256)

    def test_move_merge_u(self):
        env = Gym2048Env()
        env._grid[1, 0] = 0
        env._grid[1, 1] = 128
        env._grid[1, 2] = 0
        env._grid[1, 3] = 128
        self._save(env, 'move_merge_u0')
        _, reward, done, _, _ = env.step(gym_2048_env.UP)
        self._save(env, 'move_merge_u1')

        self.assertEqual(env._grid[1, 0], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 256)

    def test_move_merge_d(self):
        env = Gym2048Env()
        env._grid[1, 0] = 128
        env._grid[1, 1] = 0
        env._grid[1, 2] = 128
        env._grid[1, 3] = 0
        self._save(env, 'move_merge_d0')
        _, reward, done, _, _ = env.step(gym_2048_env.DOWN)
        self._save(env, 'move_merge_d1')

        self.assertEqual(env._grid[1, 3], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 256)

    def test_move_doublemerge(self):
        env = Gym2048Env()
        env._grid = np.zeros_like(env._grid)
        env._grid[1, 0] = 128
        env._grid[1, 1] = 128
        env._grid[1, 2] = 128
        env._grid[1, 3] = 128
        self._save(env, 'move_merge_double0')
        _, reward, done, _, _ = env.step(gym_2048_env.DOWN)
        self._save(env, 'move_merge_double1')

        self.assertEqual(env._grid[1, 2], 256)
        self.assertEqual(env._grid[1, 3], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 512)

    def test_merge_first_slide_second_u(self):
        env = Gym2048Env()
        env._grid[1, 0] = 4
        env._grid[1, 1] = 2
        env._grid[1, 2] = 2
        env._grid[1, 3] = 2
        self._save(env, 'test_merge_first_slide_second_u0')
        _, reward, done, _, _ = env.step(gym_2048_env.UP)
        self._save(env, 'test_merge_first_slide_second_u1')

        self.assertEqual(env._grid[1, 0], 4)
        self.assertEqual(env._grid[1, 1], 4)
        self.assertEqual(env._grid[1, 2], 2)
        self.assertFalse(done)
        self.assertEqual(reward, 4)

    def test_pack(self):
        env = Gym2048Env()
        actual, reward = env._pack([4, 2, 2, 2])
        self.assertTrue(np.array_equal([4, 4, 2, 0], actual))
        self.assertEqual(reward, 4)

    def test_can_pack_or_slide_pack(self):
        vals = [2, 2, 2, 2]
        env = Gym2048Env()
        self.assertTrue(env._can_pack_or_slide(vals))

    def test_can_pack_or_slide_slide(self):
        vals = [2, 0, 2, 2]
        env = Gym2048Env()
        self.assertTrue(env._can_pack_or_slide(vals))

    def test_can_pack_or_slide_neither(self):
        vals = [128, 64, 32, 16]
        env = Gym2048Env()
        self.assertFalse(env._can_pack_or_slide(vals))

    def test_can_pack_or_slide_leading_zeros(self):
        vals = [0, 0, 2, 4]
        env = Gym2048Env()
        self.assertFalse(env._can_pack_or_slide(vals))

    def test_can_pack_or_slide_leading_zeros_single(self):
        vals = [0, 0, 0, 4]
        env = Gym2048Env()
        self.assertFalse(env._can_pack_or_slide(vals))

    def test_can_pack_or_slide_trailing_zeros(self):
        vals = [2, 4, 0, 0]
        env = Gym2048Env()
        self.assertTrue(env._can_pack_or_slide(vals))

    def test_can_pack_or_slide_leading_and_trailing_zeros(self):
        vals = [0, 4, 0, 0]
        env = Gym2048Env()
        self.assertTrue(env._can_pack_or_slide(vals))

    def test_done(self):
        env = Gym2048Env()
        observation, _ = env.reset()
        self.assertIsNotNone(observation)
        self.assertEqual(observation['observation'].dtype, np.float32)
        self.assertEqual(observation['valid_mask'].dtype, np.int32)
        env._grid = np.asarray(
            [[16,   4, 256, 32],
             [ 8,  32,  64,  4],
             [32, 128,  16,  2],
             [16,   8,   2,  2]]
        ).transpose()
        self._save(env, 'move_merge_done0')
        observation, reward, done, _, _ = env.step(gym_2048_env.RIGHT)
        self._save(env, 'move_merge_done1')

        self.assertEqual(reward, 4)
        self.assertTrue(done)
        self.assertIsNotNone(observation['observation'])
        self.assertIsNotNone(observation['valid_mask'])
        self.assertEqual(observation['observation'].dtype, np.float32)

    def test_real_world1(self):
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[0, 4, 8, 4],
             [0, 2, 8, 2],
             [0, 4, 8, 2],
             [2, 8, 4, 2]]
        ).transpose()
        self._save(env, 'test_real_world1_0')
        _, reward, done, _, _ = env.step(gym_2048_env.UP)
        self._save(env, 'test_real_world1_1')

        self.assertFalse(done)
        self.assertEqual(reward, 20)
        self.assertEqual(env._grid[2, 0], 16)
        self.assertEqual(env._grid[2, 1], 8)
        self.assertEqual(env._grid[2, 2], 4)
        self.assertEqual(env._grid[3, 0], 4)
        self.assertEqual(env._grid[3, 1], 4)
        self.assertEqual(env._grid[3, 2], 2)

    def test_cant_move_r(self):
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[0, 0, 0, 2],
             [0, 0, 0, 2],
             [0, 0, 0, 2],
             [0, 0, 0, 2]]
        ).transpose()
        self.assertTrue(env._can_move(gym_2048_env.UP))
        self.assertTrue(env._can_move(gym_2048_env.DOWN))
        self.assertTrue(env._can_move(gym_2048_env.LEFT))
        print('interesting.......')
        self.assertFalse(env._can_move(gym_2048_env.RIGHT))

    def test_cant_move_u(self):
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[2, 2, 2, 2],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ).transpose()
        print('interesting.......')
        self.assertFalse(env._can_move(gym_2048_env.UP))

    def test_cant_move_l(self):
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[2, 0, 0, 0],
             [2, 0, 0, 0],
             [2, 0, 0, 0],
             [2, 0, 0, 0]]
        ).transpose()
        self.assertFalse(env._can_move(gym_2048_env.LEFT))

    def test_cant_move_d(self):
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [2, 2, 2, 2]]
        ).transpose()
        self.assertFalse(env._can_move(gym_2048_env.DOWN))

    def test_negative_reward(self):
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[0, 0, 0, 2],
             [0, 0, 0, 8],
             [0, 0, 2, 4],
             [0, 0, 0, 4]]
        ).transpose()
        initial_state = copy.deepcopy(env._grid)
        self._save(env, 'test_negative_reward_0')
        _, reward, done, _, _ = env.step(gym_2048_env.RIGHT)
        self._save(env, 'test_negative_reward1')

        self.assertFalse(done)
        self.assertEqual(reward, -32)
        self.assertTrue(np.array_equal(env._grid, initial_state))


    def test_score_accumulation(self):
        env = Gym2048Env()
        env.reset()
        env._grid = np.asarray(
            [[2, 2, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ).transpose()
        self.assertEqual(env._score, 0)
        env.step(gym_2048_env.LEFT)
        self.assertEqual(env._score, 4)
        env._grid = np.asarray(
            [[4, 4, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0],
             [0, 0, 0, 0]]
        ).transpose()
        env.step(gym_2048_env.LEFT)
        self.assertEqual(env._score, 12)


    def test_game_over(self):
        env = Gym2048Env()
        env._grid = np.asarray(
            [[2, 4, 2, 4],
             [4, 2, 4, 2],
             [2, 4, 2, 4],
             [4, 2, 4, 2]]
        )
        observation, reward, terminated, truncated, info = env.step(gym_2048_env.UP)
        self.assertTrue(terminated)


if __name__ == "__main__":
    unittest.main()
