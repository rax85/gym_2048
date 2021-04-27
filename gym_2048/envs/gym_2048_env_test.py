import unittest

import numpy as np

from PIL import Image

from . import gym_2048_env
from . import Gym2048Env

class TestGym2048Env(unittest.TestCase):
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

    def test_move_nomerge_l(self):
        env = Gym2048Env()
        env._grid[0, 2] = 0
        env._grid[1, 2] = 128
        env._grid[2, 2] = 0
        env._grid[3, 2] = 256
        self._save(env, 'move_nomerge_l0')
        _, reward, done, _ = env.step(gym_2048_env.LEFT)
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
        _, reward, done, _ = env.step(gym_2048_env.RIGHT)
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
        _, reward, done, _ = env.step(gym_2048_env.UP)
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
        _, reward, done, _ = env.step(gym_2048_env.DOWN)
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
        _, reward, done, _ = env.step(gym_2048_env.LEFT)
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
        _, reward, done, _ = env.step(gym_2048_env.RIGHT)
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
        _, reward, done, _ = env.step(gym_2048_env.UP)
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
        _, reward, done, _ = env.step(gym_2048_env.DOWN)
        self._save(env, 'move_merge_d1')

        self.assertEqual(env._grid[1, 3], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 256)

    def test_move_doublemerge(self):
        env = Gym2048Env()
        env._grid[1, 0] = 128
        env._grid[1, 1] = 128
        env._grid[1, 2] = 128
        env._grid[1, 3] = 128
        self._save(env, 'move_merge_double0')
        _, reward, done, _ = env.step(gym_2048_env.DOWN)
        self._save(env, 'move_merge_double1')

        self.assertEqual(env._grid[1, 2], 256)
        self.assertEqual(env._grid[1, 3], 256)
        self.assertFalse(done)
        self.assertEqual(reward, 512)

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

    def test_done(self):
        env = Gym2048Env()
        env._grid = np.asarray(
            [[16,   4, 256, 32],
             [ 8,  32,  64,  4],
             [32, 128,  16,  2],
             [16,   8,   2,  2]]
        ).transpose()
        self._save(env, 'move_merge_done0')
        _, reward, done, _ = env.step(gym_2048_env.RIGHT)
        self._save(env, 'move_merge_done1')

        self.assertEqual(reward, 4)
        self.assertTrue(done)

if __name__ == "__main__":
    unittest.main()