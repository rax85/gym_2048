"""Gym 2048 environment."""

from gymnasium.envs.registration import register

register(
    id="gym_2048/2048-v0",
    entry_point="gym_2048.envs:Gym2048Env",
    max_episode_steps=300,
)
