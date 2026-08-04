"""
Benchmark script for all Envpack Gymnasium environments.
Measures average step time and steps per second (SPS) across all registered environments.
"""

import sys
import time
import argparse
import gymnasium as gym
import envpack


def benchmark_env(env_id: str, num_steps: int = 1000, warmup_steps: int = 50):
    try:
        env = gym.make(env_id)
    except Exception as e:
        print(f"Skipping {env_id}: {e}")
        return None

    obs, _ = env.reset()

    # Warmup
    for _ in range(warmup_steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()

    # Benchmark
    start_time = time.time()
    for _ in range(num_steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            env.reset()

    end_time = time.time()
    total_time = end_time - start_time
    avg_step_time = total_time / num_steps
    sps = num_steps / total_time
    env.close()

    return {
        "env_id": env_id,
        "total_time": total_time,
        "avg_step_time": avg_step_time,
        "sps": sps,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Envpack environments")
    parser.add_argument("--env", type=str, default=None, help="Specific env ID (e.g. envpack/2048-v0)")
    parser.add_argument("--steps", type=int, default=1000, help="Number of benchmark steps per env")
    args = parser.parse_args()

    registered_envs = [spec_id for spec_id in gym.envs.registry.keys() if spec_id.startswith("envpack/")]

    if args.env:
        target_envs = [args.env]
    else:
        target_envs = registered_envs

    print(f"Benchmarking {len(target_envs)} environments ({args.steps} steps each)...\n")
    print(f"{'Environment':<30} | {'SPS':<10} | {'Avg Step Time':<15}")
    print("-" * 60)

    for env_id in target_envs:
        res = benchmark_env(env_id, num_steps=args.steps)
        if res:
            print(f"{res['env_id']:<30} | {res['sps']:<10.1f} | {res['avg_step_time']*1000:<12.3f} ms")


if __name__ == "__main__":
    main()
