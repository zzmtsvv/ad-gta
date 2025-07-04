from typing import Optional
import multiprocessing as mp
import os
import shutil
from dataclasses import dataclass
import random
from collections import defaultdict

import gymnasium as gym
import numpy as np
import pyrallis
import uuid
from matplotlib import pyplot as plt

import src.envs
# from src.envs.dark_room import train_test_goals
from src.utils.misc import set_seed


TRAIN_GOALS = np.array([
    [2, 6], [4, 0], [1, 6], [8, 7], [8, 2], [4, 7], [0, 3], [1, 7],
    [4, 8], [5, 1], [3, 8], [5, 0], [2, 1], [7, 7], [1, 0], [7, 2],
    [1, 5], [7, 6], [5, 7], [4, 5], [1, 1], [8, 4], [8, 0], [4, 2],
    [1, 4], [5, 4], [2, 3], [4, 6], [3, 4], [0, 4], [0, 7], [7, 3],
    [0, 5], [2, 2], [3, 0], [4, 4], [5, 2], [2, 8], [5, 3], [5, 6],
    [3, 5], [7, 0], [0, 2], [2, 0], [6, 1], [3, 7], [6, 0], [7, 4],
    [7, 5], [4, 3], [8, 5], [3, 1], [8, 6], [8, 8], [3, 3], [7, 8],
    [6, 2], [2, 4], [0, 8], [1, 8], [0, 0], [6, 4], [6, 5], [2, 5]
    ])


def dump_trajectories(savedir, trajectories, goal_pos):
    np.savez(
        os.path.join(savedir, f'learning_history_{goal_pos}_{str(uuid.uuid4())}.npz'),
        states=np.array(trajectories['states'], dtype=float).reshape(-1, 1),
        actions=np.array(trajectories['actions']).reshape(-1, 1),
        rewards=np.array(trajectories['rewards'], dtype=float).reshape(-1, 1),
        dones=np.int32(np.array(trajectories['terminateds']) | np.array(trajectories['truncateds'])).reshape(-1, 1),
        goal=np.array(goal_pos),
    )


@dataclass
class Config:
    seed: Optional[int] = None
    env_name: str = "Dark-Room-9x9-v0"
    num_histories: int = 2424
    num_episodes: int = 100_000
    savedir: str = 'trajectories'
    lr: float = 5e-5
    eps_coef: float = 1.0


class MultiProcQLearningWorker:
    def __init__(self, config: Config):
        self.config = config

    def __call__(self, goal):
        env = gym.make(self.config.env_name, goal_pos=goal)
        os.makedirs(self.config.savedir, exist_ok=True)
        q_table, rewards_history = self.q_learning(
            env,
            lr=self.config.lr,
            eps_coef=self.config.eps_coef,
            num_episodes=self.config.num_episodes,
            savedir=self.config.savedir,
            return_history=True
        )
        return rewards_history
    
    def generate_dataset(self):
        if self.config.seed is not None:
            set_seed(self.config.seed)

        print(len(TRAIN_GOALS), self.config.num_histories)
        assert self.config.num_histories >= len(TRAIN_GOALS)
        goal_inds = np.random.choice(len(TRAIN_GOALS), size=self.config.num_histories - len(TRAIN_GOALS), replace=True)
        # to ensure that at least once all goals are selected
        goals = np.vstack([TRAIN_GOALS, TRAIN_GOALS[goal_inds]])
        assert len(np.unique(goals, axis=0)) >= len(TRAIN_GOALS)

        print("Generating data for goals:")
        print(goals, goals.shape)
        if os.path.exists(self.config.savedir):
            shutil.rmtree(self.config.savedir)

        with mp.Pool(processes=os.cpu_count()) as pool:
            rewards_history = pool.map(MultiProcQLearningWorker(self.config), goals.tolist())
        
        return rewards_history
    
    @staticmethod
    def q_learning(
        env,
        lr=0.01,
        discount=0.9,
        num_episodes=100_000,
        savedir=None,
        seed=None,
        return_history=False,
        eps_coef=0.7
    ):
        trajectories = defaultdict(list)
        Q = np.random.uniform(size=(env.unwrapped.size * env.unwrapped.size, env.action_space.n))

        rewards_history = []
        episode_reward = 0

        num_steps = env.unwrapped.max_episode_steps * num_episodes

        eps = 1.
        eps_diff = eps_coef / num_steps

        state, _ = env.reset(seed=seed)
        term, trunc = False, False
        # for i in trange(1, num_steps + 1):
        for i in range(1, num_steps + 1):
            if term or trunc:
                state, _ = env.reset()
                rewards_history.append(episode_reward)
                episode_reward = 0

            if random.random() < eps:
                a = env.action_space.sample()
            else:
                a = np.argmax(Q[state, :])

            next_state, r, term, trunc, _ = env.step(a)
            episode_reward += r
            if term:
                Q[next_state, :] = 0

            if savedir is not None:
                trajectories['states'].append(state)
                trajectories['actions'].append(a)
                trajectories['rewards'].append(r)
                trajectories['terminateds'].append(term)
                trajectories['truncateds'].append(trunc)

            Q[state, a] += lr * (r + discount * np.max(Q[next_state, :]) - Q[state, a])

            state = next_state
            eps = max(0, eps - eps_diff)

        if savedir is not None:
            dump_trajectories(savedir, trajectories, env.unwrapped.goal_pos)

        if return_history:
            return Q, rewards_history
        return Q, None


@pyrallis.wrap()
def main(config: Config):
    worker = MultiProcQLearningWorker(config)

    rewards_history = worker.generate_dataset()
    
    if rewards_history is not None:
        rewards_history = np.array(rewards_history)
        
        plt.figure()
        plt.plot(np.arange(1, rewards_history.shape[1] + 1), rewards_history.mean(axis=0))
        # plt.xlabel('Episode')
        # plt.ylabel('Average Reward')
        # plt.title('Learning Curve')
        plt.savefig('qlearning_curve.png')
        plt.close()


if __name__ == "__main__":
    main()