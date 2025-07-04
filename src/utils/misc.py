import itertools
import os
import torch
import random
import numpy as np


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def all_goals_k2d(grid_size):
    goals = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
    goals = list(itertools.product(goals, goals))
    goals = np.array([g for g in goals if not np.array_equal(g[0], g[1])])
    return goals


def train_test_goals_k2d(grid_size, num_train_goals, num_test_goals=None, seed=42):
    set_seed(seed)
    assert num_train_goals <= grid_size**4

    goals = all_goals_k2d(grid_size)
    
    goals = np.random.permutation(goals)

    train_goals = goals[:num_train_goals]
    test_goals = goals[num_train_goals:]

    if num_test_goals is not None:
        test_goals = test_goals[:num_test_goals]

    return train_goals, test_goals