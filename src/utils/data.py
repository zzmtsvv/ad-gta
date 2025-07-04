import random
import numpy as np

import glob
from collections import defaultdict
from torch.utils.data import IterableDataset
from typing import List, Dict, Any


def load_learning_histories(path: str) -> List[Dict[str, Any]]:
    files = glob.glob(f"{path}/*.npz")

    learning_histories = []
    for filename in files:
        with np.load(filename, allow_pickle=True) as f:
            learning_histories.append({
                "states": f["states"],
                "actions": f["actions"],
                "rewards": f["rewards"],
                "dones": f["dones"],
                "goal": f["goal"],
            })

    return learning_histories


def split_to_episodes(learning_history):
    trajectories = []

    traj_data = defaultdict(list)
    for step in range(len(learning_history["dones"])):
        # append data
        traj_data["states"].append(learning_history["states"][step])
        traj_data["actions"].append(learning_history["actions"][step])
        traj_data["rewards"].append(learning_history["rewards"][step])

        if learning_history["dones"][step]:
            trajectories.append({k: np.array(v) for k, v in traj_data.items()})
            traj_data = defaultdict(list)

    return trajectories


def subsample_history(learning_history, subsample):
    trajectories = split_to_episodes(learning_history)

    subsampled_trajectories = trajectories[::subsample]
    subsampled_history = {
        "states": np.concatenate([traj["states"] for traj in subsampled_trajectories]),
        "actions": np.concatenate([traj["actions"] for traj in subsampled_trajectories]),
        "rewards": np.concatenate([traj["rewards"] for traj in subsampled_trajectories]),
    }
    return subsampled_history


class TuplesIterableDataset(IterableDataset):
    def __init__(
            self,
            data_path: str,
            seq_len: int = 60,
            subsample: int = 1
        ):
        self.seq_len = seq_len
        print("Loading training histories...")
        self.histories = load_learning_histories(data_path)
        print("Num histories:", len(self.histories))

        self.goals = np.vstack([trajectory["goal"] for trajectory in self.histories])
        self.unique_goals = np.unique(self.goals, axis=0)

        if subsample > 1:
            self.histories = [subsample_history(hist, subsample) for hist in self.histories]

    def __prepare_sample(self, history_idx, start_idx):
        history = self.histories[history_idx]
        assert history["states"].shape[0] == history["actions"].shape[0] == history["rewards"].shape[0]

        # sampling state, prev_actions, prev_rewards
        states = history["states"][start_idx:start_idx + self.seq_len].flatten()
        prev_actions = history["actions"][start_idx - 1:start_idx - 1 + self.seq_len].flatten()
        prev_rewards = history["rewards"][start_idx - 1:start_idx - 1 + self.seq_len].flatten()
        # target actions to predict from the context
        target_actions = history["actions"][start_idx:start_idx + self.seq_len].flatten()

        assert states.shape[0] == prev_actions.shape[0] == prev_rewards.shape[0] == self.seq_len
        assert target_actions.shape[0] == self.seq_len

        return states, prev_actions, prev_rewards, target_actions

    def __iter__(self):
        while True:
            history_idx = random.randint(0, len(self.histories) - 1)
            # sample in a way to avoid paddings from both sides
            start_idx = random.randint(1, self.histories[history_idx]["rewards"].shape[0] - self.seq_len - 1)
            yield self.__prepare_sample(history_idx, start_idx)
