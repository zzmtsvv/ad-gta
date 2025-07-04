import os
import uuid
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import gymnasium as gym
import itertools
from collections import defaultdict
from gymnasium.vector import SyncVectorEnv
import numpy as np
import pyrallis
import torch
from torch.nn import functional as F  # noqa
from torch.utils.data import DataLoader

import wandb

import src.envs
from src.gta_ad import TedZadouriAD
from src.utils.data import TuplesIterableDataset
from src.utils.misc import set_seed
from src.utils.schedule import cosine_annealing_with_warmup
from src.utils.visualization import per_episode_in_context, attn_layers_scores, split_info_debug
from generate_q import TRAIN_GOALS


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EvalReturnsInfoType = dict[tuple[int, int], list[float]]
EvalSeqInfoType = list[defaultdict[str, list]]


@dataclass
class TrainConfig:
    # wandb params
    project: str = "attention-gta"
    group: str = "debug"
    name: str = "ad-darkroom"
    # model params
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 4
    num_key_value_heads: int = 2
    rope_theta: int = 10_000
    seq_len: int = 200
    attention_dropout: float = 0.5
    residual_dropout: float = 0.1
    embedding_dropout: float = 0.3
    normalize_qk: bool = False
    pre_norm: bool = True
    # training params
    env_name: str = "Dark-Room-9x9-v0"
    learning_rate: float = 3e-4
    warmup_ratio: float = 0.05
    betas: Tuple[float, float] = (0.9, 0.99)
    weight_decay: float = 0.0
    clip_grad: Optional[float] = 1.0
    subsample: int = 1
    batch_size: int = 128
    update_steps: int = 125_000
    num_workers: int = 0
    label_smoothing: float = 0.0
    # evaluation params
    eval_every: int = 25_000
    eval_episodes: int = 200
    eval_train_goals: int = 10
    eval_test_goals: int = 50
    # general params
    learning_histories_path: str = "trajectories"
    checkpoints_path: Optional[str] = None
    train_seed: int = None
    data_seed: int = 0
    eval_seed: int = 1

    def __post_init__(self):
        assert (self.hidden_dim / self.num_heads) % 8 == 0, "head dim should be multiple of 8 for flash attn"

        self.name = f"{self.name}-{str(uuid.uuid4())[:8]}"
        if self.checkpoints_path is not None:
            self.checkpoints_path = os.path.join(self.checkpoints_path, self.name)


@torch.no_grad()
def evaluate_in_context_with_cache(
    env_name: str,
    model: TedZadouriAD,
    goals: np.ndarray,
    eval_episodes: int,
    seed: int | None = None
) -> tuple[EvalReturnsInfoType, EvalSeqInfoType, EvalSeqInfoType]:
    vec_env = SyncVectorEnv([lambda goal=goal: gym.make(env_name, goal_pos=goal) for goal in goals])
    tmp_env = gym.make(env_name, goal_pos=goals[0])

    # stuff to know where to collect the data for attention maps visualization
    # we will visualize attn patterns at the start, middle and end of the evaluation,
    # respectively, just to sanity check whether there is a difference 
    # with reward signal across sequence being changed (in case the model has in-context capabilities)
    model_seq_len = model.seq_len
    total_steps = eval_episodes * tmp_env.max_episode_steps
    end_stage_start = total_steps - model_seq_len
    middle_stage_start = total_steps // 2 - model_seq_len // 2
    middle_stage_end = total_steps // 2 + model_seq_len // 2
    first_stage_end = model_seq_len

    # from 1 env each
    train_sequences = [defaultdict(list) for _ in range(3)]
    test_sequences = [defaultdict(list) for _ in range(3)]

    # kv_cache = model.init_cache(batch_size=vec_env.num_envs, dtype=torch.bfloat16, device=DEVICE)
    kv_cache = model.init_cache(batch_size=vec_env.num_envs, dtype=torch.float16, device=DEVICE)

    # to track number of episodes for each goal and returns
    num_episodes = np.zeros(vec_env.num_envs)
    returns = np.zeros(vec_env.num_envs)
    # for logging
    eval_info = defaultdict(list)

    state, _ = vec_env.reset(seed=seed)
    prev_action, prev_reward = np.zeros(vec_env.num_envs), np.zeros(vec_env.num_envs)
    for step in itertools.count(start=1):
        # predict next action
        with torch.autocast(device_type="cuda"):
            # [num_envs, seq_len=1, num_actions] -> [num_envs, num_actions]
            logits, kv_cache = model(
                states=torch.as_tensor(state, dtype=torch.long, device=DEVICE)[:, None],
                prev_actions=torch.as_tensor(prev_action, dtype=torch.long, device=DEVICE)[:, None],
                prev_rewards=torch.as_tensor(prev_reward, dtype=torch.float, device=DEVICE)[:, None],
                cache=kv_cache
            )
            logits = logits[:, -1]
        
        # TODO: collect appropriate sequences for attn patterns visualization
        if 1 <= step <= first_stage_end:
            train_sequences[0]["states"].append(state[0])
            train_sequences[0]["prev_actions"].append(prev_action[0])
            train_sequences[0]["prev_rewards"].append(prev_reward[0])

            test_sequences[0]["states"].append(state[-1])
            test_sequences[0]["prev_actions"].append(prev_action[-1])
            test_sequences[0]["prev_rewards"].append(prev_reward[-1])

        if middle_stage_start <= step <= middle_stage_end - 1:
            train_sequences[1]["states"].append(state[0])
            train_sequences[1]["prev_actions"].append(prev_action[0])
            train_sequences[1]["prev_rewards"].append(prev_reward[0])

            test_sequences[1]["states"].append(state[-1])
            test_sequences[1]["prev_actions"].append(prev_action[-1])
            test_sequences[1]["prev_rewards"].append(prev_reward[-1])

        if end_stage_start <= step <= total_steps - 1:
            train_sequences[2]["states"].append(state[0])
            train_sequences[2]["prev_actions"].append(prev_action[0])
            train_sequences[2]["prev_rewards"].append(prev_reward[0])

            test_sequences[2]["states"].append(state[-1])
            test_sequences[2]["prev_actions"].append(prev_action[-1])
            test_sequences[2]["prev_rewards"].append(prev_reward[-1])

        dist = torch.distributions.Categorical(logits=logits)
        # action = dist.sample()
        action = dist.mode

        # query the world
        state, reward, terminated, truncated, _ = vec_env.step(action.cpu().numpy())
        done = terminated | truncated

        # relabel for the next step
        prev_action = action
        prev_reward = reward

        num_episodes += done.astype(int)
        returns += reward

        # log returns if done
        for i, d in enumerate(done):
            if d and num_episodes[i] <= eval_episodes:
                eval_info[tuple(goals[i])].append(returns[i])
                # reset return for this goal
                returns[i] = 0.0
                # update tqdm
                # pbar.update(1)

        # check that all goals are done
        if np.all(num_episodes > eval_episodes):
            break

    vec_env.close()
    tmp_env.close()
    return eval_info, train_sequences, test_sequences


@torch.no_grad()
def get_attn_maps(
        model: TedZadouriAD,
        train_sequences: EvalSeqInfoType,
        test_sequences: EvalSeqInfoType
) -> list[str]:
    phases = [
        "start_eval",
        "mid_eval",
        "end_eval"
    ]
    images_paths = []

    for train_seq, test_seq, phase in zip(train_sequences, test_sequences, phases):
        
        with torch.autocast(device_type="cuda"):
            train_maps_per_layer, train_tokens_states = model.get_attention_maps(
                states=torch.as_tensor(train_seq["states"], dtype=torch.long, device=DEVICE)[None, :],
                prev_actions=torch.as_tensor(train_seq["prev_actions"], dtype=torch.long, device=DEVICE)[None, :],
                prev_rewards=torch.as_tensor(train_seq["prev_rewards"], dtype=torch.float, device=DEVICE)[None, :]
            )

            test_maps_per_layer, test_tokens_states = model.get_attention_maps(
                states=torch.as_tensor(test_seq["states"], dtype=torch.long, device=DEVICE)[None, :],
                prev_actions=torch.as_tensor(test_seq["prev_actions"], dtype=torch.long, device=DEVICE)[None, :],
                prev_rewards=torch.as_tensor(test_seq["prev_rewards"], dtype=torch.float, device=DEVICE)[None, :]
            )
        
        images_paths.append(
            attn_layers_scores(
            train_maps_per_layer.cpu().numpy(),
            train_tokens_states.cpu().numpy(),
            f"train_{phase}"
        ))

        images_paths.append(
            attn_layers_scores(
            test_maps_per_layer.cpu().numpy(),
            test_tokens_states.cpu().numpy(),
            f"test_{phase}"
        ))
    
    return images_paths


@pyrallis.wrap()
def train(config: TrainConfig):
    dict_config = asdict(config)
    wandb.init(
        project=config.project,
        group=config.group,
        name=config.name,
        config=dict_config,
        save_code=True
    )

    dataset = TuplesIterableDataset(
        data_path=config.learning_histories_path,
        seq_len=config.seq_len,
        subsample=config.subsample,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=config.num_workers,
    )
    tmp_env = gym.make(config.env_name)

    mean_optimal_return = 96.7421875
    y_lim = [-0.3, 100]
    
    # workaround to keep consistent starting goal positions
    grid_size = tmp_env.unwrapped.size
    all_goals = np.mgrid[0:grid_size, 0:grid_size].reshape(2, -1).T
    
    test_goals = np.array([goal for goal in all_goals if tuple(goal) not in set(map(tuple, TRAIN_GOALS))])

    train_goals = TRAIN_GOALS[:config.eval_train_goals]

    eval_all_goals = np.vstack([train_goals, test_goals])

    # model & optimizer & scheduler setup
    print("Train seed:", config.train_seed)
    set_seed(config.train_seed)

    model = TedZadouriAD(
        num_states=tmp_env.observation_space.n,
        num_actions=tmp_env.action_space.n,
        hidden_dim=config.hidden_dim,
        seq_len=config.seq_len,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        num_key_value_heads=config.num_key_value_heads,
        rope_theta=config.rope_theta,
        attention_dropout=config.attention_dropout,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        normalize_qk=config.normalize_qk,
        pre_norm=config.pre_norm
    ).to(DEVICE)

    # if needed, test beforehand
    # model = torch.compile(model)

    optim = torch.optim.Adam(
        params=model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )

    scheduler = cosine_annealing_with_warmup(
        optimizer=optim,
        warmup_steps=int(config.update_steps * config.warmup_ratio),
        total_steps=config.update_steps,
    )

    # save config to the checkpoint
    if config.checkpoints_path is not None:
        print(f"Checkpoints path: {config.checkpoints_path}")
        os.makedirs(config.checkpoints_path, exist_ok=True)
        with open(os.path.join(config.checkpoints_path, "config.yaml"), "w") as f:
            pyrallis.dump(config, f)

    scaler = torch.GradScaler()
    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")
    for global_step, batch in enumerate(dataloader):
        if global_step > config.update_steps:
            break

        states, prev_actions, prev_rewards, target_actions = [b.to(DEVICE) for b in batch]

        states = states.to(torch.long)
        prev_actions = prev_actions.to(torch.long)
        prev_rewards = prev_rewards.to(torch.float)
        target_actions = target_actions.to(torch.long)

        with torch.autocast(device_type="cuda"):
            predicted_actions, _ = model(
                states=states, prev_actions=prev_actions, prev_rewards=prev_rewards
            )
            loss = F.cross_entropy(
                input=predicted_actions.flatten(0, 1),
                target=target_actions.flatten(0, 1),
                label_smoothing=config.label_smoothing,
            )

        scaler.scale(loss).backward()
        if config.clip_grad is not None:
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
        scaler.step(optim)
        scaler.update()
        optim.zero_grad(set_to_none=True)
        scheduler.step()

        with torch.no_grad():
            a = torch.argmax(predicted_actions.flatten(0, 1), dim=-1)
            t = target_actions.flatten()
            accuracy = torch.sum(a == t) / a.shape[0]

            wandb.log({
                "loss": loss.item(),
                "accuracy": accuracy,
                "lr": scheduler.get_last_lr()[0],
            }, step=global_step,
            )

            if global_step % config.eval_every == 0:
                model.eval()

                cache_eval_info, train_sequences, test_sequences = evaluate_in_context_with_cache(
                        env_name=config.env_name,
                        model=model,
                        goals=eval_all_goals,
                        eval_episodes=config.eval_episodes,
                        seed=config.eval_seed,
                    )
                cache_eval_info_train, cache_eval_info_test = split_info_debug(cache_eval_info, train_goals, test_goals)

                pic_name_train = per_episode_in_context(
                    eval_res=cache_eval_info_train,
                    ylim=y_lim,
                    name=f"train-viz",
                    mean_optimal_return=mean_optimal_return,
                )

                pic_name_test = per_episode_in_context(
                    eval_res=cache_eval_info_test,
                    ylim=y_lim,
                    name=f"test-viz",
                    mean_optimal_return=mean_optimal_return,
                )
                wandb.log({
                    "eval/train_graph": wandb.Image(pic_name_train),
                    "eval/test_graph": wandb.Image(pic_name_test),
                    "eval/train_mean_return": np.mean([h[-1] for h in cache_eval_info_train.values()]),
                    "eval/train_median_return": np.median([h[-1] for h in cache_eval_info_train.values()]),
                    "eval/test_mean_return": np.mean([h[-1] for h in cache_eval_info_test.values()]),
                    "eval/test_median_return": np.median([h[-1] for h in cache_eval_info_test.values()]),
                }, step=global_step
                )

                attn_imgs = get_attn_maps(model, train_sequences, test_sequences)
                wandb.log({
                    f"attn_maps/{img[:-4]}": wandb.Image(img) for img in attn_imgs
                }, step=global_step)

                if config.checkpoints_path is not None:
                    torch.save(
                        model.state_dict(),
                        os.path.join(config.checkpoints_path, f"model_{global_step}.pt"),
                    )
                model.train()

    if config.checkpoints_path is not None:
        torch.save(
            model.state_dict(), os.path.join(config.checkpoints_path, f"model_last.pt")
        )


if __name__ == "__main__":
    train()