import logging
from typing import Any
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import gridspec

logging.getLogger("matplotlib").setLevel(logging.ERROR)


EvalReturnsInfoType = dict[tuple[int, int], list[float]]
EvalSeqInfoType = list[defaultdict[str, list]]


def per_episode_in_context(
        eval_res: dict[Any, list[float]],
        name: str,
        ylim: tuple[float, float] | None = None,
        optimal_return: float | None = None
    ) -> str:
    rets = np.vstack([h for h in eval_res.values()])
    means = rets.mean(0)
    stds = rets.std(0)
    x = np.arange(1, rets.shape[1] + 1)

    fig, ax = plt.subplots(dpi=100)
    ax.grid(visible=True)
    
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    
    ax.plot(x, means)
    ax.fill_between(x, means - stds, means + stds, alpha=0.2)

    ax.set_ylabel("Return")
    ax.set_xlabel("Episodes In-Context")
    ax.set_title(f"{name}")

    if optimal_return is not None:
        ax.axhline(
            optimal_return,
            ls="--",
            color="goldenrod",
            lw=2,
            label=f"mean_optimal_return: {optimal_return:.2f}",
        )
    if optimal_return is not None:
        plt.legend()

    fig.savefig(f"rets_vs_eps_{name}.png")
    plt.close()

    return f"rets_vs_eps_{name}.png"


def attn_layers_scores(
        attn_maps_per_layer: np.ndarray,
        tokens_states_per_layer: np.ndarray,
        name: str,
        batch_idx: int = 0,
        dpi: int = 240,
        cmap: str = 'plasma',
        figsize_scale: float = 3.0,
        aggregation_fn: callable = np.mean,
    ) -> str:
    '''
        attn_maps_per_layer: [num_layers, batch_size, num_heads, seq_len, head_dim]
        tokens_states_per_layer: [num_layers, batch_size, seq_len, hidden_dim]
        aggregation_fn to aggregate hidden_dim in token states
    '''
    batch_size = attn_maps_per_layer.shape[1]
    
    attn_maps = attn_maps_per_layer[:, batch_idx, ...]  # [num_layers, num_heads, seq_len, head_dim]
    tokens_states = tokens_states_per_layer[:, batch_idx, ...]  # [num_layers, seq_len, hidden_dim]
    
    num_layers, num_heads, seq_len, head_dim = attn_maps.shape
    _, seq_len_tokens, hidden_dim = tokens_states.shape
    
    agg_tokens = aggregation_fn(tokens_states, axis=2)  # [num_layers, seq_len]

    fig = plt.figure(
        figsize=(
            num_heads * figsize_scale, 
            (num_layers + 1) * figsize_scale
        ), 
        dpi=dpi
    )
    gs = gridspec.GridSpec(
        num_layers + 1,
        num_heads, 
        figure=fig, 
        height_ratios=[1.5] * num_layers + [1]
    )
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            ax = fig.add_subplot(gs[layer_idx, head_idx])
            attn_data = attn_maps[layer_idx, head_idx]
            im = ax.imshow(attn_data, cmap=cmap, aspect='auto')
            ax.set_title(f"Layer {layer_idx}\nHead {head_idx}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)
    
    ax_token = fig.add_subplot(gs[num_layers, :])
    im = ax_token.imshow(agg_tokens, cmap=cmap, aspect='auto', interpolation='nearest')
    ax_token.set_xlabel('Sequence Position', fontsize=9)
    ax_token.set_ylabel('Layer Index', fontsize=9)
    ax_token.set_yticks(np.arange(num_layers))
    ax_token.set_yticklabels(np.arange(num_layers))
    ax_token.tick_params(axis='both', labelsize=8)
    
    divider = make_axes_locatable(ax_token)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    
    plt.tight_layout()
    plt.savefig(f"{name}.png", bbox_inches='tight')
    plt.close()

    return f"{name}.png"


def split_info_debug(
        eval_info: EvalReturnsInfoType,
        train_goals: np.ndarray,
        test_goals: np.ndarray
) -> tuple[EvalReturnsInfoType, EvalReturnsInfoType]:
    eval_info_train = defaultdict(list)
    eval_info_test = defaultdict(list)

    train_goals = train_goals.tolist()

    for i, (k, v) in enumerate(eval_info.items()):
        if list(k) in train_goals:
            eval_info_train[k] = v
        
        elif list(k) in test_goals:
            eval_info_test[k] = v

    return eval_info_train, eval_info_test
