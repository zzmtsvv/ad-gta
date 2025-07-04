import torch
from torch import nn
from torch.nn import functional as F

from src.nn.attention import TransformerBlock, KVCache
from src.nn.memeff_rope_fn import RotaryEmbedding


class Transformer(nn.Module):
    def __init__(
            self,
            seq_len: int = 40,
            embedding_dim: int = 64,
            hidden_dim: int = 256,
            num_layers: int = 4,
            num_heads: int = 4,
            num_key_value_heads: int = 2,
            rope_theta: int = 10_000,
            attention_dropout: float = 0.5,
            residual_dropout: float = 0.0,
            embedding_dropout: float = 0.1,
            pre_norm: bool = True,
            normalize_qk: bool = False,
    ):
        super().__init__()
        self.emb_drop = nn.Dropout(embedding_dropout)
        self.emb2hid = nn.Linear(embedding_dim, hidden_dim)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    num_key_value_heads=num_key_value_heads,
                    attention_dropout=attention_dropout,
                    residual_dropout=residual_dropout,
                    layer_idx=i,
                    pre_norm=pre_norm,
                    normalize_qk=normalize_qk
                )
                for i in range(num_layers)
            ]
        )
        self.seq_len = seq_len
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.rope_theta = rope_theta

        self.rotary_emb = RotaryEmbedding(
            hidden_dim // num_heads,
            seq_len,
            rope_theta
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        # taken from the nanoGPT, may be not optimal
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def init_cache(
            self,
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device
        ) -> KVCache:
        cache = KVCache(
            batch_size=batch_size,
            max_seq_len=self.seq_len,
            num_layers=self.num_layers,
            num_heads=self.num_key_value_heads,
            head_dim=self.hidden_dim // self.num_heads,
            device=device,
            dtype=dtype,
        )
        return cache
    
    def forward(
            self,
            sequence: torch.Tensor,
            cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache | None]:
        # [batch_size, seq_len, hidden_dim]
        sequence = self.emb2hid(sequence)
        position_embeddings = self.rotary_emb(sequence)

        out = self.emb_drop(sequence)
        for i, block in enumerate(self.blocks):
            out, cache = block(out, position_embeddings, cache)

        return out, cache

    def get_attention_maps(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # utils for visualization
        maps_per_layer = []
        tokens_states = []

        sequence = self.emb2hid(sequence)
        position_embeddings = self.rotary_emb(sequence)
        x = self.emb_drop(sequence)

        # in fact that is not efficient because there are 2 forward passes for the same input
        # but we're doing so in case to prevent error accumulation when replacing flash attention 
        # with manual one (from the floating point arithmetics
        # these are different operations)
        for block in self.blocks:
            maps_per_layer.append(
                block.get_attention_maps(x, position_embeddings)
            )
            x, _ = block(x, position_embeddings)
            tokens_states.append(x)
        
        # maps_per_layer: [num_layers, batch_size, num_heads, seq_len, head_dim]
        # tokens_states:  [num_layers, batch_size, hidden_dim] 
        return torch.stack(maps_per_layer), torch.stack(tokens_states)


class AD(nn.Module):
    def __init__(
            self,
            num_states: int,
            num_actions: int,
            seq_len: int = 200,
            hidden_dim: int = 256,
            num_layers: int = 4,
            num_heads: int = 4,
            attention_dropout: float = 0.5,
            residual_dropout: float = 0.0,
            embedding_dropout: float = 0.1,
            normalize_qk: bool = False,
            pre_norm: bool = True,
    ):
        super().__init__()
        self.transformer = Transformer(
            seq_len=seq_len,
            embedding_dim=num_actions + num_states + 1,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
            residual_dropout=residual_dropout,
            embedding_dropout=embedding_dropout,
            normalize_qk=normalize_qk,
            pre_norm=pre_norm,
        )
        self.action_head = nn.Linear(hidden_dim, num_actions)

        self.num_states = num_states
        self.num_actions = num_actions
        self.seq_len = seq_len

    def init_cache(
            self,
            batch_size: int,
            dtype: torch.dtype,
            device: torch.device
        ) -> KVCache:
        return self.transformer.init_cache(batch_size, dtype, device)

    def forward(
            self,
            states: torch.Tensor,        # [batch_size, seq_len]
            prev_actions: torch.Tensor,  # [batch_size, seq_len]
            prev_rewards: torch.Tensor,  # [batch_size, seq_len]
            cache: KVCache = None,
    ) -> tuple[torch.Tensor, KVCache]:
        # you can use different encodings here, but I use the most simple one,
        # which works just fine for the dark-room, key-to-door
        state_emb = F.one_hot(states, num_classes=self.num_states)
        action_emb = F.one_hot(prev_actions, num_classes=self.num_actions)
        reward_emb = prev_rewards.unsqueeze(-1)

        # [batch_size, seq_len, emb_dim * 3]
        sequence = torch.concatenate([action_emb, reward_emb, state_emb], dim=-1)
        out, cache = self.transformer(sequence, cache=cache)
        # [batch_size, seq_len, num_actions]
        out = self.action_head(out)
        return out, cache

    def get_attention_maps(
            self,
            states: torch.Tensor,        # [batch_size, seq_len]
            prev_actions: torch.Tensor,  # [batch_size, seq_len]
            prev_rewards: torch.Tensor,  # [batch_size, seq_len]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_emb = F.one_hot(states, num_classes=self.num_states)
        action_emb = F.one_hot(prev_actions, num_classes=self.num_actions)
        reward_emb = prev_rewards.unsqueeze(-1)

        # [batch_size, seq_len, emb_dim * 3]
        sequence = torch.concatenate([action_emb, reward_emb, state_emb], dim=-1)
        return self.transformer.get_attention_maps(sequence)
