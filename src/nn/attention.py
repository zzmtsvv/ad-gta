import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import repeat

try:
    from flash_attn import flash_attn_func
except ImportError:
    warnings.warn("Missing FlashAttention Install", category=Warning)

from src.nn.memeff_rope_fn import apply_rotary_emb


class KVCache:
    def __init__(
            self,
            batch_size: int,
            max_seq_len: int,
            num_layers: int,
            num_heads: int,
            head_dim: int,
            device: torch.device,
            dtype: torch.dtype,
    ):
        self.kv_shape = (batch_size, max_seq_len, num_heads, head_dim)
        
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        self.key_states: list[torch.Tensor] = [
            torch.full(self.kv_shape, fill_value=torch.nan, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.value_states: list[torch.Tensor] = [
            torch.full(self.kv_shape, fill_value=torch.nan, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.cache_seqlens: list[int] = [0] * num_layers
    
    def __len__(self) -> int:
        return len(self.key_states)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key_states[layer_idx], self.value_states[layer_idx]

    def update(
            self,
            new_k: torch.Tensor,
            new_v: torch.Tensor,
            layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_seq_len = self.cache_seqlens[layer_idx]

        if current_seq_len < self.max_seq_len:
            self.cache_seqlens[layer_idx] += 1
            insert_pos = self.cache_seqlens[layer_idx] - 1

            self.key_states[layer_idx][:, insert_pos] = new_k.squeeze(1)
            self.value_states[layer_idx][:, insert_pos] = new_v.squeeze(1)

            key_out = self.key_states[layer_idx][:, :self.cache_seqlens[layer_idx]]
            value_out = self.value_states[layer_idx][:, :self.cache_seqlens[layer_idx]]
        else:

            self.key_states[layer_idx] = torch.roll(self.key_states[layer_idx], shifts=-1, dims=1)
            self.value_states[layer_idx] = torch.roll(self.value_states[layer_idx], shifts=-1, dims=1)

            self.key_states[layer_idx][:, -1] = new_k.squeeze(1)
            self.value_states[layer_idx][:, -1] = new_v.squeeze(1)
            key_out = self.key_states[layer_idx]
            value_out = self.value_states[layer_idx]

        return key_out, value_out


class FlashRoPEAttention(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_key_value_heads: int,
            layer_idx: int,
            normalize_qk: bool = False,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads # for q
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads

        self.layer_idx = layer_idx
        self.normalize_qk = normalize_qk

        if normalize_qk:
            self.q_norm = nn.LayerNorm(hidden_dim // num_heads)
            self.k_norm = nn.LayerNorm(hidden_dim // num_heads)

        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim)
        self.kv_proj = nn.Linear(self.hidden_dim, self.num_key_value_heads * self.head_dim * 2)
        
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim)

    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            kv_cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache | None]:
        B, L, D = x.size()

        q = self.q_proj(x)
        query_states = q.view(B, L, self.num_heads, self.head_dim)

        kv_states = self.kv_proj(x).view(B, L, 2, self.num_key_value_heads, self.head_dim)
        key_states, value_states = kv_states.unbind(2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
        
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        if self.normalize_qk:
            query_states = self.q_norm(query_states).to(value_states.dtype)
            key_states = self.k_norm(key_states).to(value_states.dtype)
        
        attn_output = flash_attn_func(
            q=query_states, k=key_states, v=value_states, causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        attn_output = attn_output.reshape(B, L, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, kv_cache

    def get_attention_maps(
            self,
            x: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        assert not self.training

        B, L, D = x.size()

        q = self.q_proj(x)
        query_states = q.view(B, L, self.num_heads, self.head_dim)

        kv_states = self.kv_proj(x).view(B, L, 2, self.num_key_value_heads, self.head_dim)
        key_states, value_states = kv_states.unbind(2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)

        if self.normalize_qk:
            query_states = self.q_norm(query_states).to(value_states.dtype)
            key_states = self.k_norm(key_states).to(value_states.dtype)
        
        key_states = torch.repeat_interleave(key_states, self.num_key_value_groups, dim=2) 
        # value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=2) 

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        attn = (query_states @ key_states.transpose(-2, -1)) * (1 / math.sqrt(key_states.size(-1)))
        
        causal_mask = torch.tril(torch.ones(L, L)).to(x.device)
        attn = attn.masked_fill_(causal_mask == 0, torch.finfo(attn.dtype).min)
        
        # [B, nH, L, L]
        attn = F.softmax(attn, dim=-1)
        return attn


class TransformerBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            num_key_value_heads: int,
            attention_dropout: float,
            residual_dropout: float,
            layer_idx: int,
            pre_norm: bool = True,
            normalize_qk: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = FlashRoPEAttention(
            hidden_dim,
            num_heads,
            num_key_value_heads,
            layer_idx,
            normalize_qk,
            attention_dropout
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(residual_dropout),
        )
        self.pre_norm = pre_norm
    
    # [batch_size, seq_len, emb_dim] -> [batch_size, seq_len, emb_dim]
    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            kv_cache: KVCache | None = None
    ) -> tuple[torch.Tensor, KVCache | None]:
        if self.pre_norm:
            attention_out, kv_cache = self.attention(self.norm1(x), position_embeddings, kv_cache=kv_cache)
            x = x + self.drop(attention_out)
            x = x + self.mlp(self.norm2(x))
        else:
            attention_out, kv_cache = self.attention(x, position_embeddings, kv_cache=kv_cache)
            x = self.norm1(x + self.drop(attention_out))
            x = self.norm2(x + self.mlp(x))

        return x, kv_cache

    def get_attention_maps(
            self,
            x: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.attention.get_attention_maps(x, position_embeddings)
