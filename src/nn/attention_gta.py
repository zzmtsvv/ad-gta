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
# from src.nn.triton_rope_fn import apply_rotary_emb


class TiedKVCache:
    # https://arxiv.org/abs/2505.21487v1
    def __init__(
            self,
            batch_size: int,
            max_seq_len: int,
            num_layers: int,
            num_heads: int,
            head_dim: int,
            rope_dim: int,
            device: torch.device,
            dtype: torch.dtype,
    ):
        self.kv_shape = (batch_size, max_seq_len, num_heads, head_dim)
        self.k_rope_shape = (batch_size, max_seq_len, 1, rope_dim)
        
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.device = device
        self.dtype = dtype

        self.kv_states: list[torch.Tensor] = [
            torch.full(self.kv_shape, fill_value=torch.nan, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.key_rope: list[torch.Tensor] = [
            torch.full(self.k_rope_shape, fill_value=torch.nan, dtype=self.dtype, device=self.device)
            for _ in range(self.num_layers)
        ]
        self.cache_seqlens: list[int] = [0] * num_layers
    
    def __len__(self) -> int:
        return len(self.kv_states)

    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv_states[layer_idx], self.key_rope[layer_idx]

    def update(
            self,
            new_kv: torch.Tensor,
            new_k_rope: torch.Tensor,
            layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_seq_len = self.cache_seqlens[layer_idx]

        if current_seq_len < self.max_seq_len:
            self.cache_seqlens[layer_idx] += 1
            insert_pos = self.cache_seqlens[layer_idx] - 1

            self.kv_states[layer_idx][:, insert_pos] = new_kv.squeeze(1)
            self.key_rope[layer_idx][:, insert_pos] = new_k_rope.squeeze(1)

            kv_out = self.kv_states[layer_idx][:, :self.cache_seqlens[layer_idx]]
            k_rope_out = self.key_rope[layer_idx][:, :self.cache_seqlens[layer_idx]]
        else:

            self.kv_states[layer_idx] = torch.roll(self.kv_states[layer_idx], shifts=-1, dims=1)
            self.key_rope[layer_idx] = torch.roll(self.key_rope[layer_idx], shifts=-1, dims=1)

            self.kv_states[layer_idx][:, -1] = new_kv.squeeze(1)
            self.key_rope[layer_idx][:, -1] = new_k_rope.squeeze(1)
            kv_out = self.kv_states[layer_idx]
            k_rope_out = self.key_rope[layer_idx]

        return kv_out, k_rope_out


class FlashGTA(nn.Module):
    # https://arxiv.org/abs/2505.21487v1
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
        self.kv_proj = nn.Linear(self.hidden_dim, self.num_key_value_heads * self.head_dim)
        
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_dim)

        self.rope_dim = self.head_dim // 2
        
        self.W_rope_k = nn.Linear(self.hidden_dim, self.rope_dim) # (D, d//2)
    
    def forward(
            self,
            x: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            kv_cache: TiedKVCache | None = None
    ) -> tuple[torch.Tensor, TiedKVCache | None]:
        B, L, D = x.size()

        q = self.q_proj(x)
        q = q.view(B, L, self.num_heads, self.head_dim)

        kv_states = self.kv_proj(x).view(B, L, self.num_key_value_heads, self.head_dim) 
        query, query_rope = torch.split(q, [q.size(-1) - self.rope_dim, self.rope_dim], dim=-1)
        
        key_rope = self.W_rope_k(x)
        key_rope = key_rope.view(B, L, 1, self.rope_dim) 

        cos, sin = position_embeddings

        query_rope, key_rope = apply_rotary_emb(query_rope, key_rope, cos, sin, unsqueeze_dim=2)
        
        if kv_cache is not None:
            kv_states, key_rope = kv_cache.update(kv_states, key_rope, self.layer_idx)

        # key_rope = repeat(key_rope, 'b l 1 d -> b l h d', h=self.num_heads)
        # key_rope = key_rope.expand(-1, -1, self.num_heads, -1)
        key_rope = key_rope.expand(-1, -1, self.num_key_value_heads, -1)
        
        query_states = torch.cat([query, query_rope], dim=-1)

        kv_states_tied, value_states = torch.split(kv_states, [kv_states.size(-1) - self.rope_dim, self.rope_dim], dim=-1)  

        key_states = torch.cat([kv_states_tied, key_rope], dim=-1) 
        value_states = torch.cat([kv_states_tied, value_states], dim=-1)

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
        q = q.view(B, L, self.num_heads, self.head_dim)

        kv_states = self.kv_proj(x).view(B, L, self.num_key_value_heads, self.head_dim) 
        query, query_rope = torch.split(q, [q.size(-1) - self.rope_dim, self.rope_dim], dim=-1)
        
        key_rope = self.W_rope_k(x)
        key_rope = key_rope.view(B, L, 1, self.rope_dim) 
        
        cos, sin = position_embeddings

        query_rope, key_rope = apply_rotary_emb(query_rope, key_rope, cos, sin, unsqueeze_dim=2)
        query_states = torch.cat([query, query_rope], dim=-1)
        # key_rope = repeat(key_rope, 'b l 1 d -> b l h d', h=self.num_heads)
        key_rope = key_rope.expand(-1, -1, self.num_heads, -1)

        kv_states_tied, value_states = torch.split(kv_states, [kv_states.size(-1) - self.rope_dim, self.rope_dim], dim=-1)  
        
        kv_states_tied = torch.repeat_interleave(kv_states_tied, self.num_key_value_groups, dim=2) 
        # value_states = torch.repeat_interleave(value_states, self.num_key_value_groups, dim=2) 
        
        key_states = torch.cat([kv_states_tied, key_rope], dim=-1) 
        # value_states = torch.cat([kv_states_tied, value_states], dim=-1)

        if self.normalize_qk:
            query_states = self.q_norm(query_states)
            key_states = self.k_norm(key_states)

        # attn
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

        self.attention = FlashGTA(
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
            kv_cache: TiedKVCache | None = None
    ) -> tuple[torch.Tensor, TiedKVCache | None]:
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
