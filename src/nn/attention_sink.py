import math
import warnings
import torch
from torch import nn
from torch.nn import functional as F

from src.nn.attention import FlashAliBiCausalSelfAttention
try:
    import flash_attn
except ImportError:
    warnings.warn("Missing FlashAttention Install", category=Warning)


def get_alibi_slopes(n: int) -> list[float]:
    def get_slopes_power_of_2(n):
        start = (2**(-2**-(math.log2(n)-3)))
        ratio = start
        return [start*ratio**i for i in range(n)]

    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2**math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + get_alibi_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]


def get_alibi_relative_positions(seq_len: int) -> torch.Tensor:
    x = torch.arange(seq_len)[None, :]
    y = torch.arange(seq_len)[:, None]
    return (x - y).to(torch.float)


class FlashAliBiCausalSelfAttention(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            normalize_qk: bool = False,
            with_alibi: bool = True
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        if with_alibi:
            self.register_buffer(
                "alibi_slopes", torch.as_tensor(get_alibi_slopes(num_heads)), persistent=False
            )
        else:
            self.alibi_slopes = None
        
        if normalize_qk:
            self.q_norm = nn.LayerNorm(hidden_dim // num_heads)
            self.k_norm = nn.LayerNorm(hidden_dim // num_heads)

        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.normalize_qk = normalize_qk

    def forward(
            self,
            x: torch.Tensor,
            k_cache: torch.Tensor | None = None,
            v_cache: torch.Tensor | None = None,
            cache_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, L, D = x.size()
        # (batch_size, seq_len, 3, num_heads, head_dim)
        qkv = self.in_proj(x).reshape(B, L, 3, self.num_heads, D // self.num_heads)

        # normalizing q,k, see: https://arxiv.org/abs/2302.05442
        if self.normalize_qk:
            q, k, v = qkv.unbind(2)
            q_norm, k_norm = self.q_norm(q), self.k_norm(k)
            qkv = torch.stack([q_norm, k_norm, v], dim=2).to(qkv.dtype)

        # (batch_size, seq_len, num_heads, head_dim)
        if k_cache is None or v_cache is None or cache_seqlens is None:
            out = flash_attn.flash_attn_qkvpacked_func(
                qkv=qkv,
                dropout_p=self.dropout if self.training else 0.0,
                causal=True,
                alibi_slopes=self.alibi_slopes,
            )
        else:
            assert not self.training
            q, k, v = qkv.unbind(2)
            out = flash_attn.flash_attn_with_kvcache(
                q=q,
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                causal=True,
                alibi_slopes=self.alibi_slopes,
            )
        # (batch_size, seq_len, hidden_dim)
        out = self.out_proj(out.reshape(B, L, D))
        return out
    
    def get_alibi_mask(
            self,
            seq_len: int,
            device: torch.device
        ) -> torch.Tensor:
        # causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len)).to(device)
        # creating alibi attention bias matrix
        alibi_bias = get_alibi_relative_positions(seq_len).view(1, 1, seq_len, seq_len).to(device)

        alibi_bias = self.alibi_slopes.view(1, self.num_heads, 1, 1) * alibi_bias
        alibi_bias = alibi_bias.masked_fill(causal_mask == 0, float("-inf"))
        return alibi_bias

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.training

        B, L, D = x.size()
        # [batch_size, seq_len, 3, num_heads, head_dim] for FA2
        qkv = self.in_proj(x).reshape(B, L, 3, self.num_heads, D // self.num_heads).transpose(1, 3)
        # [batch_size, num_heads, seq_len, head_dim], num_heads <-> seq_len for torch SDPA/manual self attention implementation
        q, k, v = qkv.unbind(2)
        if self.normalize_qk:
            q, k = self.q_norm(q), self.k_norm(k)
        # attn
        attn = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        
        attn = attn + self.get_alibi_mask(L, x.device).to(x.device)
        
        # [B, nH, L, L]
        attn = F.softmax(attn, dim=-1)
        return attn


class SinkKVSelfAttention(FlashAliBiCausalSelfAttention):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            normalize_qk: bool = False,
            with_alibi: bool = True,
            num_sink_tokens: int = 1
        ) -> None:
        super().__init__(hidden_dim, num_heads, dropout, normalize_qk, with_alibi)

        # so it does not take gpu memory
        self.in_proj = None

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        self.num_sink_tokens = num_sink_tokens
        
        self.sink_k = torch.nn.Parameter(
                data=torch.zeros(num_sink_tokens, self.num_heads, hidden_dim // num_heads),
                requires_grad=True,
        )
        self.sink_v = torch.nn.Parameter(
                data=torch.zeros(num_sink_tokens, self.num_heads, hidden_dim // num_heads),
                requires_grad=True,
        )

    def forward(
            self,
            x: torch.Tensor,
            k_cache: torch.Tensor | None = None,
            v_cache: torch.Tensor | None = None,
            cache_seqlens: torch.Tensor | None = None) -> torch.Tensor:
        B, L, D = x.size()
        # (batch_size, seq_len, 3, num_heads, head_dim)
        # qkv = self.in_proj(x).reshape(B, L, 3, self.num_heads, D // self.num_heads)
        # q, k, v = qkv.unbind(2)
        q = self.q_proj(x).reshape(B, L, self.num_heads, D // self.num_heads)
        k = self.k_proj(x).reshape(B, L, self.num_heads, D // self.num_heads)
        v = self.v_proj(x).reshape(B, L, self.num_heads, D // self.num_heads)

        # normalizing q,k, see: https://arxiv.org/abs/2302.05442
        if self.normalize_qk:
            # q, k, v = qkv.unbind(2)
            # q_norm, k_norm = self.q_norm(q), self.k_norm(k)
            # qkv = torch.stack([q_norm, k_norm, v], dim=2).to(qkv.dtype)
            q, k = self.q_norm(q), self.k_norm(k)

        if k_cache is None or v_cache is None or cache_seqlens is None:
            sink_k = self.sink_k.to(q.dtype).expand(
                B, -1, -1, -1
            )
            sink_v = self.sink_v.to(q.dtype).expand(
                B, -1, -1, -1
            )

            # q, k, v = qkv.unbind(2)
            # q[:, :self.num_sink_tokens, :, :] = 0
            # k[:, :self.num_sink_tokens, : :] = sink_k
            # v[:, :self.num_sink_tokens, : :] = sink_v
            q[:, :self.num_sink_tokens] = 0
            k[:, :self.num_sink_tokens] = sink_k
            v[:, :self.num_sink_tokens] = sink_v

            out = flash_attn.flash_attn_func(
                q=q,
                k=k,
                v=v,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                dropout_p=self.dropout if self.training else 0.0
            )
        
        else:
            # the sinks are already in the k_cache and v_cache
            assert not self.training
            # q, k, v = qkv.unbind(2)
            out = flash_attn.flash_attn_with_kvcache(
                q=q,
                k=k,
                v=v,
                k_cache=k_cache,
                v_cache=v_cache,
                cache_seqlens=cache_seqlens,
                causal=True,
                alibi_slopes=self.alibi_slopes,
            )
        # (batch_size, seq_len, hidden_dim)
        out = self.out_proj(out.reshape(B, L, D))
        return out

    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        assert not self.training

        B, L, D = x.size()
        # [batch_size, seq_len, 3, num_heads, head_dim] for FA2
        # qkv = self.in_proj(x).reshape(B, L, 3, self.num_heads, D // self.num_heads)#.transpose(1, 3)
        # [batch_size, num_heads, seq_len, head_dim], num_heads <-> seq_len for torch SDPA/manual self attention implementation
        # q, k, v = qkv.unbind(2)
        q = self.q_proj(x).reshape(B, L, self.num_heads, D // self.num_heads)
        k = self.k_proj(x).reshape(B, L, self.num_heads, D // self.num_heads)
        # v = self.v_proj(x).reshape(B, L, self.num_heads, D // self.num_heads)
        
        if self.normalize_qk:
            q, k = self.q_norm(q), self.k_norm(k)
        
        sink_k = self.sink_k.to(q.dtype).expand(
            B, -1, -1, -1
        )
        # sink_v = self.sink_v.to(qkv.dtype).expand(
        #     B, -1, -1, -1
        # )

        q[:, :self.num_sink_tokens] = 0
        k[:, :self.num_sink_tokens] = sink_k

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        # v = v.transpose(1, 2)

        # attn
        attn = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        
        attn = attn + self.get_alibi_mask(L, x.device).to(x.device)
        
        # [B, nH, L, L]
        attn = F.softmax(attn, dim=-1)
        return attn


    def get_kv_sinks(self) -> tuple[torch.Tensor, torch.Tensor]:
        # [num_sink_tokens, nH, hD]
        return self.sink_k, self.sink_v


class SinkKVTransformerBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int,
            num_heads: int,
            attention_dropout: float,
            residual_dropout: float,
            num_sink_tokens: int = 3,
            normalize_qk: bool = False,
            pre_norm: bool = True,
            with_alibi: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(residual_dropout)

        self.attention = SinkKVSelfAttention(
            hidden_dim,
            num_heads,
            attention_dropout,
            normalize_qk=normalize_qk,
            with_alibi=with_alibi,
            num_sink_tokens=num_sink_tokens
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
            k_cache: torch.Tensor | None = None,
            v_cache: torch.Tensor | None = None,
            cache_seqlens: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.pre_norm:
            attention_out = self.attention(self.norm1(x), k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens)
            x = x + self.drop(attention_out)
            x = x + self.mlp(self.norm2(x))
        else:
            attention_out = self.attention(x, k_cache=k_cache, v_cache=v_cache, cache_seqlens=cache_seqlens)
            x = self.norm1(x + self.drop(attention_out))
            x = self.norm2(x + self.mlp(x))

        return x
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        return self.attention.get_attention_maps(x)

    def get_kv_sinks(self) -> tuple[torch.Tensor, torch.Tensor]:
        # [num_sink_tokens, nH, hD]
        return self.attention.get_kv_sinks()
