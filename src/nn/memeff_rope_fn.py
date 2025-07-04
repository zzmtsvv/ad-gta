import torch


def rotate_half(x: torch.Tensor, backward_pass: bool = False) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    
    if not backward_pass:
        return torch.cat((-x2, x1), dim=-1)
    
    return torch.cat((x2, -x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
    backward_pass: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    return q * cos + rotate_half(q, backward_pass) * sin, k * cos + rotate_half(k, backward_pass) * sin


class MemoryEfficientRoPEFunction(torch.autograd.Function):
    @torch.compile
    @staticmethod
    def forward(ctx, q, k, cos, sin, unsqueeze_dim=1):
        ctx.save_for_backward(cos, sin)
        ctx.unsqueeze_dim = unsqueeze_dim
        q, k = apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim, False)
        return q, k

    @torch.compile
    @staticmethod
    def backward(ctx, dq, qk):
        cos, sin = ctx.saved_tensors
        dq, dk = apply_rotary_pos_emb(dq, qk, cos, sin, ctx.unsqueeze_dim, True)
        return dq, dk, None, None, None


class RotaryEmbedding(torch.nn.Module):
    def __init__(
            self,
            dim: int,
            max_seq_len: int,
            base: float = 10_000
    ) -> None:
        super().__init__()
        
        self.base = base
        self.max_seq_len = max_seq_len
        self.dim = dim
        
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        # self.inv_freq = nn.Parameter(inv_freq, requires_grad=False)

        cos, sin = self._prepare_cache(max_seq_len)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)
    
    def _prepare_cache(
            self,
            max_seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        inv_freq_expanded = self.inv_freq[None, :, None].float()#.expand(B, -1, 1)
        t = torch.arange(max_seq_len)[None, None, :].float()#.to(device)

        freqs = (inv_freq_expanded.float() @ t.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()
        
        return cos, sin
    
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.size()

        return self.cos.expand(B, -1, -1)[:, :L, :].to(x.dtype), self.sin.expand(B, -1, -1)[:, :L, :].to(x.dtype)


def apply_rotary_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1
) -> tuple[torch.Tensor, torch.Tensor]:
    return MemoryEfficientRoPEFunction.apply(q, k, cos, sin, unsqueeze_dim)
