import torch
import torch.nn as nn

class _CrossAttentionBlock(nn.Module):
    """ queries <- cross-attention(queries, tokens) <- self-attention between queries <- MLP."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 2.0, self_attention: bool = True):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.self_attention = self_attention
        if self_attention:
            self.norm_s = nn.LayerNorm(dim)
            self.self_attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, q: torch.Tensor, kv: torch.Tensor, need_weights: bool = False):
        kv = self.norm_kv(kv)
        attn_out, weights = self.cross(self.norm_q(q), kv, kv, need_weights=need_weights, average_attn_weights=False)
        q = q + attn_out
        if self.self_attention:
            s = self.norm_s(q)
            q = q + self.self_attn(s, s, s, need_weights=False)[0]
        q = q + self.mlp(self.norm_mlp(q))
        return q, weights


class QueryAdapter(nn.Module):
    def __init__(
        self,
        token_dim: int = 384,
        num_queries: int = 8,
        dim: int = 128,
        heads: int = 4,
        layers: int = 1,
        lidar_rays: int = 0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.num_queries = num_queries
        self.dim = dim
        self.proj = nn.Linear(token_dim, dim)  # tokens -> adapter width
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.blocks = nn.ModuleList([_CrossAttentionBlock(dim, heads) for _ in range(layers)])
        self.norm_out = nn.LayerNorm(dim)
        # representation transfer: predict the teacher's lidar scan (0 = no head)
        self.lidar_head = nn.Linear(num_queries * dim, lidar_rays) if lidar_rays > 0 else None
        self.last_attention = None

    @property
    def out_dim(self) -> int:
        return self.num_queries * self.dim

    def forward(self, tokens: torch.Tensor, keep_attention: bool = False) -> torch.Tensor:
        if tokens.dim() == 2:  # flattened observation (B, T * token_dim)
            tokens = tokens.view(tokens.shape[0], -1, self.token_dim)
        kv = self.proj(tokens.to(self.proj.weight.dtype))
        q = self.queries.unsqueeze(0).expand(kv.shape[0], -1, -1)
        for i, block in enumerate(self.blocks):
            last = i == len(self.blocks) - 1
            q, weights = block(q, kv, need_weights=keep_attention and last)
        if keep_attention:
            self.last_attention = weights.detach()
        return self.norm_out(q).flatten(1)  # (B, num_queries * dim)

    def predict_lidar(self, features: torch.Tensor) -> torch.Tensor:
        assert self.lidar_head is not None, "built without a lidar head (lidar_rays=0)"
        return self.lidar_head(features)
