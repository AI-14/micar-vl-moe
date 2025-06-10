import math

import torch
import torch.nn as nn
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding
from torchvision.models import ResNet101_Weights, resnet101


class TeLU(nn.Module):
    def __init__(self) -> None:
        """Initializes TeLU."""
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes flow of TeLU.

        Args:
            x: Shape of [b, seq_len, d_model].

        Returns:
            Output: Shape of [b, seq_len, d_model].
        """
        return x * torch.tanh(torch.exp(x))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        """Initializes RMSNorm.

        Args:
            d_model: Hidden dimension.
            eps: A small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the RMSNorm normalization to the input tensor.

        Args:
            x: Shape of [b, seq_len, d_model].

        Returns:
            Output: Shape of [b, seq_len, d_model].
        """
        out = x * torch.rsqrt(
            x.pow(2).mean(-1, keepdim=True) + self.eps
        )  # [b, seq_len, d_model]

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes flow of RMSNorm.

        Args:
            x: Shape of [b, seq_len, d_model].

        Returns:
            Output: Shape of [b, seq_len, d_model].
        """
        out = self._norm(x.float()).type_as(x) * self.weight  # [b, seq_len, d_model]

        return out


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        """Initializes PositionalEmbedding class.

        Args:
            d_model: Hidden dimension.
            seq_len: Maximum sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)  # [seq_len, d_model]
        pos = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # [seq_len, 1]
        dim_pair = torch.arange(0, d_model, 2)  # [d_model // 2]
        div_term = torch.exp(
            dim_pair * (-math.log(10000.0) / d_model)
        )  # [d_model // 2]
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        pe = pe.unsqueeze(0)  # [1, seq_len, d_model]
        self.register_buffer(
            "pe", pe
        )  # Registering positional encodings as a non-learnable parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies PE.

        Args:
            x: Shape of [b, seq_len, d_model].

        Returns:
            Output: Shape of [b, seq_len, d_model].
        """
        seq_len = x.shape[1]
        out = x + self.pe[:, :seq_len, :]  # [b, seq_len, d_model]
        out = self.dropout(out)  # [b, seq_len, d_model]

        return out


class MultiScaleVisionEncoder(nn.Module):
    def __init__(self, d_v: int) -> None:
        """Initializes MultiScaleVisionEncoder.

        Args:
            d_v: Output channel dimension for unified features.
        """
        super().__init__()
        resnet = resnet101(weights=ResNet101_Weights.DEFAULT)

        # Extract ResNet stages
        self.layer1 = nn.Sequential(*list(resnet.children())[:5])  # C2: 256 channels
        self.layer2 = nn.Sequential(*list(resnet.children())[5:6])  # C3: 512 channels
        self.layer3 = nn.Sequential(*list(resnet.children())[6:7])  # C4: 1024 channels
        self.layer4 = nn.Sequential(*list(resnet.children())[7:8])  # C5: 2048 channels

        # FPN lateral connections with BatchNorm and ReLU
        self.fpn_c2 = nn.Sequential(
            nn.Conv2d(256, d_v, kernel_size=1), nn.BatchNorm2d(d_v), nn.ReLU()
        )
        self.fpn_c3 = nn.Sequential(
            nn.Conv2d(512, d_v, kernel_size=1), nn.BatchNorm2d(d_v), nn.ReLU()
        )
        self.fpn_c4 = nn.Sequential(
            nn.Conv2d(1024, d_v, kernel_size=1), nn.BatchNorm2d(d_v), nn.ReLU()
        )
        self.fpn_c5 = nn.Sequential(
            nn.Conv2d(2048, d_v, kernel_size=1), nn.BatchNorm2d(d_v), nn.ReLU()
        )

        # Feature fusion module
        self.fusion = nn.Sequential(
            nn.Conv2d(d_v, d_v, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_v),
            nn.ReLU(),
        )

        # Adaptive pooling instead of fixed downsampling
        self.pool = nn.AdaptiveAvgPool2d((7, 7))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Executes flow of MultiScaleVisionEncoder.

        Args:
            image: Shape of [b, c, h, w].

        Returns:
            Output: Shape of [b, image_seq_len, d_v].
        """
        # Extract features at different scales
        c2 = self.layer1(image)  # [b, 256, h/4, w/4]
        c3 = self.layer2(c2)  # [b, 512, h/8, w/8]
        c4 = self.layer3(c3)  # [b, 1024, h/16, w/16]
        c5 = self.layer4(c4)  # [b, 2048, h/32, w/32]

        # FPN feature processing
        p5 = self.fpn_c5(c5)  # [b, d_v, h/32, w/32]
        p4 = self.fpn_c4(c4) + nn.functional.interpolate(
            p5, size=c4.shape[2:], mode="bilinear", align_corners=False
        )  # [b, d_v, h/16, w/16]
        p3 = self.fpn_c3(c3) + nn.functional.interpolate(
            p4, size=c3.shape[2:], mode="bilinear", align_corners=False
        )  # [b, d_v, h/8, w/8]
        p2 = self.fpn_c2(c2) + nn.functional.interpolate(
            p3, size=c2.shape[2:], mode="bilinear", align_corners=False
        )  # [b, d_v, h/4, w/4]

        # Feature fusion with multi-scale information
        fused = self.fusion(
            p2
            + nn.functional.interpolate(
                p3, size=p2.shape[2:], mode="bilinear", align_corners=False
            )
            + nn.functional.interpolate(
                p4, size=p2.shape[2:], mode="bilinear", align_corners=False
            )
            + nn.functional.interpolate(
                p5, size=p2.shape[2:], mode="bilinear", align_corners=False
            )
        )  # [b, d_v, h/4, w/4]

        # Adaptive pooling and reshape
        pooled = self.pool(fused)  # [b, d_v, 7, 7]
        out = rearrange(pooled, "b c h w -> b (h w) c")  # [b, 49, d_v]

        return out


class TextEmbedding(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int, seq_len: int, dropout: float
    ) -> None:
        """Initializes TextEmbedding.

        Args:
            vocab_size: Vocab size.
            d_model: Hidden dimension.
            seq_len: Sequence length.
            dropout: Dropout probability.
        """
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEmbedding(d_model, seq_len, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes flow of TextEmbedding.

        Args:
            x: Shape of [b, text_seq_len].

        Returns:
            Output: Shape of [b, text_seq_len, d_model].
        """
        out = self.embedding(x) * math.sqrt(self.d_model)  # [b, text_seq_len, d_model]
        out = self.pe(out)  # [b, text_seq_len, d_model]

        return out


class FeedForward(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, dropout: float, act_fn: nn.Module
    ) -> None:
        """Initializes FeedForward.

        Args:
            d_model: Hidden dimension.
            d_ff: Hidden dimension of feedforward.
            dropout: Dropout probability.
            act_fn: Activation function among SiLU, GELU, SELU, and TeLU.
        """
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.act_fn = act_fn()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Executes flow of FeedFoward.

        Args:
            x: Shape of [b, seq_len, d_model].

        Returns:
            Output: Shape of [b, seq_len, d_model].
        """
        intermediate = self.act_fn(self.w1(x))  # [b, seq_len, d_ff]
        x_v = self.w3(x)  # [b, seq_len, d_ff]
        out = intermediate * x_v  # [b, seq_len, d_ff]
        out = self.dropout(self.w2(out))  # [b, seq_len, d_model]

        return out


class MoEGate(nn.Module):
    def __init__(self, d_model: int, num_experts: int, k: int) -> None:
        """Initializes MoEGate.

        Args:
            d_model: Hidden dimension.
            num_experts: Number of experts.
            k: Topk.
        """
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.gate = nn.Linear(d_model, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Executes flow of MoEGate.

        Args:
            x: Shape of [b * seq_len, d_model].

        Returns:
            Topk scores: Shape of [b * seq_len, k].
            Topk indices: Shape of [b * seq_len, k].
            Loss: Load balancing loss.
        """
        logits = self.gate(x)  # [b * seq_len, num_experts]
        scores = nn.functional.softmax(logits, dim=-1)  # [b * seq_len, num_experts]

        # Select top-k experts for each token
        topk_scores, topk_indices = torch.topk(
            scores, self.k, dim=-1
        )  # [b * seq_len, k]
        topk_scores /= topk_scores.sum(
            dim=-1, keepdim=True
        )  # Normalize top-k scores [b * seq_len, k]

        # Load balancing loss to encourage uniform expert usage
        expert_prob = scores.mean(dim=0)  # [num_experts]
        load_balancing_loss = (expert_prob * torch.log(expert_prob + 1e-8)).sum()

        return topk_scores, topk_indices, load_balancing_loss


class MoEModule(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float,
        act_fn: nn.Module,
        num_experts: int,
        k: int,
    ) -> None:
        """Initializes MoEModule.

        Args:
            d_model: Hidden dimension.
            d_ff: Hidden dimension of feedforward.
            dropout: Dropout probability.
            act_fn: Activation function among SiLU, GELU, SELU, and TeLU.
            num_experts: Number of experts.
            k: Topk.
        """
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList(
            [FeedForward(d_model, d_ff, dropout, act_fn) for _ in range(num_experts)]
        )
        self.gate = MoEGate(d_model, num_experts, k)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        """Executes flow of MoEModule.

        Args:
            x: Shape of [b, seq_len, d_model].

        Returns:
            Output: Shape of [b, seq_len, d_model].
            Loss: Load balancing loss.
        """
        batch_size, seq_len, input_dim = x.shape
        x = x.reshape(-1, input_dim)  # [b * seq_len, d_model]

        # Get top-k scores, expert indices, and load balancing loss
        topk_scores, topk_indices, load_balancing_loss = self.gate(
            x
        )  # [b * seq_len, k], [b * seq_len, k]

        output = torch.zeros_like(x)  # [b * seq_len, d_model]

        # Route inputs through selected experts
        for i in range(self.num_experts):
            mask = (topk_indices == i).float()  # [b * seq_len, k]
            if mask.sum() == 0:
                continue  # Skip if no inputs for this expert

            # Select inputs for the current expert
            selected_inputs = x * mask.sum(
                dim=-1, keepdim=True
            )  # [b * seq_len, d_model]
            expert_output = self.experts[i](selected_inputs)  # [b * seq_len, d_model]

            # Accumulate expert outputs weighted by top-k scores
            output += expert_output * topk_scores.sum(
                dim=-1, keepdim=True
            )  # [b * seq_len, d_model]

        output = output.reshape(batch_size, seq_len, input_dim)  # [b, seq_len, d_model]

        return output, load_balancing_loss


class MHLAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_latent: int,
        qk_nope_dim: int,
        qk_rope_dim: int,
        attention_dropout: float,
    ) -> None:
        """Initializes MHLAttention.

        Args:
            num_heads: Number of heads.
            d_model: Hidden dimension.
            d_latent: Latent dimension.
            qk_nope_dim: Latent dimension for no RoPE layer.
            qk_rope_dim: Latent dimension for RoPE layer.
            attention_dropout: Dropout probability.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert d_latent % num_heads == 0, "d_latent must be divisible by num_heads"

        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.qk_nope_dim = qk_nope_dim
        self.qk_rope_dim = qk_rope_dim

        self.qkv_down_norm_up = nn.Sequential(
            nn.Linear(d_model, 3 * self.dk, bias=False),
            RMSNorm(3 * self.dk),
            nn.Linear(3 * self.dk, 3 * d_latent, bias=False),
        )
        self.dropout = nn.Dropout(attention_dropout)
        self.attention_weights: torch.Tensor | None = None
        self.rope = RotaryEmbedding(dim=qk_rope_dim // 2)
        self.wo = nn.Linear(d_latent, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Executes flow of MHLAttention.

        Args:
            x: Shape of [b, seq_len, d_model].
            mask: Shape of [b, q_seq_len, k_seq_len].

        Returns:
            Output: Shape of [b, seq_len, d_model].
        """
        qkv = self.qkv_down_norm_up(x)  # [b, seq_len, 3 * d_latent]
        q, k, v = torch.chunk(
            qkv, 3, dim=-1
        )  # [b, q_seq_len, d_latent], [b, k_seq_len, d_latent], [b, v_seq_len, d_latent]

        # Split into num_heads
        q = rearrange(
            q, "b s (d h) -> b h s d", h=self.num_heads
        )  # [b, num_heads, q_seq_len, d_latent // num_heads]
        k = rearrange(
            k, "b s (d h) -> b h s d", h=self.num_heads
        )  # [b, num_heads, k_seq_len, d_latent // num_heads]
        v = rearrange(
            v, "b s (d h) -> b h s d", h=self.num_heads
        )  # [b, num_heads, v_seq_len, d_latent // num_heads]

        # Split q and k for non-RoPE and RoPE
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1
        )  # [b, num_heads, q_seq_len, qk_nope_dim], [b, num_heads, q_seq_len, qk_rope_dim]
        k_nope, k_rope = torch.split(
            k, [self.qk_nope_dim, self.qk_rope_dim], dim=-1
        )  # [b, num_heads, k_seq_len, qk_nope_dim], [b, num_heads, k_seq_len, qk_rope_dim]

        # Apply RoPE
        q_rope = self.rope.rotate_queries_or_keys(
            q_rope
        )  # [b, num_heads, q_seq_len, qk_rope_dim]
        k_rope = self.rope.rotate_queries_or_keys(
            k_rope
        )  # [b, num_heads, k_seq_len, qk_rope_dim]

        # Calculate attention
        attention_scores_nope = torch.einsum(
            "b n i d, b n j d -> b n i j", q_nope, k_nope
        )  # [b, num_heads, q_seq_len, k_seq_len]
        attention_scores_rope = torch.einsum(
            "b n i d, b n j d -> b n i j", q_rope, k_rope
        )  # [b, num_heads, q_seq_len, k_seq_len]

        total_attention_scores = (
            attention_scores_nope + attention_scores_rope
        ) / math.sqrt(
            self.qk_nope_dim + self.qk_rope_dim
        )  # [b, num_heads, q_seq_len, k_seq_len]

        if mask is not None:
            assert mask.shape[1] == q.shape[2], (
                "sequence length of mask must be equal to the sequence length of the query input"
            )
            mask = mask.unsqueeze(1)  # [b, 1, q_seq_len, k_seq_len]
            total_attention_scores.masked_fill_(mask == 0, value=-1e9)

        total_attention_scores = total_attention_scores.softmax(
            dim=-1
        )  # [b, num_heads, q_seq_len, k_seq_len]
        total_attention_scores = self.dropout(
            total_attention_scores
        )  # [b, num_heads, q_seq_len, k_seq_len]
        self.attention_weights = total_attention_scores

        out = torch.einsum(
            "b n i j, b n j d -> b n i d", total_attention_scores, v
        )  # [b, num_heads, q_seq_len, d_latent // num_heads]
        out = rearrange(
            out, "b h s d -> b s (h d)", h=self.num_heads
        )  # [b, q_seq_len, d_latent]

        # Final projection
        out = self.wo(out)  # [b, q_seq_len, d_model]

        return out


class CMHLAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        d_model: int,
        d_latent: int,
        qk_nope_dim: int,
        qk_rope_dim: int,
        attention_dropout: float,
    ) -> None:
        """Initializes CMHLAttention.

        Args:
            num_heads: Number of heads.
            d_model: Hidden dimension.
            d_latent: Latent dimension.
            qk_nope_dim: Latent dimension for no RoPE layer.
            qk_rope_dim: Latent dimension for RoPE layer.
            attention_dropout: Dropout probability.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert d_latent % num_heads == 0, "d_latent must be divisible by num_heads"

        self.num_heads = num_heads
        self.dk = d_model // num_heads
        self.qk_nope_dim = qk_nope_dim
        self.qk_rope_dim = qk_rope_dim

        self.q_down_norm_up = nn.Sequential(
            nn.Linear(d_model, self.dk, bias=False),
            RMSNorm(self.dk),
            nn.Linear(self.dk, d_latent, bias=False),
        )
        self.kv_down_norm_up = nn.Sequential(
            nn.Linear(d_model, 2 * self.dk, bias=False),
            RMSNorm(2 * self.dk),
            nn.Linear(2 * self.dk, 2 * d_latent, bias=False),
        )
        self.dropout = nn.Dropout(attention_dropout)
        self.attention_weights: torch.Tensor | None = None
        self.rope = RotaryEmbedding(dim=qk_rope_dim // 2)
        self.wo = nn.Linear(d_latent, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        image_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Executes flow of CMHLAttention.

        Args:
            x: Shape of [b, seq_len, d_model].
            image_feats: Shape of [b, image_seq_len, d_model].

        Returns:
            Output: Shape of [b, seq_len, d_model].
        """
        q = self.q_down_norm_up(x)  # [b, seq_len, d_latent]
        kv = self.kv_down_norm_up(image_feats)  # [b, image_seq_len, 2 * d_latent]
        k, v = torch.chunk(
            kv, 2, dim=-1
        )  # [b, k_seq_len, d_latent], [b, v_seq_len, d_latent]

        # Split into num_heads
        q = rearrange(
            q, "b s (d h) -> b h s d", h=self.num_heads
        )  # [b, num_heads, q_seq_len, d_latent // num_heads]
        k = rearrange(
            k, "b s (d h) -> b h s d", h=self.num_heads
        )  # [b, num_heads, k_seq_len, d_latent // num_heads]
        v = rearrange(
            v, "b s (d h) -> b h s d", h=self.num_heads
        )  # [b, num_heads, v_seq_len, d_latent // num_heads]

        # Split q and k for non-RoPE and RoPE
        q_nope, q_rope = torch.split(
            q, [self.qk_nope_dim, self.qk_rope_dim], dim=-1
        )  # [b, num_heads, q_seq_len, qk_nope_dim], [b, num_heads, q_seq_len, qk_rope_dim]
        k_nope, k_rope = torch.split(
            k, [self.qk_nope_dim, self.qk_rope_dim], dim=-1
        )  # [b, num_heads, k_seq_len, qk_nope_dim], [b, num_heads, k_seq_len, qk_rope_dim]

        # Apply RoPE
        q_rope = self.rope.rotate_queries_or_keys(
            q_rope
        )  # [b, num_heads, q_seq_len, qk_rope_dim]
        k_rope = self.rope.rotate_queries_or_keys(
            k_rope
        )  # [b, num_heads, k_seq_len, qk_rope_dim]

        # Calculate attention
        attention_scores_nope = torch.einsum(
            "b n i d, b n j d -> b n i j", q_nope, k_nope
        )  # [b, num_heads, q_seq_len, k_seq_len]
        attention_scores_rope = torch.einsum(
            "b n i d, b n j d -> b n i j", q_rope, k_rope
        )  # [b, num_heads, q_seq_len, k_seq_len]

        total_attention_scores = (
            attention_scores_nope + attention_scores_rope
        ) / math.sqrt(
            self.qk_nope_dim + self.qk_rope_dim
        )  # [b, num_heads, q_seq_len, k_seq_len]

        total_attention_scores = total_attention_scores.softmax(
            dim=-1
        )  # [b, num_heads, q_seq_len, k_seq_len]
        total_attention_scores = self.dropout(
            total_attention_scores
        )  # [b, num_heads, q_seq_len, k_seq_len]
        self.attention_weights = total_attention_scores

        out = torch.einsum(
            "b n i j, b n j d -> b n i d", total_attention_scores, v
        )  # [b, num_heads, q_seq_len, d_latent // num_heads]
        out = rearrange(
            out, "b h s d -> b s (h d)", h=self.num_heads
        )  # [b, q_seq_len, d_latent]

        # Final projection
        out = self.wo(out)  # [b, q_seq_len, d_model]

        return out


class GatedFusion(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_latent: int,
        num_heads: int,
        qk_nope_dim: int,
        qk_rope_dim: int,
        act_fn: nn.Module,
        attention_dropout: float,
        dropout: float,
    ) -> None:
        """Initializes GatedFusion.

        Args:
            d_model: Hidden dimension.
            d_latent: Latent dimension.
            num_heads: Number of heads.
            qk_nope_dim: Latent dimension for no RoPE layer.
            qk_rope_dim: Latent dimension for RoPE layer.
            act_fn: Activation function among SiLU, GELU, SELU, and TeLU.
            attention_dropout: Dropout probability.
            dropout: Dropout probability.
        """
        super().__init__()

        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)

        self.cross_attention = CMHLAttention(
            num_heads, d_model, d_latent, qk_nope_dim, qk_rope_dim, attention_dropout
        )

        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            act_fn(),
            nn.Linear(d_model, d_model),
            nn.Sigmoid(),
        )
        self.global_proj = nn.Linear(d_model * 2, d_model)

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            act_fn(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, image_feats: torch.Tensor) -> torch.Tensor:
        """Executes flow of GatedFusion.

        Args:
            x: Shape of [b, seq_len, d_model].
            image_feats: Shape of [b, image_seq_len, d_model].

        Returns:
            Output: Shape of [b, seq_len, d_model].
        """
        # Cross-attention block with pre-norm
        normed_x = self.norm1(x)  # [b, seq_len, d_model]
        normed_img = self.norm2(image_feats)  # [b, 49, d_model]

        attended_img = self.cross_attention(normed_x, normed_img)
        attended_img = attended_img + x  # Residual connection

        # Gating block i.e. global context from normalized image features
        global_img = torch.cat(
            [
                normed_img.mean(dim=1, keepdim=True),  # [b, 1, d_model]
                normed_img.max(dim=1, keepdim=True)[0],  # [b, 1, d_model]
            ],
            dim=-1,
        )  # [b, 1, d_model*2]
        global_img = self.global_proj(global_img)  # [b, 1, d_model]

        # Pre-norm before gating
        normed_attended = self.norm3(attended_img)  # [b, seq_len, d_model]

        # Compute gate using normalized features
        gate_input = torch.cat(
            [normed_attended, normed_x], dim=-1
        )  # [b, seq_len, d_model*2]
        gate = self.gate(gate_input)  # [b, seq_len, d_model]

        # Apply gated fusion
        gated = gate * normed_attended + (1 - gate) * normed_x  # [b, seq_len, d_model]

        # Final fusion block
        combined = torch.cat(
            [
                gated,  # [b, seq_len, d_model]
                normed_attended,  # [b, seq_len, d_model]
                global_img.expand(-1, x.shape[1], -1),  # [b, seq_len, d_model]
            ],
            dim=-1,
        )  # [b, seq_len, d_model*3]

        # Apply fusion and add residual
        output = self.fusion(combined) + attended_img  # [b, seq_len, d_model]

        return output


class EncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_latent: int,
        d_ff: int,
        num_heads: int,
        qk_nope_dim: int,
        qk_rope_dim: int,
        act_fn: nn.Module,
        attention_dropout: float,
        dropout: float,
    ) -> None:
        """Initializes EncoderBlock.

        Args:
            d_model: Hidden dimension.
            d_latent: Latent dimension.
            d_ff: Hidden dimension of feedforward.
            num_heads: Number of heads.
            qk_nope_dim: Latent dimension for no RoPE layer.
            qk_rope_dim: Latent dimension for RoPE layer.
            act_fn: Activation function among SiLU, GELU, SELU, and TeLU.
            attention_dropout: Dropout probability.
            dropout: Dropout probability.
        """
        super().__init__()
        self.norm_sa = RMSNorm(d_model)
        self.self_attention = MHLAttention(
            num_heads, d_model, d_latent, qk_nope_dim, qk_rope_dim, attention_dropout
        )

        self.norm_ff = RMSNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout, act_fn)

    def forward(self, image_feats: torch.Tensor) -> torch.Tensor:
        """Executes flow of EncoderBlock.

        Args:
            image_feats: Shape of [b, image_seq_len, d_model].

        Returns:
            Output: Shape of [b, image_seq_len, d_model].
        """
        residual_sa = image_feats
        out_sa = self.norm_sa(image_feats)  # [b, image_seq_len, d_model]
        out_sa = self.self_attention(out_sa)  # [b, image_seq_len, d_model]
        out_sa = residual_sa + out_sa  # [b, image_seq_len, d_model]

        residual_ff = out_sa
        out_ff = self.norm_ff(out_sa)  # [b, image_seq_len, d_model]
        out_ff = self.ff(out_ff)  # [b, image_seq_len, d_model]
        out_ff = residual_ff + out_ff  # [b, image_seq_len, d_model]

        final_out = out_ff  # [b, image_seq_len, d_model]

        return final_out


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_latent: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        qk_nope_dim: int,
        qk_rope_dim: int,
        act_fn: nn.Module,
        attention_dropout: float,
        dropout: float,
    ) -> None:
        """Initializes Encoder.

        Args:
            d_model: Hidden dimension.
            d_latent: Latent dimension.
            d_ff: Hidden dimension of feedforward.
            num_heads: Number of heads.
            num_layers: Number of layers.
            qk_nope_dim: Latent dimension for no RoPE layer.
            qk_rope_dim: Latent dimension for RoPE layer.
            act_fn: Activation function among SiLU, GELU, SELU, and TeLU.
            attention_dropout: Dropout probability.
            dropout: Dropout probability.
        """
        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model,
                    d_latent,
                    d_ff,
                    num_heads,
                    qk_nope_dim,
                    qk_rope_dim,
                    act_fn,
                    attention_dropout,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, image_feats: torch.Tensor) -> torch.Tensor:
        """Executes flow of Encoder.

        Args:
            image_feats: Shape of [b, image_seq_len, d_model].

        Returns:
            Output: Shape of [b, image_seq_len, d_model].
        """
        out = image_feats

        for layer in self.layers:
            out = layer(out)  # [b, image_seq_len, d_model]

        return out


class DecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_latent: int,
        d_ff: int,
        num_heads: int,
        qk_nope_dim: int,
        qk_rope_dim: int,
        act_fn: nn.Module,
        attention_dropout: float,
        dropout: float,
        num_experts: int,
        k: int,
    ) -> None:
        """Initializes DecoderBlock.

        Args:
            d_model: Hidden dimension.
            d_latent: Latent dimension.
            d_ff: Hidden dimension of feedforward.
            num_heads: Number of heads.
            qk_nope_dim: Latent dimension for no RoPE layer.
            qk_rope_dim: Latent dimension for RoPE layer.
            act_fn: Activation function among SiLU, GELU, SELU, and TeLU.
            attention_dropout: Dropout probability.
            dropout: Dropout probability.
            num_experts: Number of experts.
            k: Topk.
        """
        super().__init__()
        self.norm_sa = RMSNorm(d_model)
        self.self_attention = MHLAttention(
            num_heads, d_model, d_latent, qk_nope_dim, qk_rope_dim, attention_dropout
        )

        self.gated_fusion = GatedFusion(
            d_model,
            d_latent,
            num_heads,
            qk_nope_dim,
            qk_rope_dim,
            act_fn,
            attention_dropout,
            dropout,
        )

        self.norm_moe = RMSNorm(d_model)
        self.moe = MoEModule(d_model, d_ff, dropout, act_fn, num_experts, k)

    def forward(
        self,
        x: torch.Tensor,
        image_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Executes flow of DecoderBlock.

        Args:
            x: Shape of [b, seq_len, d_model].
            image_feats: Shape of [b, image_seq_len, d_model].
            mask: Shape of [b, q_seq_len, k_seq_len].

        Returns:
            Output: Shape of [b, seq_len, d_model].
            Loss: Load balancing loss.
        """
        residual_sa = x
        out_sa = self.norm_sa(x)  # [b, seq_len, d_model]
        out_sa = self.self_attention(out_sa, mask)  # [b, seq_len, d_model]
        out_sa = residual_sa + out_sa  # [b, seq_len, d_model]

        residual_gf = out_sa
        out_gf = self.gated_fusion(out_sa, image_feats)  # [b, seq_len, d_model]
        out_gf = residual_gf + out_gf  # [b, seq_len, d_model]

        # Moe
        residual_moe = out_gf
        moe_out = self.norm_moe(out_gf)  # [b, seq_len, d_model]
        moe_out, lb_loss = self.moe(moe_out)  # [b, seq_len, d_model]
        moe_out = residual_moe + moe_out  # [b, seq_len, d_model]

        final_out = moe_out

        return final_out, lb_loss


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_latent: int,
        d_ff: int,
        num_heads: int,
        num_layers: int,
        qk_nope_dim: int,
        qk_rope_dim: int,
        act_fn: nn.Module,
        attention_dropout: float,
        dropout: float,
        num_experts: int,
        k: int,
    ) -> None:
        """Initializes Decoder.

        Args:
            d_model: Hidden dimension.
            d_latent: Latent dimension.
            d_ff: Hidden dimension of feedforward.
            num_heads: Number of heads.
            num_layers: Number of layers.
            qk_nope_dim: Latent dimension for no RoPE layer.
            qk_rope_dim: Latent dimension for RoPE layer.
            act_fn: Activation function among SiLU, GELU, SELU, and TeLU.
            attention_dropout: Dropout probability.
            dropout: Dropout probability.
            num_experts: Number of experts.
            k: Topk.
        """
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model,
                    d_latent,
                    d_ff,
                    num_heads,
                    qk_nope_dim,
                    qk_rope_dim,
                    act_fn,
                    attention_dropout,
                    dropout,
                    num_experts,
                    k,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        image_feats: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, float]:
        """Executes flow of Decoder.

        Args:
            x: Shape of [b, seq_len, d_model].
            image_feats: Shape of [b, image_seq_len, d_model].
            mask: Shape of [b, q_seq_len, k_seq_len].

        Returns:
            Output: Shape of [b, seq_len, d_model].
            Loss: Total load balancing loss.
        """
        total_lb_loss = 0.0

        out = x

        for layer in self.layers:
            out, lb_loss = layer(out, image_feats, mask)  # [b, seq_len, d_model]
            total_lb_loss = total_lb_loss + lb_loss

        total_lb_loss = total_lb_loss / self.num_layers

        return out, total_lb_loss
