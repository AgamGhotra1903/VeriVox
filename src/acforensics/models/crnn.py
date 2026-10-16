"""CRNN Architecture: 2D CNN Encoder + BiGRU + Multi-Head Self-Attention + Attention Pooling.

Author: Agam Ghotra (IIT2024226)
"""

from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from acforensics.constants import LFCC_DEFAULT_DIM, LOGMEL_DEFAULT_DIM


class ConvBlock(nn.Module):
    """Two-layer 2D Convolutional block with BatchNorm, ReLU, MaxPool, and Spatial Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pool_size: tuple[int, int] = (2, 2),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size, stride=pool_size),
            nn.Dropout2d(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CNNEncoder(nn.Module):
    """4-stage 2D CNN hierarchical spectral encoder.

    Preserves temporal resolution (T') while aggressively condensing frequency bands (F').
    """

    def __init__(self, in_channels: int = 1, channels: list[int] = [32, 64, 128, 128], dropout: float = 0.1):
        super().__init__()
        self.b1 = ConvBlock(in_channels, channels[0], pool_size=(2, 2), dropout=dropout)
        self.b2 = ConvBlock(channels[0], channels[1], pool_size=(2, 2), dropout=dropout)
        self.b3 = ConvBlock(channels[1], channels[2], pool_size=(2, 1), dropout=dropout)
        self.b4 = ConvBlock(channels[2], channels[3], pool_size=(2, 1), dropout=dropout)
        self.out_channels = channels[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, 1, F, T)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        return x  # (B, 128, F', T')


class AttentionPooling(nn.Module):
    """Learned additive attention pooling over temporal frames.

    Produces frame-level importance weights alpha_t that sum to 1.0,
    which are returned for visualization in the forensic inspection UI.
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculates attention pooled vector and temporal weights.

        Args:
            x: Tensor of shape (B, T, D).

        Returns:
            Tuple of:
            - pooled representation: (B, D)
            - attention weights alpha: (B, T)
        """
        # scores shape: (B, T, 1)
        scores = self.score_net(x)
        alpha = F.softmax(scores, dim=1).squeeze(-1)  # (B, T)
        # Weighted sum: (B, T, 1) * (B, T, D) -> sum over T -> (B, D)
        pooled = (alpha.unsqueeze(-1) * x).sum(dim=1)
        return pooled, alpha


class CRNNForensics(nn.Module):
    """Deployable Convolutional Recurrent Neural Network with Self-Attention.

    Architecture:
    1. 2D CNN Encoder (extracts local spectro-temporal formant anomalies)
    2. Frequency collapse & Linear Projection to 128-d
    3. BiGRU (captures long-range temporal co-articulation and vocoder pitch consistency)
    4. Multi-Head Self-Attention + Attention Pooling
    5. Linear Classifier Head (outputs logits for bona fide vs spoof)
    """

    def __init__(
        self,
        in_channels: int = 1,
        in_freq_dim: int = LFCC_DEFAULT_DIM,
        features: str = "lfcc",  # 'lfcc', 'logmel', 'combined'
        use_gru: bool = True,
        use_attention: bool = True,
        pooling: Literal["attention", "mean"] = "attention",
        proj_dim: int = 128,
        gru_hidden: int = 128,
        num_gru_layers: int = 2,
        gru_dropout: float = 0.3,
        num_attn_heads: int = 4,
        num_classes: int = 2,
    ):
        super().__init__()
        self.features = features
        self.use_gru = use_gru
        self.use_attention = use_attention
        self.pooling_mode = pooling

        # Encoder branch(es)
        if features == "combined":
            # Two-branch architecture: LFCC branch (60) + Log-Mel branch (80)
            self.cnn_lfcc = CNNEncoder(in_channels=1)
            self.cnn_logmel = CNNEncoder(in_channels=1)
            # Compute collapsed frequency dimensions for both
            # After 4 pools: F' = F // (2 * 2 * 2 * 2) = F // 16
            f_lfcc_out = max(1, LFCC_DEFAULT_DIM // 16)   # 60 // 16 = 3
            f_logmel_out = max(1, LOGMEL_DEFAULT_DIM // 16) # 80 // 16 = 5
            collapsed_dim = (128 * f_lfcc_out) + (128 * f_logmel_out)
        else:
            self.cnn = CNNEncoder(in_channels=in_channels)
            f_out = max(1, in_freq_dim // 16)
            collapsed_dim = 128 * f_out

        # Projection before RNN
        self.proj = nn.Sequential(
            nn.Linear(collapsed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )

        # Recurrent Stage
        rnn_out_dim = gru_hidden * 2 if use_gru else proj_dim
        if use_gru:
            self.gru = nn.GRU(
                input_size=proj_dim,
                hidden_size=gru_hidden,
                num_layers=num_gru_layers,
                dropout=gru_dropout if num_gru_layers > 1 else 0.0,
                bidirectional=True,
                batch_first=True,
            )
        else:
            self.gru = None

        # Self-Attention Stage
        if use_attention:
            self.mha = nn.MultiheadAttention(
                embed_dim=rnn_out_dim,
                num_heads=num_attn_heads,
                dropout=0.1,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(rnn_out_dim)
        else:
            self.mha = None

        # Pooling Stage
        if pooling == "attention":
            self.pooler = AttentionPooling(in_dim=rnn_out_dim)
        else:
            self.pooler = None

        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_aux: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input feature tensor of shape (B, 1, F, T) or (B, F, T).
            x_aux: Optional secondary feature for combined two-branch mode.

        Returns:
            Tuple of:
            - logits: (B, 2)
            - attention_weights: (B, T')
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, F, T)

        # 1. CNN Feature Extraction
        if self.features == "combined" and x_aux is not None:
            if x_aux.dim() == 3:
                x_aux = x_aux.unsqueeze(1)
            h1 = self.cnn_lfcc(x)       # (B, 128, F1', T')
            h2 = self.cnn_logmel(x_aux) # (B, 128, F2', T')

            B, C1, F1, T1 = h1.shape
            B, C2, F2, T2 = h2.shape
            T_common = min(T1, T2)
            h1 = h1[:, :, :, :T_common]
            h2 = h2[:, :, :, :T_common]

            # Flatten C and F: (B, T, C*F)
            h1_flat = h1.permute(0, 3, 1, 2).contiguous().view(B, T_common, C1 * F1)
            h2_flat = h2.permute(0, 3, 1, 2).contiguous().view(B, T_common, C2 * F2)
            h_combined = torch.cat([h1_flat, h2_flat], dim=-1)
            feat_seq = self.proj(h_combined)
        else:
            h = self.cnn(x)  # (B, 128, F', T')
            B, C, F_p, T_p = h.shape
            # Collapse frequency: (B, T', C*F')
            h_flat = h.permute(0, 3, 1, 2).contiguous().view(B, T_p, C * F_p)
            feat_seq = self.proj(h_flat)  # (B, T', proj_dim)

        # 2. Recurrent Modeling
        if self.use_gru and self.gru is not None:
            rnn_seq, _ = self.gru(feat_seq)  # (B, T', 256)
        else:
            rnn_seq = feat_seq

        # 3. Multi-Head Self-Attention
        if self.use_attention and self.mha is not None:
            attn_out, _ = self.mha(rnn_seq, rnn_seq, rnn_seq)
            rnn_seq = self.attn_norm(rnn_seq + attn_out)

        # 4. Pooling
        if self.pooling_mode == "attention" and self.pooler is not None:
            pooled, attn_weights = self.pooler(rnn_seq)
        else:
            pooled = rnn_seq.mean(dim=1)
            # Uniform attention placeholder for UI
            B, T_p, _ = rnn_seq.shape
            attn_weights = torch.full((B, T_p), 1.0 / T_p, device=x.device)

        # 5. Classifier Head
        logits = self.classifier(pooled)
        return logits, attn_weights
