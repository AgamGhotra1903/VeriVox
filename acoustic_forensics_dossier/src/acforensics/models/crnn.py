"""CRNN Architecture: 2D CNN Encoder + BiGRU + Multi-Head Self-Attention + Attention Pooling.

Supports both LFCC (60-dim) and Log-Mel (80-dim) inputs with configurable dropouts,
He (Kaiming) initialization for convolutional and linear layers, and Xavier initialization
for attention projections.

Author: Agam Ghotra (IIT2024226)
"""

from typing import Optional, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from acforensics.constants import (
    BONAFIDE,
    SPOOF,
    LFCC_DEFAULT_DIM,
    LOGMEL_DEFAULT_DIM,
)


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
        # Expected input shape: (B, C_in, F, T)
        # Expected output shape: (B, C_out, F // pool_f, T // pool_t)
        return self.block(x)


class CNNEncoder(nn.Module):
    """4-stage 2D CNN hierarchical spectral encoder.

    Preserves temporal resolution (T') while aggressively condensing frequency bands (F').
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: Optional[list[int]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        if channels is None:
            channels = [32, 64, 128, 128]
        self.channels = channels
        self.dropout = dropout

        self.b1 = ConvBlock(in_channels, channels[0], pool_size=(2, 2), dropout=dropout)
        self.b2 = ConvBlock(channels[0], channels[1], pool_size=(2, 2), dropout=dropout)
        self.b3 = ConvBlock(channels[1], channels[2], pool_size=(2, 1), dropout=dropout)
        self.b4 = ConvBlock(channels[2], channels[3], pool_size=(2, 1), dropout=dropout)
        self.out_channels = channels[3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expected input: (B, C=1, F, T)
        assert x.ndim == 4, f"CNNEncoder expected 4D tensor (B, C, F, T), got {tuple(x.shape)}"
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        # Expected output: (B, channels[3], F', T')
        assert x.shape[1] == self.out_channels, (
            f"Expected {self.out_channels} channels, got {x.shape[1]}"
        )
        return x


class AttentionPooling(nn.Module):
    """Learned additive attention pooling over temporal frames.

    Produces frame-level importance weights alpha_t that sum to 1.0,
    which are returned for visualization in the forensic inspection UI.
    """

    def __init__(self, in_dim: int = 256, hidden_dim: int = 64):
        super().__init__()
        self.in_dim = in_dim
        linear1 = nn.Linear(in_dim, hidden_dim)
        linear1._tag = "attention"
        linear2 = nn.Linear(hidden_dim, 1, bias=False)
        linear2._tag = "attention"
        self.score_net = nn.Sequential(
            linear1,
            nn.Tanh(),
            linear2,
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
        assert x.ndim == 3, f"AttentionPooling expected 3D tensor (B, T, D), got {tuple(x.shape)}"
        B, T, D = x.shape
        assert D == self.in_dim, f"AttentionPooling expected feature dimension {self.in_dim}, got {D}"

        # scores shape: (B, T, 1)
        scores = self.score_net(x)
        alpha = F.softmax(scores, dim=1).squeeze(-1)  # (B, T)
        # Weighted sum: (B, T, 1) * (B, T, D) -> sum over T -> (B, D)
        pooled = (alpha.unsqueeze(-1) * x).sum(dim=1)
        assert pooled.shape == (B, D), f"Expected pooled shape ({B}, {D}), got {tuple(pooled.shape)}"
        assert alpha.shape == (B, T), f"Expected alpha shape ({B}, {T}), got {tuple(alpha.shape)}"
        return pooled, alpha


class CRNNForensics(nn.Module):
    """Deployable Convolutional Recurrent Neural Network with Self-Attention.

    Architecture:
    1. 2D CNN Encoder (extracts local spectro-temporal formant anomalies)
    2. Frequency collapse & Linear Projection to proj_dim (default: 128)
    3. BiGRU (captures long-range temporal co-articulation and vocoder pitch consistency)
    4. Multi-Head Self-Attention + Attention Pooling (or Mean Pooling)
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
        conv_dropout: float = 0.1,
        classifier_dropout: float = 0.3,
        num_attn_heads: int = 4,
        num_classes: int = 2,
        init_weights: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.in_freq_dim = in_freq_dim
        self.features = features
        self.use_gru = use_gru
        self.use_attention = use_attention
        self.pooling_mode = pooling
        self.proj_dim = proj_dim
        self.gru_hidden = gru_hidden
        self.num_gru_layers = num_gru_layers
        self.gru_dropout = gru_dropout
        self.conv_dropout = conv_dropout
        self.classifier_dropout = classifier_dropout
        self.num_attn_heads = num_attn_heads
        self.num_classes = num_classes

        # 1. Encoder branch(es)
        if features == "combined":
            # Two-branch architecture: LFCC branch (60) + Log-Mel branch (80)
            self.cnn_lfcc = CNNEncoder(in_channels=1, dropout=conv_dropout)
            self.cnn_logmel = CNNEncoder(in_channels=1, dropout=conv_dropout)
            # Compute collapsed frequency dimensions for both
            # After 4 pools (2x, 2x, 2x, 2x in freq): F' = F // 16
            f_lfcc_out = max(1, LFCC_DEFAULT_DIM // 16)    # 60 // 16 = 3
            f_logmel_out = max(1, LOGMEL_DEFAULT_DIM // 16) # 80 // 16 = 5
            collapsed_dim = (128 * f_lfcc_out) + (128 * f_logmel_out)
        else:
            self.cnn = CNNEncoder(in_channels=in_channels, dropout=conv_dropout)
            f_out = max(1, in_freq_dim // 16)
            collapsed_dim = 128 * f_out

        # 2. Projection before RNN
        self.proj = nn.Sequential(
            nn.Linear(collapsed_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )

        # 3. Recurrent Stage
        self.rnn_out_dim = gru_hidden * 2 if use_gru else proj_dim
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

        # 4. Multi-Head Self-Attention Stage
        if use_attention:
            self.mha = nn.MultiheadAttention(
                embed_dim=self.rnn_out_dim,
                num_heads=num_attn_heads,
                dropout=0.1,
                batch_first=True,
            )
            if hasattr(self.mha, "out_proj") and isinstance(self.mha.out_proj, nn.Linear):
                self.mha.out_proj._tag = "attention"
            self.attn_norm = nn.LayerNorm(self.rnn_out_dim)
        else:
            self.mha = None

        # 5. Pooling Stage
        if pooling == "attention":
            self.pooler = AttentionPooling(in_dim=self.rnn_out_dim)
        else:
            self.pooler = None

        # 6. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(self.rnn_out_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(128, num_classes),
        )

        # 7. Apply Weight Initialization
        if init_weights:
            self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        """Applies He (Kaiming) initialization to conv and linear layers, and Xavier to attention."""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # Check if this linear layer belongs to an attention projection or score net
            # If so, use Xavier initialization; otherwise use He (Kaiming)
            is_attention = any(
                tag in getattr(m, "_tag", "")
                for tag in ("attention", "score_net", "mha")
            )
            if is_attention or m.out_features == 1:
                nn.init.xavier_uniform_(m.weight)
            else:
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.MultiheadAttention):
            if m.in_proj_weight is not None:
                nn.init.xavier_uniform_(m.in_proj_weight)
            if m.out_proj is not None and m.out_proj.weight is not None:
                nn.init.xavier_uniform_(m.out_proj.weight)
            if m.in_proj_bias is not None:
                nn.init.zeros_(m.in_proj_bias)
            if m.out_proj is not None and m.out_proj.bias is not None:
                nn.init.zeros_(m.out_proj.bias)
        elif isinstance(m, nn.GRU):
            for param_name, param in m.named_parameters():
                if "weight_ih" in param_name:
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in param_name:
                    nn.init.orthogonal_(param)
                elif "bias" in param_name:
                    nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        x_aux: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input feature tensor of shape (B, 1, F, T), (B, F, T), or (F, T).
               - For LFCC: F = 60, T ~ 400
               - For Log-Mel: F = 80, T ~ 400
            x_aux: Optional secondary feature for combined two-branch mode.

        Returns:
            Tuple of:
            - logits: (B, num_classes) where col 0 = SPOOF, col 1 = BONAFIDE
            - attention_weights: (B, T') normalized frame importance weights
        """
        # Shape handling on input
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
        elif x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, F, T)

        assert x.dim() == 4, f"CRNN expected 4D input tensor (B, C, F, T), got {tuple(x.shape)}"
        B, C, F_dim, T_dim = x.shape
        assert C == 1, f"Expected 1 channel in spectrogram input, got {C}"

        # 1. CNN Feature Extraction
        if self.features == "combined" and x_aux is not None:
            if x_aux.dim() == 2:
                x_aux = x_aux.unsqueeze(0).unsqueeze(0)
            elif x_aux.dim() == 3:
                x_aux = x_aux.unsqueeze(1)
            assert x_aux.dim() == 4, f"x_aux must be 4D (B, 1, F, T), got {tuple(x_aux.shape)}"

            h1 = self.cnn_lfcc(x)       # (B, 128, F1', T')
            h2 = self.cnn_logmel(x_aux) # (B, 128, F2', T')

            B1, C1, F1_p, T1_p = h1.shape
            B2, C2, F2_p, T2_p = h2.shape
            assert B1 == B2, "Batch size mismatch between two branches"
            T_common = min(T1_p, T2_p)
            h1 = h1[:, :, :, :T_common]
            h2 = h2[:, :, :, :T_common]

            # Flatten C and F: (B, T_common, C*F)
            h1_flat = h1.permute(0, 3, 1, 2).contiguous().view(B, T_common, C1 * F1_p)
            h2_flat = h2.permute(0, 3, 1, 2).contiguous().view(B, T_common, C2 * F2_p)
            h_combined = torch.cat([h1_flat, h2_flat], dim=-1)
            feat_seq = self.proj(h_combined)  # (B, T_common, proj_dim)
            T_p = T_common
        else:
            h = self.cnn(x)  # (B, 128, F', T')
            assert h.dim() == 4, f"CNN output must be 4D, got {tuple(h.shape)}"
            B_h, C_h, F_p, T_p = h.shape
            assert B_h == B and C_h == 128, f"Unexpected CNN output shape: {tuple(h.shape)}"

            # Collapse frequency: (B, T', C*F')
            h_flat = h.permute(0, 3, 1, 2).contiguous().view(B, T_p, C_h * F_p)
            feat_seq = self.proj(h_flat)  # (B, T', proj_dim)

        assert feat_seq.shape == (B, T_p, self.proj_dim), (
            f"Expected feat_seq shape ({B}, {T_p}, {self.proj_dim}), got {tuple(feat_seq.shape)}"
        )

        # 2. Recurrent Modeling
        if self.use_gru and self.gru is not None:
            rnn_seq, _ = self.gru(feat_seq)  # (B, T', gru_hidden * 2)
        else:
            rnn_seq = feat_seq

        assert rnn_seq.shape == (B, T_p, self.rnn_out_dim), (
            f"Expected rnn_seq shape ({B}, {T_p}, {self.rnn_out_dim}), got {tuple(rnn_seq.shape)}"
        )

        # 3. Multi-Head Self-Attention
        if self.use_attention and self.mha is not None:
            attn_out, _ = self.mha(rnn_seq, rnn_seq, rnn_seq)
            rnn_seq = self.attn_norm(rnn_seq + attn_out)

        # 4. Pooling
        if self.pooling_mode == "attention" and self.pooler is not None:
            pooled, attn_weights = self.pooler(rnn_seq)  # pooled: (B, rnn_out_dim), attn: (B, T')
        else:
            pooled = rnn_seq.mean(dim=1)  # (B, rnn_out_dim)
            # Uniform attention placeholder for UI visualization
            attn_weights = torch.full((B, T_p), 1.0 / T_p, device=x.device)

        assert pooled.shape == (B, self.rnn_out_dim), (
            f"Expected pooled shape ({B}, {self.rnn_out_dim}), got {tuple(pooled.shape)}"
        )
        assert attn_weights.shape == (B, T_p), (
            f"Expected attn_weights shape ({B}, {T_p}), got {tuple(attn_weights.shape)}"
        )

        # 5. Classifier Head
        logits = self.classifier(pooled)  # (B, num_classes)
        assert logits.shape == (B, self.num_classes), (
            f"Expected logits shape ({B}, {self.num_classes}), got {tuple(logits.shape)}"
        )

        return logits, attn_weights

    def predict_score(
        self,
        x: torch.Tensor,
        x_aux: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Computes continuous countermeasure score where higher value = more bona fide (genuine).

        Convention:
            Score = Logit(BONAFIDE) - Logit(SPOOF)  (Log-Likelihood Ratio).
            Higher score indicates higher confidence in genuine/authentic speech.

        Args:
            x: Input feature tensor (B, 1, F, T), (B, F, T), or (F, T).
            x_aux: Optional auxiliary tensor for combined two-branch mode.

        Returns:
            1D Tensor of shape (B,) with real-valued scores.
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)

        if x_aux is not None:
            if x_aux.dim() == 2:
                x_aux = x_aux.unsqueeze(0).unsqueeze(0)
            elif x_aux.dim() == 3:
                x_aux = x_aux.unsqueeze(1)

        logits, _ = self.forward(x, x_aux=x_aux)
        return logits[:, BONAFIDE] - logits[:, SPOOF]
