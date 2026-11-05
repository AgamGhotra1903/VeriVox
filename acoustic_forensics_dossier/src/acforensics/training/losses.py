"""Loss functions for class-imbalanced synthetic speech detection.

Author: Agam Ghotra (IIT2024226)
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from acforensics.constants import BONAFIDE, SPOOF


class ClassWeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy loss with inverse frequency weights to counteract ASVspoof imbalance."""

    def __init__(self, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """Focal Loss (Lin et al.) focusing gradient descent on hard spoof/bona fide boundaries.

    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(
        self,
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=-1)
        p = torch.exp(log_p)

        target_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
        p_t = (p * target_one_hot).sum(dim=-1)
        log_p_t = (log_p * target_one_hot).sum(dim=-1)

        focal_weight = torch.pow(1.0 - p_t, self.gamma)

        if self.weight is not None:
            w_t = (self.weight.to(logits.device) * target_one_hot).sum(dim=-1)
            loss = -w_t * focal_weight * log_p_t
        else:
            loss = -focal_weight * log_p_t

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
