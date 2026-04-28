import torch
import torch.nn as nn
import torch.nn.functional as F


def euclidean_dist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise euclidean distance between rows of x and y.
    x: (N, D), y: (M, D) -> returns (N, M).

    MPS-safe implementation: avoids in-place ops on the sqrt output and
    uses a strictly positive epsilon to keep gradients finite at d=0.
    """
    m, n = x.size(0), y.size(0)
    # Squared L2 norms, broadcast to (m, n)
    xx = (x * x).sum(dim=1, keepdim=True).expand(m, n)
    yy = (y * y).sum(dim=1, keepdim=True).expand(n, m).t()
    # Squared distance; use a fresh variable for the clamped version
    # to avoid any in-place mutation that confuses MPS autograd.
    dist_sq = xx + yy - 2.0 * x @ y.t()
    dist_sq = torch.clamp(dist_sq, min=1e-12)
    dist = torch.sqrt(dist_sq + 1e-12)   # extra epsilon AFTER sqrt arg keeps grad finite
    return dist

class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float | None = 0.3):
        super().__init__()
        self.margin = margin

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        dist = euclidean_dist(features, features)
        labels = labels.view(-1)

        mask_pos = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask_pos.fill_diagonal_(False)
        mask_neg = labels.unsqueeze(0) != labels.unsqueeze(1)

        pos_dist = dist.masked_fill(~mask_pos, float("-inf"))
        neg_dist = dist.masked_fill(~mask_neg, float("inf"))

        valid = mask_pos.any(dim=1) & mask_neg.any(dim=1)
        if not valid.any():
            return dist.sum() * 0.0

        dist_ap = pos_dist.max(dim=1).values[valid]
        dist_an = neg_dist.min(dim=1).values[valid]

        if self.margin is not None:
            loss = F.relu(self.margin + dist_ap - dist_an)
        else:
            loss = F.softplus(dist_ap - dist_an)
        return loss.mean()


class CrossEntropyLabelSmoothing(nn.Module):
    def __init__(self, num_classes: int, epsilon: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.num_classes <= 1:
            return F.cross_entropy(logits, labels)

        log_probs = self.log_softmax(logits)
        with torch.no_grad():
            targets = torch.zeros_like(log_probs).scatter_(1, labels.unsqueeze(1), 1)
            targets = (1 - self.epsilon) * targets + self.epsilon / (self.num_classes - 1) * (1 - targets)
        return (-targets * log_probs).sum(dim=1).mean()


class CombinedReIDLoss(nn.Module):
    def __init__(self, num_classes: int, margin: float | None = 0.3,
                 ce_weight: float = 1.0, triplet_weight: float = 1.0,
                 label_smoothing_epsilon: float = 0.1):
        super().__init__()
        self.ce = CrossEntropyLabelSmoothing(num_classes, epsilon=label_smoothing_epsilon)
        self.triplet = BatchHardTripletLoss(margin=margin)
        self.ce_weight = ce_weight
        self.triplet_weight = triplet_weight

    def forward(self, triplet_features, logits, labels):
        loss_ce = self.ce(logits, labels)
        loss_tri = self.triplet(triplet_features, labels)
        total = self.ce_weight * loss_ce + self.triplet_weight * loss_tri
        return total, {"ce": loss_ce.item(), "triplet": loss_tri.item(), "total": total.item()}
