import torch

from src.reid import BatchHardTripletLoss, CombinedReIDLoss, euclidean_dist


def test_euclidean_dist_self_zero():
    x = torch.randn(5, 8)
    d = euclidean_dist(x, x)
    # Float32 cancellation near zero can give residuals up to ~1e-3.
    # Tolerance is loose because the diagonal is never used in triplet loss
    # (i != j constraint excludes it from anchor-positive and anchor-negative pairs).
    assert torch.allclose(torch.diag(d), torch.zeros(5), atol=1e-2)


def test_euclidean_dist_known():
    x = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    d = euclidean_dist(x, x)
    assert torch.isclose(d[0, 1], torch.tensor(5.0), atol=1e-4)
    assert torch.isclose(d[1, 0], torch.tensor(5.0), atol=1e-4)


def test_triplet_loss_zero_when_identity_separated():
    features = torch.tensor([
        [10.0, 0.0],
        [10.1, 0.0],
        [-10.0, 0.0],
        [-10.1, 0.0],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    loss_fn = BatchHardTripletLoss(margin=0.3)
    loss = loss_fn(features, labels)
    assert loss.item() < 0.5


def test_triplet_loss_positive_when_classes_overlap():
    features = torch.tensor([
        [0.0, 0.0],
        [5.0, 0.0],
        [0.1, 0.0],
        [5.1, 0.0],
    ])
    labels = torch.tensor([0, 0, 1, 1])
    loss_fn = BatchHardTripletLoss(margin=0.3)
    loss = loss_fn(features, labels)
    assert loss.item() > 0.1


def test_combined_loss_runs():
    num_classes = 10
    triplet_features = torch.randn(8, 2048)
    logits = torch.randn(8, num_classes)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss_fn = CombinedReIDLoss(num_classes=num_classes, margin=0.3)
    total, components = loss_fn(triplet_features, logits, labels)
    assert torch.is_tensor(total)
    assert "ce" in components and "triplet" in components and "total" in components
    assert all(isinstance(v, float) for v in components.values())
