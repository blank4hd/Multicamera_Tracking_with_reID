import torch

from src.reid import ReIDModel


def test_model_inference_mode():
    model = ReIDModel(num_classes=0, embedding_dim=256, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 256, 128)
    with torch.no_grad():
        embedding, triplet_features = model(x)
    assert embedding.shape == (2, 256)
    assert triplet_features.shape == (2, 2048)
    norms = embedding.norm(p=2, dim=1)
    assert torch.allclose(norms, torch.ones(2), atol=1e-5)


def test_model_training_mode_with_logits():
    num_classes = 100
    model = ReIDModel(num_classes=num_classes, embedding_dim=256, pretrained=False)
    model.train()
    x = torch.randn(4, 3, 256, 128)
    embedding, triplet_features, logits = model(x, return_logits=True)
    assert embedding.shape == (4, 256)
    assert triplet_features.shape == (4, 2048)
    assert logits.shape == (4, num_classes)


def test_model_no_classifier_when_num_classes_zero():
    model = ReIDModel(num_classes=0, embedding_dim=256, pretrained=False)
    assert model.classifier is None


def test_model_last_stride_change():
    model = ReIDModel(num_classes=0, embedding_dim=256, pretrained=False, last_stride=1)
    model.eval()
    x = torch.randn(1, 3, 256, 128)
    with torch.no_grad():
        feat_map = model.backbone(x)
    assert feat_map.shape[2] == 16
    assert feat_map.shape[3] == 8
