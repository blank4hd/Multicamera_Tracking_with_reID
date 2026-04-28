import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ReIDModel(nn.Module):
    def __init__(self, num_classes=0, embedding_dim=256, pretrained=True, last_stride=1):
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        backbone = resnet50(weights=weights)

        if last_stride == 1:
            backbone.layer4[0].conv2.stride = (1, 1)
            backbone.layer4[0].downsample[0].stride = (1, 1)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.feature_dim = 2048
        self.bnneck = nn.BatchNorm1d(self.feature_dim)
        self.bnneck.bias.requires_grad_(False)
        nn.init.constant_(self.bnneck.weight, 1.0)
        nn.init.constant_(self.bnneck.bias, 0.0)

        self.embedding_layer = nn.Linear(self.feature_dim, embedding_dim, bias=False)
        nn.init.kaiming_normal_(self.embedding_layer.weight, mode="fan_out")

        self.num_classes = num_classes
        if num_classes > 0:
            self.classifier = nn.Linear(self.feature_dim, num_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)
        else:
            self.classifier = None

    def forward(self, x, return_logits=False):
        feat_map = self.backbone(x)
        pooled = self.global_pool(feat_map).flatten(1)
        triplet_features = pooled
        bn_features = self.bnneck(pooled)
        embedding = self.embedding_layer(bn_features)
        embedding = F.normalize(embedding, p=2, dim=1)

        if return_logits and self.classifier is not None:
            logits = self.classifier(bn_features)
            return embedding, triplet_features, logits
        return embedding, triplet_features
