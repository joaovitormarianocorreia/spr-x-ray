import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import ResNet50_Weights


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.model = torchvision.models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )

        return_nodes = {
            "avgpool": "avgpool"
        }

        for name, param in self.model.named_parameters():
            if 'conv' in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.feat = create_feature_extractor(
            self.model,
            return_nodes=return_nodes
        )

        self.fc2 = nn.Linear(2048, 64)

    def forward(self, x):
        batch_size = x.size(0)
        xf = self.feat(x)
        x = xf["avgpool"]
        x = x.view(batch_size, -1)
        x = self.fc2(x)
        return x
