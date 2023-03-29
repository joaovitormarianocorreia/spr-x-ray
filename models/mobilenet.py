import torch.nn as nn
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import MobileNet_V3_Small_Weights


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        self.model = torchvision.models.mobilenet_v3_small(
            pretrained=True,
            weights=MobileNet_V3_Small_Weights.DEFAULT
        )

        return_nodes = {
            "avgpool": "avgpool"
        }

        for param in self.model.parameters():
            param.requires_grad = True

        self.feat = create_feature_extractor(
            self.model,
            return_nodes=return_nodes
        )

        self.fc2 = nn.Linear(576, 64)

    def forward(self, x):
        batch_size = x.size(0)
        xf = self.feat(x)
        x = xf["avgpool"]
        x = x.view(batch_size, -1)
        x = self.fc2(x)
        return x
