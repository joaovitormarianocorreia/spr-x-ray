import torchvision
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models import efficientnet


class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.model = torchvision.models.efficientnet_b0(
            pretrained=True,
            weights=efficientnet.EfficientNet_B0_Weights.DEFAULT
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
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        xf = self.feat(x)
        x = xf["avgpool"]
        x = x.view(batch_size, -1)
        x = self.fc2(x)
        return x
