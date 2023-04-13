import torch
import torch.nn as nn
import timm


class SwingNetSingleRegression(nn.Module):
    def __init__(self):
        super(SwingNetSingleRegression, self).__init__()
        self.model = timm.create_model(
            "swin_base_patch4_window12_384",
            pretrained=True,
            num_classes=0
        )

        for param in self.model.parameters():
            param.requires_grad = False

        self.drop = nn.Dropout(0.35)
        self.fc2 = nn.Linear(1024, 1)

        # self.drop2 = nn.Dropout(0.35)
        # self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        batch_size = x.size(0)
        # x = F.relu(self.fc1(x))
        # x = self.fc2(x)
        xf = self.model(x)
        x = xf.view(batch_size, -1)
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        # x = F.relu6(x)
        # x = self.fc3(x)
        # x = self.drop2(x)

        return x
