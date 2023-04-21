import torchxrayvision as xrv
import torch.nn.functional as F
import torch.nn as nn


class XDenseNet(nn.Module):
    def __init__(self):
        super(XDenseNet, self).__init__()
        self.bb = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.bb.op_threshs = None
        self.fc = nn.Linear(1024, 64)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.bb.features2(x)
        x = x.view(batch_size, -1)
        x = self.drop(x)
        x = F.relu(self.fc(x))
        return x
