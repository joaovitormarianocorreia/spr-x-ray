import torch.nn as nn
import torch.nn.functional as F


class RegNet(nn.Module):
    def __init__(self):
        super(RegNet, self).__init__()

        self.fc1 = nn.Linear(20, 1)
        self.fc2 = nn.Linear(20, 64)
        # self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        # batch_size = x.size(0)
        x = self.fc1(x)
        x = F.relu(x)
        # x = x.view(batch_size, -1)
        x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)

        return x
