import torch
import torch.nn as nn


class EmbeddingClassifier(nn.Module):
    def __init__(self):
        super(EmbeddingClassifier, self).__init__()
        self.drop = nn.Dropout(0.35)
        self.fc = nn.Linear(1024, 1)

    def forward(self, x):
        # batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        x = self.drop(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x


class KamilJanczyk(nn.Module):
    def __init__(self):
        super(KamilJanczyk, self).__init__()
        self.global_average_pooling = nn.AvgPool1d(3, padding=1)
        self.dropout1 = nn.Dropout(0.35)
        self.dense_layer = nn.Linear(342, 128)
        self.dropout2 = nn.Dropout(0.35)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        # batch_size = x.size(0)
        # x = x.view(batch_size, -1)
        x = self.global_average_pooling(x)
        x = self.dropout1(x)
        x = self.dense_layer(x)
        x = self.dropout2(x)
        x = self.fc(x)
        x = torch.sigmoid(x)
        return x
