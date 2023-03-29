import torch
import torch.nn as nn


class TwinNet(nn.Module):
    def __init__(self, model: nn.Module,
                 in_features=64, out_features=64,
                 output_dim=1, p_drop=0.3):
        """Modelo Twin

        Args:
            model (nn.Module): Modelo do backbone
            in_features (int, optional): Tamanho do tensor de saída do
                backbone. Defaults to 64.
            out_features (int, optional): Tamanho do tensor de saída da
                camada de projeção. Defaults to 64.
            output_dim (int, optional): _description_. Defaults to 1.
        """
        super(TwinNet, self).__init__()
        self.backbone = model
        self.fc1 = nn.Linear(in_features, out_features)
        self.fc2 = nn.Linear(in_features, out_features)
        self.fc3 = nn.Linear(2*out_features, output_dim)
        self.drop = nn.Dropout(p_drop)

    def forward(self, x1, x2):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        out1 = self.fc1(x1)
        out2 = self.fc2(x2)
        out = torch.cat((out1, out2), dim=1)
        out = self.drop(out)
        out = self.fc3(out)
        return out
