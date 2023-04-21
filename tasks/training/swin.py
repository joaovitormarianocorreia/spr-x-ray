import argparse
import tqdm
import os.path as opath
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from loader import XRayEmbeddingDataset
from torch.optim.lr_scheduler import CosineAnnealingLR


class RegressionBlock(nn.Module):
    def __init__(self):
        super(RegressionBlock, self).__init__()
        self.drop = nn.Dropout(0.35)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


def main(args):

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    model = RegressionBlock()

    dataset = XRayEmbeddingDataset(
        args.annotations_file,
        args.img_dir
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True
    )

    dataset_val = XRayEmbeddingDataset(
        args.val_files,
        args.img_dir
    )

    data_loader_val = DataLoader(
        dataset_val,
        batch_size=args.batch,
        shuffle=False
    )

    # Define the loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-4
    )

    # Train the model
    model.to(device)

    bar = tqdm.tqdm(range(args.epochs))
    for epoch in bar:
        running_loss = 0.0
        for i, (images, ages) in enumerate(data_loader):
            images = images.to(device).float()
            ages = ages.to(device).float()[..., None]

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, ages)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 0:
                bar.set_description(f"E: {epoch} I:{i} L:{running_loss/(i+1)}")
            del loss

        epoch_loss = running_loss / len(data_loader)
        print(f"Epoch [{epoch+1}], Loss: {epoch_loss:.4f}")
        scheduler.step()
        model.eval()

        with torch.no_grad():
            val_loss = 0
            for i, (images, ages) in enumerate(data_loader_val, 0):
                images = images.to(device).float()
                ages = ages.to(device).float()[..., None]
                outputs = model(images)
                loss = criterion(outputs, ages)
                val_loss += loss.item()
                del loss

            mean_loss = val_loss/(len(data_loader_val))
            print("Validation loss: ", val_loss/(len(data_loader_val)))
            checkpoint_path = opath.join(
                args.weight, f"simple_swin_reg_e-{epoch+1}-l-{mean_loss}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    checkpoint_path = opath.join(args.weight, "simple_swin_reg_final.pth")
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-b", "--batch",
                   help="Tamanho do batch", type=int,
                   default=16)
    p.add_argument("-e", "--epochs",
                   help="Número de épocas de treinamento", type=int,
                   default=100)
    p.add_argument("-l", "--lr",
                   help="Taxa de aprendizado", type=float,
                   default=1e-3)
    p.add_argument("-f", "--annotations_file",
                   help="Arquivo com as anotações de treinamento", type=str,
                   default="/media/joao/nih-spr/train.csv")
    p.add_argument("-v", "--val_files",
                   help="Arquivo com as anotações de validação", type=str,
                   default="/media/joao/nih-spr/val.csv")
    p.add_argument("-d", "--img_dir",
                   help="Diretório com os embeddings", type=str,
                   default="/media/joao/nih-spr/embeddings/swin")
    p.add_argument("-w", "--weight",
                   help="Diretório com pesos para treinamento", type=str,
                   default="/media/joao/nih-spr/weights")
    p.add_argument("-D", "--device",
                   help="Dispositivo a ser utilizado [cuda ou cpu]", type=str,
                   default=None)
    args = p.parse_args()
    main(args)
