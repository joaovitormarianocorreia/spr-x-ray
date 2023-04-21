import argparse
import tqdm
import os.path as opath
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from loader import XRayPairDataset
from models.twin import TwinNet
from models.xdensenet import XDenseNet
import torchvision
import torchxrayvision as xrv
import numpy as np


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = torchvision.transforms.Compose([xrv.datasets.XRayResizer(224)])

    def my_transform(image):
        img = np.array(image)
        img = xrv.datasets.normalize(img, 255)
        img = img.mean(2)[None, ...]
        img = transform(img)
        img = torch.from_numpy(img)

        return {"image": img}

    dataset = XRayPairDataset(
        annotations_file=args.annotations_file,
        img_dir=args.img_dir,
        nr_pairs=args.nr_pairs,
        transform=my_transform,
        gray=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True
    )

    dataset_val = XRayPairDataset(
        annotations_file=args.val_files,
        img_dir=args.img_dir,
        nr_pairs=args.nr_pairs,
        transform=my_transform,
        gray=False,
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch,
        shuffle=True
    )

    # cria backbone
    backbone = XDenseNet()
    # cria rede twin
    model = TwinNet(
        backbone,
        in_features=64,
        out_features=16,
        output_dim=1,
        p_drop=0.25
    )
    model.to(device)

    checkpoint_path = opath.join(args.weight, "top_xresnet_reg_final.pth")

    if opath.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint)

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
        amsgrad=True
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=0
    )

    curr_lr = args.lr
    bar = tqdm.tqdm(range(args.epochs))
    for epoch in bar:
        running_loss = 0.0
        model.train()
        for i, data in enumerate(dataloader, 0):
            X1, X2, Y = data
            X1 = X1.float()
            X2 = X2.float()
            X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)

            optimizer.zero_grad()

            outputs = model(X1, X2)
            loss = criterion(outputs, Y.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                bar.set_description(
                    "[%d, %5d] loss: %.3f LR: %.5f"
                    % (epoch + 1, i + 1, running_loss / 100, curr_lr)
                )
                running_loss = 0.0
            del loss

        scheduler.step()
        curr_lr = optimizer.state_dict()["param_groups"][0]["lr"]

        model.eval()

        with torch.no_grad():
            val_loss = 0
            for i, data in enumerate(dataloader_val, 0):
                Xt1, Xt2, Yt = data
                Xt1 = Xt1.float()
                Xt2 = Xt2.float()
                Xt1, Xt2, Yt = Xt1.to(device), Xt2.to(device), Yt.to(device)

                outputs = model(Xt1, Xt2)
                loss = criterion(outputs, Yt.float())
                val_loss += loss.item()
                del loss

            mean_loss = val_loss / (len(dataloader_val))
            print("Validation loss: ", val_loss / (len(dataloader_val)))
            checkpoint_path = opath.join(
                args.weight, f"top_xresnet_reg_e-{epoch+1}-l-{mean_loss}.pth"
            )
            torch.save(model.state_dict(), checkpoint_path)

    checkpoint_path = opath.join(args.weight, f"top_xresnet_reg_final.pth")
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-b",
        "--batch",
        help="Tamanho do Batch",
        type=int,
        default=8
    )
    p.add_argument(
        "-e",
        "--epochs",
        help="Épocas",
        type=int,
        default=10
    )
    p.add_argument(
        "-l",
        "--lr",
        help="Learning rate",
        type=float,
        default=1e-3
    )
    p.add_argument(
        "-f",
        "--annotations_file",
        help="Arquivo com as anotações",
        type=str,
        default="/media/joao/nih-spr/train.csv",
    )
    p.add_argument(
        "-v",
        "--val_files",
        help="Arquivo com as anotações",
        type=str,
        default="/media/joao/nih-spr/val.csv",
    )
    p.add_argument(
        "-d",
        "--img_dir",
        help="Diretório das imagens",
        type=str,
        default="/media/joao/nih-spr/images",
    )
    p.add_argument(
        "-w",
        "--weight",
        help="Weight",
        type=str,
        default="/media/joao/nih-spr/weights/xdensenet",
    )
    p.add_argument(
        "-n",
        "--nr_pairs",
        help="pares por amostra",
        type=int,
        default=10
    )
    args = p.parse_args()
    main(args)
