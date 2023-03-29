import tqdm
import torch
import argparse
import torch.nn as nn
import os.path as opath
import albumentations as A
import torch.optim as optim

from models.twin import TwinNet
from models.resnet import ResNet
from loader import XRayPairDataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2


def main(args):
    # clean the cache memory before training
    torch.cuda.empty_cache()

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose([
        A.ShiftScaleRotate(rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.RandomCrop(height=200, width=200, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Resize(224, 224),
        ToTensorV2(),
    ])

    val_transform = A.Compose([
        ToTensorV2(),
    ])

    dataset = XRayPairDataset(
        annotations_file=args.annotations_file,
        img_dir=args.img_dir,
        transform=train_transform,
        gray=False,
        nr_pairs=args.nr_pairs
    )

    print(len(dataset))

    dataset_val = XRayPairDataset(
        annotations_file=args.val_files,
        img_dir=args.img_dir,
        transform=val_transform,
        gray=False,
        nr_pairs=args.nr_pairs
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=args.batch,
        shuffle=True
    )

    backbone = ResNet()

    model = TwinNet(
        backbone,
        in_features=64,
        out_features=8,
        output_dim=1,
        p_drop=0.5
    )

    model.to(device)

    checkpoint_path = opath.join(args.weights_dir, "resnet.pth")

    if opath.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    bar = tqdm.tqdm(range(args.epochs))
    for epoch in bar:
        running_loss = 0.0
        model.train()
        for i, data in enumerate(dataloader, 0):
            X1, X2, Y = data
            X1 = X1.float()/255.0
            X2 = X2.float()/255.0
            X1, X2, Y = X1.to(device), X2.to(device), Y.to(device)

            optimizer.zero_grad()

            outputs = model(X1, X2)

            loss = criterion(outputs, Y.float())
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            if i % 100 == 99:
                bar.set_description('[%d, %5d] loss: %.3f' %
                                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
            del loss

        model.eval()

        with torch.no_grad():
            val_loss = 0
            for i, data in enumerate(dataloader_val, 0):
                Xt1, Xt2, Yt = data
                Xt1 = Xt1.float()/255.0
                Xt2 = Xt2.float()/255.0
                Xt1, Xt2, Yt = Xt1.to(device), Xt2.to(device), Yt.to(device)

                outputs = model(Xt1, Xt2)
                loss = criterion(outputs, Yt.float())
                val_loss += loss.item()
                del loss

            print("Validation loss: ", val_loss/(len(dataloader_val)))
            checkpoint_path = opath.join(
                args.weights_dir, f"mobilenet_reg_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-b",
        "--batch",
        help="Tamanho do Batch",
        type=int,
        default=1
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
        default="/home/joao/Data/spr/age/xray_age_train.csv"
    )
    p.add_argument(
        "-v",
        "--val_files",
        help="Arquivo com as anotações",
        type=str,
        default="/home/joao/Data/spr/age/xray_age_val.csv"
    )
    p.add_argument(
        "-d",
        "--img_dir",
        help="Diretório das imagens",
        type=str,
        default="/home/joao/Data/spr/age/images224"
    )
    p.add_argument(
        "-w",
        "--weights_dir",
        help="Diretório dos pesos",
        type=str,
        default="/home/joao/Data/spr/weights"
    )
    p.add_argument(
        "-n",
        "--nr_pairs",
        help="pares por amostra",
        type=int,
        default=1
    )
    args = p.parse_args()
    main(args)
