import torch
import argparse
import pandas as pd
import os.path as opath
import albumentations as A

from models.twin import TwinNet
from models.resnet import ResNet
from loader import XRaySingleDataset
from torch.utils.data import DataLoader
from albumentations.pytorch import ToTensorV2


def main(args):

    # clean the cache memory before training
    torch.cuda.empty_cache()

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    val_transform = A.Compose([
        ToTensorV2(),
    ])

    dataset = XRaySingleDataset(
        annotations_file=args.annotations_file,
        img_dir=args.img_dir,
        transform=val_transform,
        gray=False,
        nr_pairs=20
    )

    dataset_val = XRaySingleDataset(
        annotations_file=args.val_files,
        img_dir=args.test_dir,
        transform=val_transform,
        gray=False,
        nr_pairs=20
    )

    dataloader_val = DataLoader(
        dataset_val,
        batch_size=1,
        shuffle=False
    )

    backbone = ResNet()

    model = TwinNet(
        backbone,
        in_features=64,
        out_features=8,
        output_dim=1,
        p_drop=0.5
    )

    model = model.to(device)

    checkpoint_path = opath.join(args.weights_dir, "resnet_epochs-50_batch-32_lr-0.001_pairs-8.pth")

    if opath.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        print(checkpoint.keys())
        model.load_state_dict(checkpoint)

    model.eval()

    ypred = []

    with torch.no_grad():
        for i, data in enumerate(dataloader_val, 0):
            Xt1, Yt = data
            Xt1 = Xt1.float()/255.0
            Xt1 = Xt1.to(device)
            mean, std = dataset.eval(model, Xt1)
            age = round(mean*100)
            ypred.append([i, age])

            print("Predicted age: ", age)

    df = pd.DataFrame(ypred)
    df.to_csv('result_resnet.csv')


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
        default="/home/joao/Data/spr/age/test.csv"
    )
    p.add_argument(
        "-d",
        "--img_dir",
        help="Diretório das imagens",
        type=str,
        default="/home/joao/Data/spr/age/images224"
    )
    p.add_argument(
        "-k",
        "--test_dir",
        help="Diretório das imagens",
        type=str,
        default="/home/joao/Data/spr/age/test224"
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
