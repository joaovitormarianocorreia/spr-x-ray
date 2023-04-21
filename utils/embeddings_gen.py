import argparse
import tqdm
import os.path as opath
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize, Compose
from loader import XRayDataset
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config


def main(args):

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = args.device

    model = timm.create_model("swin_base_patch4_window12_384",
                              pretrained=True,
                              num_classes=0)

    config = resolve_data_config({}, model=model)

    transform_m = create_transform(**config)

    transform = Compose([
        Resize((384, 384)),
        transform_m,
    ])

    dataset = XRayDataset(args.annotations_file,
                          args.img_dir,
                          transform=transform,
                          return_filename=True)

    data_loader = DataLoader(dataset,
                             batch_size=1,
                             shuffle=False)

    model = model.to(device)

    model.eval()
    with torch.no_grad():
        for i, (images, ages, filename) in tqdm.tqdm(enumerate(data_loader, 0)):
            images = images.to(device).float()
            ages = ages.to(device).float()[..., None]
            outputs = model(images)
            # print(filename, outputs.to('cpu')[0])
            file = opath.join(args.dst, f"{filename[0]}.pt")
            torch.save(outputs.to('cpu')[0], file)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-f", "--annotations_file",
                   help="Arquivo com as anotações", type=str,
                   default="/home/joao/data/spr/age/sample_submission_age.csv")
    p.add_argument("-p", "--img_dir",
                   help="path ", type=str,
                   default="/home/joao/data/spr/age/test")
    p.add_argument("-d", "--dst",
                   help="Destination", type=str,
                   default="/home/joao/data/spr/age/test_embeddings")
    p.add_argument("-D", "--device",
                   help="device", type=str,
                   default=None)
    args = p.parse_args()
    main(args)
