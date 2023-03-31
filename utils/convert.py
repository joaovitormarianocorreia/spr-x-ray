import glob
import tqdm
import argparse
from PIL import Image
import os.path as opath


def main(args):
    src = args.src
    src_files = glob.glob(opath.join(src, "*.png"))
    bar = tqdm.tqdm(src_files)
    for file in bar:
        filename = opath.basename(file)
        img = Image.open(file)
        img = img.resize((int(args.size), int(args.size)))
        img.save(opath.join(args.dst, filename))
        bar.set_description(f"F: {filename}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "-s",
        "--src",
        help="Source",
        type=str,
        default="/home/joao/Data/spr/age/test"
    )
    p.add_argument(
        "-d",
        "--dst",
        help="Diretório Destino",
        type=str,
        default="/home/joao/Data/spr/age/test224"
    )
    p.add_argument(
        "-z",
        "--size",
        help="Tamanho",
        type=int,
        default=224
    )
    args = p.parse_args()
    main(args)
