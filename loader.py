import os
import torch
import pandas as pd
import numpy as np

from PIL import Image
from torchvision.io import read_image


class XRayDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            str(self.img_dir),
            str(self.img_labels.iloc[idx, 0]).zfill(6)+'.png'
        )
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label


class XRayPairDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, gray=True, nr_pairs=1):
        self.gray = gray
        self.img_labels = pd.read_csv(annotations_file)

        self.img_dir = img_dir
        self.transform = transform

        images = []
        for idx, imgid in enumerate(self.img_labels["imageId"]):
            img_path = os.path.join(
                str(self.img_dir),
                str(self.img_labels.iloc[idx, 0]).zfill(6)+'.png'
            )
            images.append(img_path)

        images = np.array(images, dtype=str)
        labels = np.array(self.img_labels["age"].tolist(), dtype=float)

        self.images = images
        self.labels = images

        self.X = []
        self.Y = []

        for x1, y1 in zip(images, labels):
            idx = np.random.randint(images.shape[0], size=nr_pairs)
            X2 = images[idx]
            Y2 = labels[idx]
            for x2, y2 in zip(X2, Y2):
                self.X.append([x1, x2])
                self.Y.append([y1 - y2])
                self.X.append([x2, x1])
                self.Y.append([y2 - y1])
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

        self.Y = np.array(self.Y)

        self.Y = self.Y / 100.0

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        img_path1 = self.X[idx, 0]
        img_path2 = self.X[idx, 1]

        if self.gray:
            image1 = Image.open(img_path1).convert("L")
            image2 = Image.open(img_path2).convert("L")
        else:
            image1 = Image.open(img_path1).convert("RGB")
            image2 = Image.open(img_path2).convert("RGB")

        label = self.Y[idx]
        if self.transform:
            image1 = self.transform(image=np.array(image1))["image"]
            image2 = self.transform(image=np.array(image2))["image"]
        return image1, image2, label


class XRaySingleDataset(torch.utils.data.Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, gray=True, nr_pairs=1):
        self.gray = gray
        self.nr_pairs = nr_pairs
        self.img_labels = pd.read_csv(annotations_file)

        self.img_dir = img_dir
        self.transform = transform

        images = []
        for idx, imgid in enumerate(self.img_labels["imageId"]):
            img_path = os.path.join(
                str(self.img_dir),
                str(self.img_labels.iloc[idx, 0]).zfill(6)+'.png'
            )
            images.append(img_path)

        images = np.array(images, dtype=str)
        labels = np.array(self.img_labels["age"].tolist(), dtype=float)

        self.images = images
        self.labels = images

        self.X = []
        self.Y = []

        # create the pairs
        for x1, y1 in zip(images, labels):
            self.X.append(x1)
            self.Y.append([y1])

        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

        self.Y = self.Y / 100.0

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        img_path1 = self.X[idx]
        if self.gray:
            image1 = Image.open(img_path1).convert("L")

        else:
            image1 = Image.open(img_path1).convert("RGB")

        label = self.Y[idx]
        if self.transform:
            image1 = self.transform(image=np.array(image1))["image"]
        return image1, label

    def eval(self, model, x):

        indices = np.random.randint(self.images.shape[0], size=self.nr_pairs)
        Yout = []
        for idx in indices:
            xtrain, ytrain = self[idx]
            xtrain = xtrain.float()/255.0
            xtrain = xtrain[None, ...]
            y = 0.5*model(x, xtrain) - 0.5*model(xtrain, x)+ytrain
            Yout.append(y.numpy())

        return np.mean(np.squeeze(Yout)), np.std(np.squeeze(Yout))
