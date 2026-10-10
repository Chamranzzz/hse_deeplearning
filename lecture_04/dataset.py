import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.v2 as tfs
import torch
import os
import json
from PIL import Image
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.transforms.v2 as v2

class SunDataset(data.Dataset):
    def __init__(self, path, train=True, transform=None):
        self.path = os.path.join(path, "train" if train else "test")
        self.transform = transform

        with open(os.path.join(self.path, "format.json"), "r") as fp:
            self.format = json.load(fp)

        self.length = len(self.format)
        self.files = tuple(self.format.keys())
        self.targets = tuple(self.format.values())


    def __getitem__(self, item):
        file_path = os.path.join(self.path, self.files[item])
        img = Image.open(file_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(self.targets[item], dtype=torch.float32)

    def __len__(self):
        return self.length

transforms = tfs.Compose([tfs.ToImage(), tfs.ToDtype(torch.float32, scale=True)])
d_train = SunDataset('dataset_reg', transform=transforms)
train_data = data.DataLoader(d_train, batch_size=32, shuffle=True)
