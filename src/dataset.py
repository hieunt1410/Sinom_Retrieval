import os
import torch
import json
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset

class ImgDataset(Dataset):
    def __init__(self, path, split, transform=None):
        self.transform = transform
        
        self.targets = []
        self.images = [os.path.join(path, f'pairs/print/{i}.png') for i in self.targets]
        with open(os.path.join(path, f'{split}.csv'), 'r') as f:
            line = f.readlines()
            for i in line:
                self.targets.append(int(i.strip()))
                self.images.append(os.path.join(path, f'pairs/print/{i.strip()}.png'))

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        
        img = Image.open(img_path)
        target = [0] * 252
        target[self.targets[idx]] = 1
        
        if self.transform:
            img = self.transform(img)
        
        return {
            'image': img,
            'label': torch.tensor(target)
        }