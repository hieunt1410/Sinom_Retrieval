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
        with open(os.path.join(path, f'{split}.csv'), 'r') as f:
            line = f.readline()
            self.targets = line.strip().split(',')
            self.images = [os.path.join(path, f'pairs/print/{i}.png') for i in targets]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.path, 'pairs/print', self.images[idx])
        
        img = Image.open(img_path).convert('RGB')
        target = [] * len(self.targets)
        target[self.targets[idx]] = 1
        
        if self.transform:
            img = self.transform(img)
        
        return {
            'image': img,
            'label': torch.tensor(target)
        }