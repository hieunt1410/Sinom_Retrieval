import os
import torch
import json
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset

class ImgDataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform
        
        images = [i for i in os.listdir(os.path.join(path, 'pairs/print'))]
        targets = [i for i in range(len(os.listdir(os.path.join(path, 'pairs/stl'))))]

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