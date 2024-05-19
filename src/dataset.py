import os
import torch
import json
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data.dataset import Dataset

class ImgDataset(Dataset):
    def __init__(self, path, split, transform=None):
        self.path = path
        self.transform = transform
        
        querylist = os.listdir(os.path.join(path, 'queries'))
        targetlist = os.listdir(os.path.join(path, 'database_2D'))
        querylist = sorted(querylist)
        targetlist = sorted(targetlist)
        
        if split == 'train':
            self.query = querylist
            self.target = targetlist
        elif split == 'valid':
            self.query = querylist[int(0.8*len(querylist)):int(0.9*len(querylist))]
            self.target = targetlist[int(0.8*len(targetlist)):int(0.9*len(targetlist))]
        elif split == 'test':
            self.query = querylist[int(0.9*len(querylist)):]
            self.target = targetlist[int(0.9*len(targetlist)):]

    def __len__(self):
        return len(self.query)
    
    def __getitem__(self, idx):
        img_query_loc = os.path.join(self.path, 'queries', self.query[idx])
        img_target_loc = os.path.join(self.path, 'database_2D', self.target[idx])
        
        img_query = Image.open(img_query_loc).convert('RGB')
        img_target = Image.open(img_target_loc).convert('RGB')
        
        if self.transform is not None:
            img_query = self.transform(img_query)
            img_target = self.transform(img_target)
            
        return img_query, img_target