from src.dataset import *
from torchvision import transforms
import torch
import os


def get_data(args, split='train'):
    data = ImgDataset(args.data_path, split, transform=transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]))
    return data


def save_model(args, model, name=''):
    name = name if len(name) > 0 else 'default_model'
    torch.save(model, f'pretrained_models/{name}.pt')


def load_model(args, name=''):
    name = name if len(name) > 0 else 'default_model'
    model = torch.load(f'pretrained_models/{name}.pt')
    return model