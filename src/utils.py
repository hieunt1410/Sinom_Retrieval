from src.dataset import *
from torchvision import transforms
import torch
import os
import configs.config as cfg

def get_data(args, split='train'):
    data = ImgDataset(args.data_path, split, transform=transforms.Compose([
                        transforms.Resize((512, 512)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]))
    return data


def save_model(model, name, device):
    if not os.path.exists('pretrained'):
        os.makedirs('pretrained')
        
    if name == 'encoder':
        torch.save(model.state_dict(), cfg.ENCODER_MODEL_PATH)
    else:
        torch.save(model.state_dict(), cfg.DECODER_MODEL_PATH)


def load_model(model, name, device):
    if name == 'encoder':
        model.load_state_dict(torch.load(cfg.ENCODER_MODEL_PATH, map_location=device))
    else:
        model.load_state_dict(torch.load(cfg.DECODER_MODEL_PATH, map_location=device))
        
    return model