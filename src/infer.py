import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
import argparse
from models import CNN

parser = argparse.ArgumentParser()
parser.add_argument('--img_path', type=str, default='data/queries/228.png')
parser.add_argument('--model_path', type=str, default='pretrained_models/default_model.pt')
args = parser.parse_args()

transform=transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

model = CNN(252)

def infer(img_path, model_path='pretrained_models/default_model.pt'):
    model.load_state_dict(torch.load(model_path))
    img = Image.open(img_path).convert('RGB')
    img = transform(img)
    output = model(img)
    _, cols = torch.topk(output, 5)
    return cols
    

if __name__ == '__main__':
    print(infer(args.img_path, args.model_path))