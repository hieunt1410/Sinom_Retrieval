import torch
import argparse
from torch.utils.data import DataLoader
from src.utils import *
from src import train
import numpy as np
import os

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='CNN', help='name of the model to use.')
parser.add_argument('--data_path', type=str, default='data', help='path for storing the dataset')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='batch size')
parser.add_argument('--clip', type=float, default=0.8, help='gradient clip value (default: 0.8)')
parser.add_argument('--lr', type=float, default=2e-5, help='initial learning rate (default: 2e-5)')
parser.add_argument('--optim', type=str, default='AdamW', help='optimizer to use (default: AdamW)')
parser.add_argument('--num_epochs', type=int, default=3, help='number of epochs (default: 3)')
parser.add_argument('--log_interval', type=int, default=5, help='frequency of result logging (default: 100)')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--when', type=int, default=2, help='when to decay learning rate (default: 2)')

args = parser.parse_args()
output_dim = 252

criterion = 'CrossEntropyLoss'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("Start loading the data....")
train_data = get_data(args, 'train')
valid_data = get_data(args, 'valid')
test_data = get_data(args, 'test')

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
print('Finish loading the data....')

hyp_params = args
hyp_params.device = device
hyp_params.n_train, hyp_params.n_valid, hyp_params.n_test = len(train_data), len(valid_data), len(test_data)
    
hyp_params.model = args.model.strip()
hyp_params.output_dim = output_dim
hyp_params.criterion = criterion

if __name__ == '__main__':
    test_loss = train.initiate(hyp_params, train_loader, valid_loader, test_loader)