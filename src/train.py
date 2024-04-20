from src import models
from src.utils import *
import numpy as np
import time
import sys

from transformers import BertModel
from transformers import BertTokenizer

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch import nn
from tqdm import tqdm

# from src.metrics import *

import os

def initiate(hyp_params, train_loader, valid_loader, test_loader=None):
    feature_extractor = torch.hub.load('pytorch/vision:v0.10.0', hyp_params.cnn_model, pretrained=True)
    for param in feature_extractor.features.parameters():
        param.requires_grad = False

    hyp_params.feature_extractor = feature_extractor
    model = getattr(models, hyp_params.model)(hyp_params)
    model.to(hyp_params.device)
    
    optimizer = getattr(optim, hyp_params.optim)(model.parameters(), lr=hyp_params.lr, weight_decay=1e-4)
    criterion = getattr(nn, hyp_params.criterion)()
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    
    
    settings = {'model': model,
                'optimizer': optimizer,
                'criterion': criterion,
                'scheduler': scheduler}

    return train_model(settings, hyp_params, train_loader, valid_loader, test_loader)

####################################################################
#
# Training and evaluation scripts
#
####################################################################

def train_model(settings, hyp_params, train_loader, valid_loader, test_loader):
    model = settings['model']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    
    def train(model, bert, tokenizer, feature_extractor, optimizer, criterion):
        epoch_loss = 0
        model.train()
        num_batches = hyp_params.n_train // hyp_params.batch_size
        proc_loss, proc_size = 0, 0
        total_loss = 0.0
        losses = []
        results = []
        truths = []

        for data_batch in tqdm(train_loader):
            targets = data_batch["label"]
            images = data_batch['image']
            
            targets = targets.to(hyp_params.device)
            images = images.to(hyp_params.device)

            optimizer.zero_grad()

            outputs = model(images)
            preds = outputs
            
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), hyp_params.clip)
            optimizer.step()
            
            total_loss += loss.item() * hyp_params.batch_size
            results.append(preds)
            truths.append(targets)
                
        avg_loss = total_loss / hyp_params.n_train
        results = torch.cat(results)
        truths = torch.cat(truths)
        return results, truths, avg_loss

    def evaluate(model, bert, tokenizer, feature_extractor, criterion, test=False):
        model.eval()
        loader = test_loader if test else valid_loader
        total_loss = 0.0

        results = []
        truths = []
        correct_predictions = 0

        with torch.no_grad():
            for data_batch in loader:
                targets = data_batch["label"]
                images = data_batch['image']
                
                # input_ids = input_ids.to(hyp_params.device)
                targets = targets.to(hyp_params.device)
                images = images.to(hyp_params.device)            

                outputs = model(images)
                preds = outputs
                
                total_loss += criterion(outputs, targets).item() * hyp_params.batch_size
                correct_predictions += torch.sum(preds == targets)

                # Collect the results into dictionary
                results.append(preds)
                truths.append(targets)

        avg_loss = total_loss / (hyp_params.n_test if test else hyp_params.n_valid)

        results = torch.cat(results)
        truths = torch.cat(truths)
        return results, truths, avg_loss

    best_valid = 1e8
    # writer = SummaryWriter('runs/'+hyp_params.model)
    for epoch in range(1, hyp_params.num_epochs+1):
        
        train_results, train_truths, train_loss = train(model, bert, tokenizer, feature_extractor, optimizer, criterion)
        val_results, val_truths, val_loss = evaluate(model, bert, tokenizer, feature_extractor, criterion, test=False)
        
        scheduler.step(val_loss)

        train_mrr= metrics(train_results, train_truths)
        val_mrr = metrics(val_results, val_truths)
        
        if epoch == 1:
            print(f'Epoch  |     Train Loss     |     Train MRR     |     Valid Loss     |     Valid MRR     |')
        
        print(f'{epoch:^7d}|{train_loss:^20.4f}|{train_mrr:^24.4f}|{val_loss:^20.4f}|{val_mrr:^24.4f}|')

        if val_loss < best_valid:
            print(f"Saved model at pretrained_models/{hyp_params.name}.pt!")
            save_model(hyp_params, model, name=hyp_params.name)
            best_valid = val_loss

    if test_loader is not None:
        model = load_model(hyp_params, name=hyp_params.name)
        results, truths, val_loss = evaluate(model, bert, tokenizer, feature_extractor, criterion, test=True)
        test_mrr = metrics(results, truths)
        
        print("\n\nTest MRR {:5.4f}".format(test_mrr))

    sys.stdout.flush()