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

from src.metrics import *
import configs.config as cf

import os

def initiate(hyp_params, train_loader, valid_loader, test_loader=None):
    encoder = getattr(models, 'ConvEncoder')()
    encoder.to(hyp_params.device)
    
    decoder = getattr(models, 'ConvDecoder')()
    decoder.to(hyp_params.device)
    
    autoencoder_params = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = getattr(optim, hyp_params.optim)(autoencoder_params, lr=hyp_params.lr, weight_decay=1e-4)
    criterion = getattr(nn, hyp_params.criterion)()
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=hyp_params.when, factor=0.1, verbose=True)
    
    settings = {'encoder': encoder,
                'decoder': decoder,
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
    encoder = settings['encoder']
    decoder = settings['decoder']
    optimizer = settings['optimizer']
    criterion = settings['criterion']
    scheduler = settings['scheduler']
    
    def train(encoder, decoder, optimizer, criterion):
        encoder.train()
        decoder.train()
            
        for batch_idx, (query_img, target_img) in enumerate(train_loader):
            query_img = query_img.to(hyp_params.device)
            target_img = target_img.to(hyp_params.device)
            
            optimizer.zero_grad()
            
            enc_output = encoder(query_img)
            dec_output = decoder(enc_output)
            
            loss = criterion(dec_output, target_img)            
            loss.backward()
            
            optimizer.step()
        
        return loss.item()
            

    def evaluate(encoder, decoder, criterion, test=False):
        encoder.eval()
        decoder.eval()
        
        with torch.no_grad():
            
            for batch_idx, (query_img, target_img) in enumerate(valid_loader if not test else test_loader):
                query_img = query_img.to(hyp_params.device)
                target_img = target_img.to(hyp_params.device)
                
                enc_output = encoder(query_img)
                dec_output = decoder(enc_output)
                
                loss = criterion(dec_output, target_img)
        
        return loss.item()

    best_valid = 1e8
    # writer = SummaryWriter('runs/'+hyp_params.model)
    for epoch in range(1, hyp_params.num_epochs+1):
        
        train_loss = train(encoder, decoder, optimizer, criterion)
        val_loss = evaluate(encoder, decoder, criterion, test=False)
        
        scheduler.step(val_loss)

        # train_mrr= metrics(train_results, train_truths)
        # val_mrr = metrics(val_results, val_truths)
        train_mrr = 0
        val_mrr = 0
        
        if epoch == 1:
            print(f'Epoch  |     Train Loss     |     Train MRR     |     Valid Loss     |     Valid MRR     |')
        
        print(f'{epoch:^7d}|{train_loss:^20.4f}|{train_mrr:^19.4f}|{val_loss:^20.4f}|{val_mrr:^19.4f}|')

        if val_loss < best_valid:
            print(f"Saved model at pretrained_models/default_model.pt!")
            save_model(encoder, 'encoder', hyp_params.device)
            save_model(decoder, 'decoder', hyp_params.device)
            
            best_valid = val_loss
            
            
    if test_loader is not None:
        encoder = getattr(models, 'ConvEncoder')()
        encoder = load_model(encoder, 'encoder', hyp_params.device)
        decoder = getattr(models, 'ConvDecoder')()
        decoder = load_model(decoder, 'decoder', hyp_params.device)
        
        val_loss = evaluate(encoder, decoder, criterion, test=True)
        # test_mrr = metrics(results, truths)
        
        print("\n\nTest loss {:5.4f}".format(val_loss))
        

    def create_embedding(encoder, full_loader, embedding_dim):
        encoder.eval()
        
        embedding = torch.randn(embedding_dim)
        
        with torch.no_grad():
            for batch_idx, (query_img, target_img) in enumerate(full_loader):
                query_img = query_img.to(hyp_params.device)
                target_img = target_img.to(hyp_params.device)
                
                enc_output = encoder(query_img)
                
                embedding = torch.cat((embedding, enc_output), dim=0)
                
        return embedding
    
    embedding = create_embedding(
        encoder, train_loader, cfg.EMBEDDING_SHAPE)
    
    numpy_embedding = embedding.cpu().detach().numpy()
    num_images = numpy_embedding.shape[0]
    
    flatten_embedding = numpy_embedding.reshape((num_images, -1))
    np.save(cfg.EMBEDDING_PATH, flatten_embedding)

    sys.stdout.flush()