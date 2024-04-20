import torch
import numpy as np
from torchmetrics.retrieval import RetrievalMRR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mrr = RetrievalMRR()
mrr.to(device)

def metrics(results, truths):
    return mrr(torch.tensor(results), torch.tensor(truths), top_k=5)
