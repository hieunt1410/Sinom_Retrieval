import torch
import numpy as np
from torchmetrics.retrieval import RetrievalMRR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mrr = RetrievalMRR(top_k=5)
mrr.to(device)

def metrics(results, truths):
    # results, indices = torch.topk(results, 5)
    indexes = torch.ones((truths.shape[0], truths.shape[1]), dtype=torch.long, device=device)
    return mrr(torch.tensor(results), torch.tensor(truths), indexes=indexes)
