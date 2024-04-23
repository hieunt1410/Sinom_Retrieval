import torch
import numpy as np
from torchmetrics.functional.retrieval import retrieval_reciprocal_rank
from torchmetrics.retrieval import RetrievalMRR
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mrr = RetrievalMRR(top_k=5)
mrr.to(device)

def metrics(results, truths):
    # all_reciprocal_ranks = [retrieval_reciprocal_rank(results[i], truths[i]) for i in range(len(truths))]
    indexes = (torch.zeros_like(results[0]) + torch.arange(results.shape[0], device=device, dtype=torch.long).view(-1, 1)).flatten()
    # return torch.mean(torch.stack(all_reciprocal_ranks))
    return mrr(results.flatten(), truths.flatten(), indexes)
