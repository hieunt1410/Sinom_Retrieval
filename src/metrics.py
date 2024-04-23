import torch
import numpy as np
from torchmetrics.functional.retrieval import retrieval_reciprocal_rank
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# mrr = RetrievalMRR(top_k=5)
# mrr.to(device)

def metrics(results, truths):
    all_reciprocal_ranks = [retrieval_reciprocal_rank(results[i], truths[i]) for i in range(len(truths))]
    return torch.mean(torch.stack(all_reciprocal_ranks))
