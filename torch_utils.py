import torch.nn as nn
import torch
from tqdm import tqdm
import numpy as np


class CategoryFeatureEmbedding(nn.Module):
    def __init__(self, num_uniq_values, embed_dim):
        '''
        '''
        super().__init__()
        num_uniq_values = torch.LongTensor(num_uniq_values)
        csum = torch.cumsum(num_uniq_values, dim=0)
        num_emb = csum[-1]
        num_uniq_values = torch.LongTensor(num_uniq_values).reshape(1, 1, -1)
        self.register_buffer('num_uniq_values', num_uniq_values)
        
        starts = torch.cat(
            (torch.LongTensor([0]), csum[:-1])).reshape(1, -1)
        self.register_buffer('starts', starts)
        
        self.embeddings = nn.Embedding(
            num_emb, embed_dim, sparse=True)
        self.embed_dim = embed_dim

        self.layer_norm_output = nn.LayerNorm(embed_dim)
        pass

    def forward(self, x):
        # x = x + 1
        if torch.any(x < 0):
            raise RuntimeError(str(x))
        
        if torch.any(torch.ge(x, self.num_uniq_values)):
            raise RuntimeError(str(x))
            pass
        
        x = x + self.starts
        x = self.embeddings(x).sum(dim=-2)
        return self.layer_norm_output(x)
        # return x
    pass


def batch_to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
        pass
    elif isinstance(batch, dict):
        b = dict()
        for k, v in batch.items():
            b[k] = batch_to_device(v, device)
            pass
        return b
        pass
    elif isinstance(batch, list) or isinstance(batch, tuple):
        return [batch_to_device(x, device) for x in batch]
        pass
    else:
        return batch
    pass

def get_optimizer_params(
        named_parameters, learning_rate, weight_decay, no_decay_keys=['bias', 'layer_norm'], must_include_keys=None, must_exclude_keys=None):
    
    named_parameters = list(named_parameters)
    
    if must_include_keys is not None:
        named_parameters = [(n, p) for n, p in named_parameters if any(fk in n for fk in must_include_keys)]
        pass

    if must_exclude_keys is not None:
        named_parameters = [(n, p) for n, p in named_parameters if not any(fk in n for fk in must_exclude_keys)]
        pass

    optimizer_parameters = [
        {'params': [p for n, p in named_parameters if any(nd in n for nd in no_decay_keys)],
            'lr': learning_rate, 'weight_decay': 0.0 },
        {'params': [p for n, p in named_parameters if not any(nd in n for nd in no_decay_keys)],
            'lr': learning_rate, 'weight_decay': weight_decay }
    ]
    return optimizer_parameters


def chunked_topk(emb1_file, emb2_file, k, device):
    emb1 = np.load(emb1_file)
    emb2 = np.load(emb2_file)

    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    
    chunk_size = 512

    emb2 = torch.from_numpy(emb2).to(device)
    topn = []
    topn_scores = []
    for i in tqdm(range(0, len(emb1), chunk_size)):
        emb1_chunk = emb1[i:i+chunk_size]
        emb1_chunk = torch.from_numpy(emb1_chunk).to(device)
        
        with torch.no_grad():
            scores = torch.matmul(emb1_chunk, emb2.t())
            scores, idx = scores.topk(k, dim=1)
            pass

        idx = idx.cpu().numpy()
        topn.append(idx)
        scores = scores.cpu().numpy()
        topn_scores.append(scores)
        pass

    topn = np.vstack(topn)
    topn_scores = np.vstack(topn_scores)

    return topn, topn_scores