import sys
import importlib
import models
import torch
import datasets as datasets
import numpy as np
import global_data
from config import config
from tqdm import tqdm
import argparse
import torch_utils


def main(part, ckpt_file, output_file, device, event_type):
    global_data.init()
    backbone = models.OTTORecallModel(config)

    if part == 'article':
        dataset = datasets.OTTOPredictArticleEmbeddingDataset()
    else:
        dataset = datasets.OTTOPredictSessionEmbeddingDataset(max_session_size=config['max_session_size'])
        pass

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_data_workers'],
        collate_fn=datasets.collate_fn)
    
    ckpt = torch.load(ckpt_file, map_location='cpu')
    backbone.load_state_dict(ckpt)

    backbone = backbone.to(device)
    backbone.eval()
    embs = []
    for batch in tqdm(loader):
        batch = torch_utils.batch_to_device(batch, device)
        
        with torch.no_grad():
            if part == 'article':
                data_event_type = torch.ones(batch['feat_article'].shape[0], dtype=torch.long, device=device) * event_type
                emb = backbone.forward_article(batch['feat_article'], data_event_type).squeeze(1)
            else:
                data_event_type = torch.ones(batch['feat_session'].shape[0], dtype=torch.long, device=device) * event_type
                emb = backbone.forward_session(batch['feat_session'], batch['session_mask'], data_event_type)
                pass
            pass
        embs.append(emb.detach().cpu().numpy())
        pass

    embs = np.vstack(embs)

    np.save(output_file, embs)

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['session', 'article'], required=True)
    parser.add_argument('--ckpt_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--event_type', type=int, required=True)
    args = parser.parse_args()
    main(args.part, args.ckpt_file, args.output_file, args.device, args.event_type)
    pass