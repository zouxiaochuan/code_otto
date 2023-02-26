
import torch
import numpy as np
from config import config
import models
import datasets2 as datasets
from tqdm import tqdm
import torch_utils



def predict_session_embedding(session_splits, ckpt_file, output_file, device, event_type):
    backbone = models.OTTORecallModel(config)

    dataset = datasets.OTTOPredictSessionEmbeddingDataset(session_splits, max_session_size=config['max_session_size'])

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
            data_event_type = torch.ones(batch['feat_session'].shape[0], dtype=torch.long, device=device) * event_type
            emb = backbone.forward_session(batch['feat_session'], batch['session_mask'], data_event_type)
            pass
        embs.append(emb.detach().cpu().numpy())
        pass

    embs = np.vstack(embs)

    np.save(output_file, embs)

    pass

    pass