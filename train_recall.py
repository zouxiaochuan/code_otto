import datasets
import torch.utils.data
from config import config
import models
import torch_utils
import torch.optim
import timm.scheduler
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import global_data
import os
import argparse
import torch


run_id = 'event_type_merge_sparse_margin0.3_embnorm_h256_sigmoidloss_pooltransformer_samearticleemb_cosschedule'


def compute_average_gradients(parameters):
    
    norm = 0
    nnz = 0
    for p in parameters:
        if p.grad is not None:
            norm += p.grad.data.norm(2).item()**2
            if p.grad.is_sparse:
                nnz += p.grad._values().numel()
            else:
                nnz += p.grad.data.numel()
                pass
            pass
        pass
    
    if nnz > 0:
        return norm**0.5 / nnz
    else:
        return 0


def main(device, load_model, load_emb):
    global_data.init()

    model = models.OTTORecallModel(config).to(device)
    if len(load_model) > 0:
        model.load_state_dict(torch.load(load_model, map_location=device))
        pass

    if len(load_emb) > 0:
        omodel = torch.load(load_emb, map_location=device)
        article_weight = omodel['layer_embed_article.embeddings.weight']
        model.layer_embed_article.embeddings.weight.data[:] = article_weight
        model.layer_embed_session.embeddings.weight.data[:article_weight.shape[0], :] = article_weight
        model.layer_embed_session.embeddings.weight.data[article_weight.shape[0]:, :] = omodel['layer_embed_session.embeddings.weight']

        model.layer_encode_session.load_state_dict({k[21:]:v for k,v in omodel.items() if k.startswith('layer_encode_session')})
        pass
    
    dataset_train = datasets.OTTORecallTrainDataset(
        num=5000000, negative_num=config['negative_num'], max_predict_days=config['max_predict_days'],
        max_session_size=config['max_session_size'])

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config['batch_size'],
        num_workers=config['num_data_workers'],
        collate_fn=datasets.collate_fn,
        shuffle=True
    )

    optimizer = torch.optim.AdamW(
        torch_utils.get_optimizer_params(model.named_parameters(), config['learning_rate'], config['weight_decay'], must_exclude_keys=['embeddings']))
    
    # optimizer = torch.optim.AdamW(model.parameters(), config['learning_rate'])
    optimizer_article = torch.optim.SparseAdam(
        torch_utils.get_optimizer_params(model.layer_embed_article.embeddings.named_parameters(), config['learning_rate'], config['weight_decay']))
    
    optimizer_session = torch.optim.SparseAdam(
        torch_utils.get_optimizer_params(model.layer_embed_session.embeddings.named_parameters(), config['learning_rate'], config['weight_decay']))

    # scheduler = timm.scheduler.StepLRScheduler(
    #     optimizer, decay_t=config['learning_rate_decay_epochs'], decay_rate=config['learning_rate_decay_rate'],
    #     warmup_t=config['warmup_epochs'], warmup_lr_init=1e-6)
    
    num_steps_per_epoch = len(loader_train)

    scheduler = timm.scheduler.CosineLRScheduler(
        optimizer, t_initial=config['num_epochs'] * num_steps_per_epoch, lr_min=1e-6,
        warmup_t=config['warmup_epochs'] * num_steps_per_epoch, warmup_lr_init=1e-6,
        t_in_epochs=False
    )
    

    model_save_path = os.path.join(config['model_save_path'], run_id)
    os.makedirs(model_save_path, exist_ok=True)

    total_batches = 0
    for iepoch in range(config['num_epochs']):
        model.train()
        pbar = tqdm(loader_train)
        running_stats = dict()
        for ibatch, batch in enumerate(pbar):
            batch = torch_utils.batch_to_device(batch, device)
            scores = model(batch)
            y = batch['label']
            y_mask = batch['predict_mask']
            scores[:, 0] = scores[:, 0] - config['margin']
            scores = scores * 20
            loss = nn.functional.binary_cross_entropy_with_logits(scores, y, reduction='none')
            # triplet loss
            # loss = scores[:, 0:1] - scores[:, 1:]
            # loss = -torch.clamp(loss, max=0.0)

            # scores = scores + (1-y_mask) * -100000
            # loss = nn.functional.cross_entropy(scores, torch.zeros(scores.shape[0], dtype=torch.long, device=device))
            # loss = loss.mean()
            # y_mask[:, 0] = y_mask.sum(dim=1)
            # num_neg = y_mask[:, 1:].sum(dim=1)
            # y_mask[:, 1:] = y_mask[:, 1:] * 8 / num_neg[:, None]

            loss = (loss * y_mask[:, :]).sum() / y_mask[:, :].sum()
            
            optimizer.zero_grad()
            optimizer_article.zero_grad()
            optimizer_session.zero_grad()
            loss.backward()

            # compute gradient norm except embeddings
            # avg_grad_model = compute_average_gradients([p for n, p in model.named_parameters() if 'embeddings' not in n])
            # avg_grad_session = compute_average_gradients(model.layer_embed_session.parameters())
            # avg_grad_article = compute_average_gradients(model.layer_embed_article.parameters())

            for g in optimizer_article.param_groups:
                g['lr'] = optimizer.param_groups[0]['lr'] * 100
                pass
            for g in optimizer_session.param_groups:
                g['lr'] = optimizer.param_groups[0]['lr'] * 100
                pass

            optimizer.step()
            optimizer_article.step()
            optimizer_session.step()

            loss = loss.item()

            current_stats = dict()
            current_stats['loss'] = loss
            # current_stats['gn_a'] = avg_grad_article
            # current_stats['gn_s'] = avg_grad_session
            # current_stats['gn_o'] = avg_grad_model

            for k, v in current_stats.items():
                if k not in running_stats:
                    running_stats[k] = v
                else:
                    running_stats[k] = running_stats[k] * 0.99 + v * 0.01
                    pass
                pass

            pbar.set_postfix(**running_stats, lr=optimizer.param_groups[0]['lr'])
            total_batches += 1
            scheduler.step_update(total_batches)
            pass
        pbar.close()
        # scheduler.step(iepoch)


        # save model
        torch.save(model.state_dict(), f'{model_save_path}/recall_model_epoch{iepoch:03d}.pth')
        pass
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--load_emb', type=str, default='')
    args = parser.parse_args()

    main(args.device, args.load_model, args.load_emb)
    pass