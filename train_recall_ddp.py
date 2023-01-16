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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


run_id = 'event_type0_ddp'


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group('nccl', rank=rank, world_size=world_size)
    pass


def main(rank, num_processes, load_model):
    setup(rank, num_processes)
    device = torch.device('cuda', rank)
    torch.cuda.set_device(rank)
    torch.cuda.empty_cache()

    global_data.init()
    model = models.OTTORecallModel(config).to(device)
    if len(load_model) > 0:
        model.load_state_dict(torch.load(load_model, map_location=device))
        pass
    
    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    dataset_train = datasets.OTTORecallTrainDataset(
        num=2000000, negative_num=config['negative_num'], max_predict_days=config['max_predict_days'])

    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=config['batch_size'],
        num_workers=config['num_data_workers'],
        collate_fn=datasets.collate_fn
    )

    optimizer = torch.optim.AdamW(
        torch_utils.get_optimizer_params(model, config['learning_rate'], config['weight_decay']))
    
    # optimizer2 = torch.optim.SparseAdam(
    #     torch_utils.get_optimizer_params(model, config['learning_rate'], config['weight_decay'], must_include_keys=['embeddings']))

    scheduler = timm.scheduler.StepLRScheduler(
        optimizer, decay_t=config['learning_rate_decay_epochs'], decay_rate=config['learning_rate_decay_rate'],
        warmup_t=config['warmup_epochs'], warmup_lr_init=1e-6)
    
    # scheduler2 = timm.scheduler.StepLRScheduler(
    #     optimizer2, decay_t=config['learning_rate_decay_epochs'], decay_rate=config['learning_rate_decay_rate'],
    #     warmup_t=config['warmup_epochs'], warmup_lr_init=1e-6)

    if rank == 0:
        model_save_path = os.path.join(config['model_save_path'], run_id)
        os.makedirs(model_save_path, exist_ok=True)
        pass

    for iepoch in range(config['num_epochs']):
        model.train()

        if rank == 0:
            pbar = tqdm(loader_train)
            running_stats = dict()
            pass

        for ibatch, batch in enumerate(loader_train):
            batch = torch_utils.batch_to_device(batch, device)
            scores = ddp_model(batch)
            y = batch['label']
            y_mask = batch['predict_mask']
            loss = nn.functional.binary_cross_entropy_with_logits(scores, y, reduction='none')
            # scores = scores + (1-y_mask) * -100000
            # loss = nn.functional.cross_entropy(scores, torch.zeros(scores.shape[0], dtype=torch.long, device=device))
            # loss = loss.mean()
            # y_mask[:, 0] = y_mask.sum(dim=1)
            num_neg = y_mask[:, 1:].sum(dim=1)
            y_mask[:, 1:] = y_mask[:, 1:] * 8 / num_neg[:, None]

            loss = (loss * y_mask).sum() / y_mask.sum()
            
            optimizer.zero_grad()
            # optimizer2.zero_grad()
            loss.backward()
            optimizer.step()
            # optimizer2.step()

            if rank == 0:
                loss = loss.item()

                current_stats = dict()
                current_stats['loss'] = loss

                for k, v in current_stats.items():
                    if k not in running_stats:
                        running_stats[k] = v
                    else:
                        running_stats[k] = running_stats[k] * 0.99 + v * 0.01
                        pass
                    pass
        
                pbar.set_postfix(**running_stats, lr=optimizer.param_groups[0]['lr'])
                pbar.update()
                pass
            pass
        pbar.close()
        scheduler.step(iepoch)
        # scheduler2.step(iepoch)

        # save model
        torch.save(model.state_dict(), f'{model_save_path}/recall_model_epoch{iepoch:03d}.pth')
        pass
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--num_gpus', type=int, required=True)
    args = parser.parse_args()

    num_gpus = args.num_gpus
    mp.spawn(main, nprocs=num_gpus, args=(num_gpus, args.load_model), join=True)
    pass