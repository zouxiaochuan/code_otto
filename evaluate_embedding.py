import torch
import numpy as np
from tqdm import tqdm
import argparse
from config import config
import os
import json
import constants


def main(session_emb, article_emb, device, recall_num, ground_truth):
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    
    chunk_size = 512

    article_emb = torch.from_numpy(article_emb).to(device)
    topn = []
    for i in tqdm(range(0, len(session_emb), chunk_size)):
        session_emb_chunk = session_emb[i:i+chunk_size]
        session_emb_chunk = torch.from_numpy(session_emb_chunk).to(device)
        
        with torch.no_grad():
            scores = torch.matmul(session_emb_chunk, article_emb.t())
            scores, idx = scores.topk(recall_num, dim=1)
            pass

        idx = idx.cpu().numpy()
        topn.append(idx)
        pass

    topn = np.vstack(topn)

    total_num = 0
    hit_num = 0
    for gt, pred in tqdm(zip(ground_truth, topn)):
        gt = set(gt)

        for p in pred:
            if p in gt:
                hit_num += 1
                pass
            pass

        total_num += len(gt)
        pass

    print(f'hit_num: {hit_num}, total_num: {total_num}, recall: {hit_num / total_num}')
    pass


def read_ground_truth(filename):
    results = [list() for _ in range(3)]
    with open(filename) as fin:
        for line in tqdm(fin):
            rec = json.loads(line)
            for name, idx in constants.etype_map.items():
                if name in rec['labels']:
                    gt = rec['labels'][name]
                    if name == 'clicks':
                        results[idx].append(set([gt]))
                        pass
                    else:
                        results[idx].append(set(gt))
                        pass
                    pass
                else:
                    results[idx].append(set())
                    pass
                pass
            pass
        pass
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--session_emb_file', type=str, default=os.path.join(config['mid_data_path'], 'emb_session.npy'))
    parser.add_argument('--article_emb_file', type=str, default=os.path.join(config['mid_data_path'], 'emb_article.npy'))
    parser.add_argument('--recall_num', type=int, default=100)
    parser.add_argument('--ground_truth_file', type=str, default=os.path.join(config['raw_data_path'], 'test_labels.jsonl'))
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--event_type', type=int, required=True)
    args = parser.parse_args()

    session_emb = np.load(args.session_emb_file)
    article_emb = np.load(args.article_emb_file)
    ground_truth = read_ground_truth(args.ground_truth_file)

    main(session_emb, article_emb, args.device, args.recall_num, ground_truth[args.event_type])

    pass