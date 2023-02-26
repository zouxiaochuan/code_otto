import argparse
from predict_embedding import main as predict_embedding_main
from evaluate_embedding import main as evaluate_embedding_main
from evaluate_embedding import read_ground_truth

from config import config
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_file', type=str, required=True)
    args = parser.parse_args()
    for part in ['article', 'session']:
        for et in range(3):
            emb_file = os.path.join(config['mid_data_path'], f'emb_{part}_et{et}.npy')
            predict_embedding_main(part, args.ckpt_file, emb_file, 'cuda:3', et)
        pass

    ground_truth = read_ground_truth(os.path.join(config['raw_data_path'], 'test_labels.jsonl'))
    for et in range(3):
        session_emb = np.load(os.path.join(config['mid_data_path'], f'emb_session_et{et}.npy'))
        article_emb = np.load(os.path.join(config['mid_data_path'], f'emb_article_et{et}.npy'))
        evaluate_embedding_main(
            session_emb=session_emb, article_emb=article_emb, device='cuda:3', recall_num=230,
            ground_truth=ground_truth[et])
        pass
    pass