
import argparse
import os
from tqdm import tqdm
from config import config
from predict_embedding import main as predict_embedding_main
import global_data
import numpy as np
import otto_utils
import torch_utils
import common_utils


def extract_article_embedding(config, load_model, device):
    for event_type in range(3):
        article_emb_file = os.path.join(config['mid_data_path'], f'emb_article_et{event_type}.npy')
        predict_embedding_main('article', load_model, article_emb_file, 'cuda:7', event_type)
        pass


def generate_session_splits(num_generate, config):
    sid_index = global_data.sid_index
    ts = global_data.ts
    etype = global_data.etype
    aid = global_data.aid
    num_train = global_data.data_info['num_train']
    
    for itype in range(3):
        session_splits = []
        session_splits_positives = []
        for i in tqdm(range(num_train)):
            i_reverse = num_train - i

            if i_reverse < 0:
                break

            session_start = sid_index[i_reverse]
            session_end = sid_index[i_reverse + 1]

            ts_start = ts[session_start]
            ts_diff = ts[session_start: session_end] - ts_start
            candidates = np.argwhere(ts_diff < (config['max_predict_days'] * 24 * 3600 * 1000)).flatten()

            if len(candidates) <= 1:
                continue

            select_idx = np.random.choice(len(candidates)-1)

            positives = etype[session_start + candidates[select_idx + 1:]] == itype
            if positives.sum() == 0:
                continue

            positives_aid = aid[session_start + candidates[select_idx + 1:][positives]]
            if itype == 0:
                positives_aid = positives_aid[:1]
                pass

            positives_aid = np.unique(positives_aid)

            session_splits.append((i_reverse, candidates[select_idx]))
            session_splits_positives.append(positives_aid)

            if len(session_splits) >= num_generate:
                break
            pass

        session_splits = np.array(session_splits, dtype='int32')
        session_splits_file = os.path.join(config['mid_data_path'], f'session_splits_et{itype}.npy')
        session_splits_positives_file = os.path.join(config['mid_data_path'], f'session_splits_positives_et{itype}.pkl')
        np.save(session_splits_file, session_splits)
        common_utils.save_obj(session_splits_positives, session_splits_positives_file)
        pass

    return session_splits


def extract_session_embedding(load_model, device):
    for itype in range(3):
        session_splits = np.load(os.path.join(config['mid_data_path'], f'session_splits_et{itype}.npy'))
        session_emb_file = os.path.join(config['mid_data_path'], f'emb_session_et{itype}_train.npy')
        otto_utils.predict_session_embedding(session_splits, load_model, session_emb_file, device, itype)
        pass
    pass


def extract_topk_candidates(config, k, device):
    for itype in range(3):
        session_emb_file = os.path.join(config['mid_data_path'], f'emb_session_et{itype}_train.npy')
        article_emb_file = os.path.join(config['mid_data_path'], f'emb_article_et{itype}.npy')

        topk, topk_scores = torch_utils.chunked_topk(session_emb_file, article_emb_file, k, device='cuda:7')

        topk_file = os.path.join(config['mid_data_path'], f'topk_et{itype}_train.npy')
        np.save(topk_file, topk)

        topk_scores_file = os.path.join(config['mid_data_path'], f'topk_scores_et{itype}_train.npy')
        np.save(topk_scores_file, topk_scores)
        pass
    pass

def extract_test_session_candidates(config, load_model, k, device):
    for itype in range(3):
        session_emb_file_test = os.path.join(config['mid_data_path'], f'emb_session_et{itype}_test.npy')
        predict_embedding_main('session', load_model, session_emb_file_test, device, itype)

        article_emb_file = os.path.join(config['mid_data_path'], f'emb_article_et{itype}.npy')
        # calculate test topk candidates
        topk_test, topk_scores_test = torch_utils.chunked_topk(session_emb_file_test, article_emb_file, k, device=device)

        topk_file_test = os.path.join(config['mid_data_path'], f'topk_et{itype}_test.npy')
        np.save(topk_file_test, topk_test)

        topk_scores_file_test = os.path.join(config['mid_data_path'], f'topk_scores_et{itype}_test.npy')
        np.save(topk_scores_file_test, topk_scores_test)
        pass
    pass

def main(load_model, num_generate):

    # extract article embedding
    # extract_article_embedding(config, load_model, 'cuda:7')

    # for every session select split
    # generate_session_splits(num_generate, config)

    # extract session embedding
    # extract_session_embedding(load_model, 'cuda:7')

    # calculate topk candidates
    # extract_topk_candidates(config, 100, 'cuda:7')

    # generate test session
    extract_test_session_candidates(config, load_model, 100, 'cuda:7')

    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--load_model', type=str, default='', required=True)
    parser.add_argument('--num_generate', type=int, default=100000)
    # parser.add_argument('--load_emb', type=str, default='')
    args = parser.parse_args()

    global_data.init()
    main(args.load_model, args.num_generate)
    pass