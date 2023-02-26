import os
import numpy as np
import common_utils
import global_data
import collections
from tqdm import tqdm
from config import config


def extract_feat(session_splits, topk, topk_scores, sid_index, aid, etype):
    num_instances = topk.shape[0] * topk.shape[1]

    feat = np.zeros((num_instances, 4), dtype='float32')

    for i in tqdm(range(topk.shape[0])):
        sid, split_idx = session_splits[i]
        session_start = sid_index[sid]
        split_eid = session_start + split_idx

        aid_type_counts = collections.defaultdict(int)
        for k in range(session_start, split_eid+1):
            aid_type_counts[(aid[k], etype[k])] += 1
            pass

        for j in range(topk.shape[1]):
            feat[i * topk.shape[1] + j, 0] = topk_scores[i, j]

            for k in range(3):
                feat[i * topk.shape[1] + j, 1 + k] = aid_type_counts.get((topk[i, j], k), 0)
                pass
            pass
        pass

    return feat

def extract_label(session_splits, session_splits_positives, topk):
    num_instances = topk.shape[0] * topk.shape[1]

    label = np.zeros((num_instances, 1), dtype='int32')

    for i in tqdm(range(topk.shape[0])):

        positives = set(session_splits_positives[i])
        for j in range(topk.shape[1]):
        
            if topk[i, j] in positives:
                label[i * topk.shape[1] + j] = 1
                pass
            pass
        pass   

    return label 

def main(config):
    sid_index = global_data.sid_index
    aid = global_data.aid
    etype = global_data.etype

    # extract training feature
    for itype in range(3):
        topk_file = os.path.join(config['mid_data_path'], f'topk_et{itype}_train.npy')
        topk_scores_file = os.path.join(config['mid_data_path'], f'topk_scores_et{itype}_train.npy')
        session_splits_file = os.path.join(config['mid_data_path'], f'session_splits_et{itype}.npy')
        session_splits_positives_file = os.path.join(config['mid_data_path'], f'session_splits_positives_et{itype}.pkl')

        topk = np.load(topk_file)
        topk_scores = np.load(topk_scores_file)
        session_splits = np.load(session_splits_file)
        session_splits_positives = common_utils.load_obj(session_splits_positives_file)

        feat = extract_feat(session_splits, topk, topk_scores, sid_index, aid, etype)
        label = extract_label(session_splits, session_splits_positives, topk)

        feat_file = os.path.join(config['mid_data_path'], f'feat_et{itype}_train.npy')
        label_file = os.path.join(config['mid_data_path'], f'label_et{itype}_train.npy')
        np.save(feat_file, feat)
        np.save(label_file, label)
        pass

    # extract test feature
    for itype in range(3):
        topk_file = os.path.join(config['mid_data_path'], f'topk_et{itype}_test.npy')
        topk_scores_file = os.path.join(config['mid_data_path'], f'topk_scores_et{itype}_test.npy')

        topk = np.load(topk_file)
        topk_scores = np.load(topk_scores_file)
        session_splits = []
        num_test = global_data.data_info['num_test']
        num_train = global_data.data_info['num_train']
        for i in range(num_test):
            sid = num_train + i
            session_start = sid_index[sid]
            session_end = sid_index[sid+1]
            session_splits.append((sid, session_end - session_start - 1))
            pass
        
        session_splits = np.array(session_splits, dtype='int32')
        feat = extract_feat(session_splits, topk, topk_scores, sid_index, aid, etype)

        feat_file = os.path.join(config['mid_data_path'], f'feat_et{itype}_test.npy')
        np.save(feat_file, feat)
        pass

    pass

if __name__ == '__main__':
    global_data.init()
    main(config)
    pass