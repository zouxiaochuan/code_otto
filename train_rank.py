import os
import numpy as np
import lightgbm as lgb
import common_utils
import argparse
from config import config


def main(config, itype):
    feat_train_file = os.path.join(config['mid_data_path'], f'feat_et{itype}_train.npy')
    label_train_file = os.path.join(config['mid_data_path'], f'label_et{itype}_train.npy')

    feat_train = np.load(feat_train_file)
    label_train = np.load(label_train_file).flatten()

    model = lgb.LGBMClassifier(num_leaves=31, learning_rate=0.1, n_estimators=100)
    model.fit(
        feat_train, label_train, eval_set=[(feat_train, label_train)], eval_metric='auc')

    common_utils.save_obj(model, os.path.join(config['mid_data_path'], f'rank_model_et{itype}.pkl'))

    feat_test = np.load(os.path.join(config['mid_data_path'], f'feat_et{itype}_test.npy'))

    test_proba = model.predict(feat_test)
    test_topk = np.load(os.path.join(config['mid_data_path'], f'topk_et{itype}_test.npy'))

    test_proba = test_proba.reshape(test_topk.shape)
    test_proba_order = np.argsort(test_proba, axis=1)

    
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--event_type', type=int, default=0)

    args = parser.parse_args()

    main(config, args.event_type)    
    pass