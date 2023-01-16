import numpy as np
import os
import sys
from config import config
import common_utils


self_module = sys.modules[__name__]

def init(names=None):
    if names is None:
        names = ['sid_index', 'ts', 'aid', 'etype', 'eid2sid']
        for i in range(3):
            names.append(f'train_eids_type{i}')
            pass
        pass

    for name in names:
        setattr(self_module, name, np.load(os.path.join(config['mid_data_path'], name + '.npy')))
        pass

    if 'train_eids_type0' in names:
        self_module.train_eids = [getattr(self_module, f'train_eids_type{i}') for i in range(3)]
        pass

    if os.path.exists(os.path.join(config['mid_data_path'], 'data_info.pkl')):
        self_module.data_info = common_utils.load_obj(os.path.join(config['mid_data_path'], 'data_info.pkl'))
        pass
    pass