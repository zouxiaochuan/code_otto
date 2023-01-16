from config_full import config
import os
from datetime import datetime
import numpy as np
from tqdm import tqdm
import common_utils


def train_test_ts():
    sid_index = np.load(os.path.join(config['mid_data_path'], 'sid_index.npy'))
    num_train = config['num_train']
    ts = np.load(os.path.join(config['mid_data_path'], 'ts.npy'))

    ts_train = ts[:sid_index[num_train]] * 0.001
    ts_test = ts[sid_index[num_train]:] * 0.001

    train_min_dt = datetime.fromtimestamp(ts_train.min()).strftime('%Y-%m-%d %H:%M:%S')
    train_max_dt = datetime.fromtimestamp(ts_train.max()).strftime('%Y-%m-%d %H:%M:%S')
    test_min_dt = datetime.fromtimestamp(ts_test.min()).strftime('%Y-%m-%d %H:%M:%S')
    test_max_dt = datetime.fromtimestamp(ts_test.max()).strftime('%Y-%m-%d %H:%M:%S')

    print(f'train min: {train_min_dt}, max: {train_max_dt}')
    print(f'test min: {test_min_dt}, max: {test_max_dt}')

    pass


def max_min_aid():
    aid = np.load(os.path.join(config['mid_data_path'], 'aid.npy'))
    print(f'max aid: {aid.max()}, min aid: {aid.min()}')
    pass

def type_num():
    etype = np.load(os.path.join(config['mid_data_path'], 'etype.npy'))
    print(np.bincount(etype))
    pass

def session_interval():
    sid_index = np.load(os.path.join(config['mid_data_path'], 'sid_index.npy'))
    ts = np.load(os.path.join(config['mid_data_path'], 'ts.npy'))

    intervals = []
    for i in tqdm(range(len(sid_index) - 1)):
        start = sid_index[i]
        end = sid_index[i + 1]
        interval = ts[end - 1] - ts[start]
        intervals.append(interval)
        pass

    intervals = np.array(intervals) / (1000*60*60)
    print(f'max interval: {np.max(intervals)}, min interval: {np.min(intervals)}')

def session_interval_less7day():
    sid_index = np.load(os.path.join(config['mid_data_path'], 'sid_index.npy'))
    ts = np.load(os.path.join(config['mid_data_path'], 'ts.npy'))
    
    # train_sid_index = sid_index[:config['num_train']]

    print(len(sid_index))
    result = []

    for sid in tqdm(range(len(sid_index) - 1)):
        s_start = sid_index[sid]
        s_end = sid_index[sid + 1] - 1
        start = ts[s_start]
        end = ts[s_end]
        interval = (end - start) / (1000*60*60)
        if interval <= 7*24:
            result.append(s_end - s_start + 1)
            pass
        pass
    print(len(result))
    print(np.mean(result))
    pass

def event_num():
    data_info = common_utils.load_obj(os.path.join(config['mid_data_path'], 'data_info.pkl'))
    sid_index = np.load(os.path.join(config['mid_data_path'], 'sid_index.npy'))

    event_num = []
    for sid in tqdm(range(data_info['num_train'])):
        start = sid_index[sid]
        end = sid_index[sid + 1]
        event_num.append(end - start)
        pass

    print(np.mean(event_num))

    event_num = []
    for sid in tqdm(range(data_info['num_train'], data_info['num_train'] + data_info['num_test'])):
        start = sid_index[sid]
        end = sid_index[sid + 1]
        event_num.append(end - start)
        pass

    print(np.mean(event_num))
    pass

def session_average_size():
    sid_index = np.load(os.path.join(config['mid_data_path'], 'sid_index.npy'))
    
    sizes = []
    for sid in tqdm(range(len(sid_index) - 1)):
        start = sid_index[sid]
        end = sid_index[sid + 1]
        sizes.append(end - start)
        pass

    print(np.mean(sizes))
    pass

if __name__ == '__main__':
    # train_test_ts()
    # num_valid()
    # max_min_aid()
    # type_num()
    # session_interval()
    # session_interval_less7day()
    # event_num()
    session_average_size()
    pass