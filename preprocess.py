import os
from config import config
import itertools
import multiprocessing as mp
import json
from tqdm import tqdm
import numpy as np
import constants
import common_utils


def process_line(param):
    line = param
    rec = json.loads(line)
    sid = rec['session']

    aids = []
    tss = []
    types = []

    for event in rec['events']:
        aid = event['aid']
        ts = event['ts']
        type = constants.etype_map[event['type']]

        aids.append(aid)
        tss.append(ts)
        types.append(type)
        pass

    return sid, aids, tss, types

def process_raw_data(config):
    file_train_raw = os.path.join(config['raw_data_path'], 'train.jsonl')
    file_test_raw = os.path.join(config['raw_data_path'], 'test.jsonl')
    
    lines_train = open(file_train_raw, 'r').readlines()
    lines_test = open(file_test_raw, 'r').readlines()
    lines = lines_train + lines_test

    num_train = len(lines_train)
    num_test = len(lines_test)
    # total_num = config['num_train'] + config['num_test']

    pool = mp.Pool(processes=32)
    results = pool.map(process_line, tqdm(lines), chunksize=10000)

    sid_index = []
    aid = []
    ts = []
    etype = []
    enum = 0
    original_sid = []
    for r in tqdm(results):
        event_sid, event_aids, event_tss, event_types = r

        # select_range = [ts - event_tss[0] <= 7 * 24 * 60 * 60 * 1000 for ts in event_tss]
        # event_aids = [aid for aid, select in zip(event_aids, select_range) if select]
        # event_types = [type for type, select in zip(event_types, select_range) if select]
        # event_tss = [ts for ts, select in zip(event_tss, select_range) if select]

        sid_index.append(enum)
        aid.extend(event_aids)
        ts.extend(event_tss)
        etype.extend(event_types)
        enum += len(event_aids)
        original_sid.append(event_sid)
        pass
    
    # assert len(sid_index) == total_num

    print(f'total events: {enum}')

    sid_index.append(enum)

    sid_index = np.array(sid_index, dtype=np.int64)
    aid = np.array(aid, dtype=np.int64)
    ts = np.array(ts, dtype=np.int64)
    etype = np.array(etype, dtype=np.int8)
    original_sid = np.array(original_sid, dtype=np.int64)

    pool.close()

    os.makedirs(config['mid_data_path'], exist_ok=True)
    np.save(os.path.join(config['mid_data_path'], 'sid_index.npy'), sid_index)
    np.save(os.path.join(config['mid_data_path'], 'aid.npy'), aid)
    np.save(os.path.join(config['mid_data_path'], 'ts.npy'), ts)
    np.save(os.path.join(config['mid_data_path'], 'etype.npy'), etype)
    np.save(os.path.join(config['mid_data_path'], 'original_sid.npy'), original_sid)
    data_info = {
        'num_train': num_train,
        'num_test': num_test,
    }
    common_utils.save_obj(data_info, os.path.join(config['mid_data_path'], 'data_info.pkl'))

    pass


def generate_train_candidates_process_fn(param):
    session_start, session_end = param
    import global_data
    ts = global_data.ts
    etype = global_data.etype

    candidates = [[], [], []]
    for ievent in reversed(range(session_start+1, session_end)):
        its = ts[ievent]
        itype = etype[ievent]

        for jevent in reversed(range(session_start, ievent)):
            jts = ts[jevent]

            if (jts - its) > 0:
                raise RuntimeError('logic error')
                pass

            if (its - jts) > 7 * 24 * 60 * 60 * 1000:
                # the test interval is 7 day
                break

            candidates[itype].append((jevent, ievent))

            jtype = etype[jevent]

            if itype == 0 and jtype==0:
                break
            pass
        pass
    return candidates
    pass

def generate_train_candidates_(config):
    import global_data
    global_data.init(names=['sid_index', 'ts', 'etype'])

    sid_index = global_data.sid_index
    pool = mp.Pool()
    params = [(sid_index[isession], sid_index[isession + 1]) for isession in range(len(sid_index) - 1)]
    results = pool.map(generate_train_candidates_process_fn, params)
    candidates = [[], [], []]

    for r in tqdm(results):
        for i in range(3):
            candidates[i].extend(r[i])
            pass
        pass

    pool.close()
    
    for i in range(3):
        candidates[i] = np.array(candidates[i], dtype=np.int64)
        np.save(os.path.join(config['mid_data_path'], f'train_candidate_pairs_{i}.npy'), candidates[i])
        pass
    pass

def generate_eid2sid(global_data, config):
    eid2sid = np.zeros(global_data.aid.shape, dtype=np.int64)
    for isession in tqdm(range(len(global_data.sid_index) - 1)):
        session_start = global_data.sid_index[isession]
        session_end = global_data.sid_index[isession + 1]
        eid2sid[session_start:session_end] = isession
        pass

    np.save(os.path.join(config['mid_data_path'], 'eid2sid.npy'), eid2sid)
    
    return eid2sid


def generate_train_candidates(global_data, config, eid2sid):
    for itype in range(3):

        train_eids = np.argwhere(global_data.etype == itype).flatten()
        
        session_end = 0
        candidates = []
        for eid in tqdm(train_eids):
            if eid >= session_end:
                sid = eid2sid[eid]
                session_start = global_data.sid_index[sid]
                session_end = global_data.sid_index[sid + 1]
                ts_start = global_data.ts[session_start]
                pass

            if eid == session_start:
                # first event is not a candidate
                continue

            if (global_data.ts[eid] - ts_start) <= config['max_predict_days'] * 24 * 60 * 60 * 1000:
                if itype == 0:
                    # check last click is in first 7 days
                    for j in reversed(range(session_start, eid)):
                        if global_data.etype[j] == 0:
                            if (global_data.ts[j] - ts_start) <= 7 * 24 * 60 * 60 * 1000:
                                candidates.append(eid)
                                pass
                            break
                        pass
                    pass
                else:
                    candidates.append(eid)
                    pass
                pass
            pass

        candidates = np.array(candidates, dtype=np.int64)
        print(f'type {itype}, num candidates: {candidates.shape}')
        np.save(os.path.join(config['mid_data_path'], f'train_eids_type{itype}.npy'), candidates)
        pass
    pass



def generate_extra_data(config):
    import global_data
    global_data.init(names=['sid_index', 'aid', 'ts', 'etype'])

    # eid2sid = generate_eid2sid(global_data, config)

    # generate_train_candidates(global_data, config, eid2sid)

    generate_article_feat_type_rate(global_data.aid, global_data.ts, global_data.etype, 24 * 60 * 60 * 1000, 'article_feat_type_rate_1d')
    
    pass


def generate_article_feat_type_rate(aid, ts, etype, interval, name):
    ts_interval = ts // interval

    rows = np.stack(
        [ts_interval, etype, aid], axis=1)
    
    rows = rows.astype(np.int32)
    
    urows, counts = np.unique(rows, axis=0, return_counts=True)
    utypes, etype_counts = np.unique(rows[:, [0, 1]], axis=0, return_counts=True)

    etype_counts = {tuple(r):c for r, c in tqdm(zip(utypes, etype_counts))}

    rate = np.zeros(counts.shape, dtype=np.float32)
    for i in tqdm(range(urows.shape[0])):
        rate[i] = counts[i] / etype_counts[(urows[i, 0], urows[i, 1])]
        pass

    feat_table_keys = urows
    feat_table_values = rate[:, None]

    np.save(os.path.join(config['mid_data_path'], f'{name}_keys.npy'), feat_table_keys)
    np.save(os.path.join(config['mid_data_path'], f'{name}_values.npy'), feat_table_values)
    pass

if __name__ == '__main__':
    # process_raw_data(config)
    generate_extra_data(config)
    pass