'''
covisitation matrix.
validation result: {'clicks': 0.4938548612558914, 'carts': 0.38454106112593284, 'orders': 0.626808552742872, 'total': 0.5408329361090922}
'''


from config import config
import global_data
import numpy as np
import dataio
from tqdm import tqdm


def calc_covisition_matrix(sid_index, aid, ts):
    aid_session = np.zeros(aid.shape, dtype=np.int64)
    for i in range(sid_index.shape[0] - 1):
        start = sid_index[i]
        end = sid_index[i + 1]
        aid_session[start:end] = i
        pass

    aid_session_next = np.roll(aid_session, -1)
    aid_next = np.roll(aid, -1)
    ts_next = np.roll(ts, -1)

    aid_pair = np.stack([aid, aid_next], axis=1)

    next_same_session = aid_session_next == aid_session
    next_same_session[-1] = False
    next_within_day = (ts_next - ts) < 86400000
    next_within_day[-1] = False

    pair_mask = next_same_session & next_within_day
    aid_pair = aid_pair[pair_mask, :]

    upair, pair_counts = np.unique(aid_pair, axis=0, return_counts=True)

    order = np.argsort(pair_counts)[::-1]
    upair = upair[order, :]
    pair_counts = pair_counts[order]

    mat = dict()
    for (prev_aid, next_aid), cnt in zip(upair, tqdm(pair_counts)):
        row = mat.get(prev_aid)
        if row is None:
            row = []
            mat[prev_aid] = row
            pass
        row.append((next_aid, cnt))
        pass

    return mat


if __name__ == '__main__':
    sid_index = global_data.sid_index
    ts = global_data.ts
    aid = global_data.aid

    test_start = sid_index[-config['num_test']]

    comatrix = calc_covisition_matrix(sid_index, aid, ts)

    test_sid = np.arange(config['num_train'], config['num_train'] + config['num_test'])

    results = []
    for sid in tqdm(test_sid):
        estart = sid_index[sid]
        eend = sid_index[sid + 1]

        session_aid = aid[estart:eend]
        
        uaid, counts = np.unique(session_aid, return_counts=True)

        top_aid = uaid[np.argsort(counts)[::-1]][:20].tolist()
        
        session_most_frequency = []
        for i in range(3):
            session_most_frequency.append(top_aid.copy())
            rlen = len(session_most_frequency[i])
            if rlen < 20:
                last_aid = session_most_frequency[i][-1]
                last_aid_covisitation = comatrix.get(last_aid)
                if last_aid_covisitation is not None:
                    session_most_frequency[i].extend([a for a, _ in last_aid_covisitation[:20 - rlen]])
                    pass
                pass
            pass

        results.append((sid, session_most_frequency[0], session_most_frequency[1], session_most_frequency[2]))
        pass

    dataio.write_result(results, 'submission.csv')
    pass