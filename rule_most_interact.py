'''
most interacted items ignoring event types, if not enough, fill with most frequent items each type.
validation result: {'clicks': 0.32827333449537294, 'carts': 0.314755707897211, 'orders': 0.58768987210464, 'total': 0.47986796908148455}
'''


from config import config
import global_data
import numpy as np
import dataio
from tqdm import tqdm


def most_frequency(aid, etype, topn=20):
    
    uaid, aidmap = np.unique(aid, return_inverse=True)

    counts = np.zeros((uaid.shape[0], 3), dtype=np.int64)
    np.add.at(counts, (aidmap, etype), 1)

    counts_order = np.argsort(counts, axis=0, )[::-1, :]

    results= []
    for i in range(3):
        result_list = []
        for j in range(min(topn, uaid.shape[0])):
            jidx = counts_order[j, i]
            if counts[jidx, i] == 0:
                break
            else:
                result_list.append(uaid[jidx])
            pass
        results.append(result_list)
        pass

    return results
    pass

if __name__ == '__main__':
    global_data.init()
    sid_index = global_data.sid_index
    ts = global_data.ts
    aid = global_data.aid
    etype = global_data.etype

    test_start = sid_index[-global_data.data_info['num_test']]

    test_aid = aid[test_start:]
    test_etype = etype[test_start:]

    total_most_frequency = most_frequency(test_aid, test_etype, topn=20)

    test_sid = np.arange(global_data.data_info['num_train'], global_data.data_info['num_train'] + global_data.data_info['num_test'])

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
            if len(top_aid) < 20:
                session_most_frequency[i].extend(total_most_frequency[i][:20 - len(top_aid)])
                pass
            pass

        results.append((sid, session_most_frequency[0], session_most_frequency[1], session_most_frequency[2]))
        pass

    dataio.write_result(results, 'submission.csv')
    pass