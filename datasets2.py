
import torch
import torch.utils.data
import global_data
import numpy as np
import common_utils
import constants

class OTTOBaseDataset(torch.utils.data.Dataset):
    def __init__(self, max_session_size):
        self.sid_index = global_data.sid_index
        self.ts = global_data.ts
        self.aid = global_data.aid
        self.etype = global_data.etype

        self.eid2sid = global_data.eid2sid
        self.num_aid = np.max(self.aid) + 1
        self.train_eids = global_data.train_eids        
        self.max_session_size = max_session_size
        pass

    def time_encoding(self, ts):
        ts = ts[-1] - ts
        minutes = ts // (60 * 1000)
        hours = ts // (60 * 60 * 1000)
        days = ts // (24 * 60 * 60 * 1000)

        minutes = np.minimum(minutes, constants.time_encoding_max_minutes - 1)
        hours = np.minimum(hours, constants.time_encoding_max_hours - 1)
        days = np.minimum(days, constants.time_encoding_max_days - 1)

        return np.stack([minutes, hours, days], axis=1)

    def extract_feat_session(self, split_eid):
        sid = self.eid2sid[split_eid]
        session_start = self.sid_index[sid]

        feat_eids = np.arange(session_start, split_eid+1)
        feat_eids_ts = self.ts[feat_eids]
        feat_eids_type = self.etype[feat_eids]

        time_code = self.time_encoding(feat_eids_ts)
        feat_eids_aid = self.aid[feat_eids]

        feat = np.concatenate(
            [feat_eids_aid[:, None], time_code, feat_eids_type[:, None]],
            axis=1
        )

        # note: session feature is in reverse order
        return feat[::-1][:self.max_session_size]

    def extract_feat_article(self, aids):

        return aids[:, None]


class OTTORecallTrainDataset(OTTOBaseDataset):
    def __init__(self, num, negative_num, max_predict_days, max_session_size):
        super().__init__(max_session_size=max_session_size)
        self.sid_index = global_data.sid_index
        self.ts = global_data.ts
        self.aid = global_data.aid
        self.etype = global_data.etype
        self.num = num

        self.eid2sid = global_data.eid2sid
        self.negative_num = negative_num
        self.num_aid = np.max(self.aid) + 1
        self.train_eids = global_data.train_eids
        self.max_predict_days = max_predict_days
        pass


    def sample_data(self, ):
        while True:
            # sample one positive data
            # nums = [len(self.train_eids[itype]) for itype in range(3)]
            # total_num = np.sum(nums)
            # sample_idx = np.random.choice(total_num)

            # if sample_idx < nums[0]:
            #     sample_type = 0
            #     eidx = self.train_eids[sample_type][sample_idx]
            # elif sample_idx < (nums[0] + nums[1]):
            #     sample_type = 1
            #     eidx = self.train_eids[sample_type][sample_idx - nums[0]]
            # else:
            #     sample_type = 2
            #     eidx = self.train_eids[sample_type][sample_idx - nums[0] - nums[1]]
            #     pass

            sample_type = np.random.choice(3, p=[0.6, 0.2, 0.2])
            eidx = np.random.choice(self.train_eids[sample_type])

            sidx = self.eid2sid[eidx]
            session_start = self.sid_index[sidx]
            session_end = self.sid_index[sidx + 1]
            ts_start = self.ts[session_start]
            if eidx == session_start:
                continue

            candidates = (self.ts[session_start: eidx] - ts_start)<= (7 * 24 * 3600 * 1000)
            if sample_type == 0:
                for j in reversed(range(session_start, eidx)):
                    if self.etype[j] == 0:
                        candidates[: j-session_start] = False
                        break
                    pass
                pass

            candidates = np.argwhere(candidates).flatten()
            if len(candidates) == 0:
                continue

            choosed = np.random.choice(candidates)
            split_eid = session_start + choosed

            predict_eid = eidx

            # sample negative data
            negative_aids = np.random.choice(self.num_aid, self.negative_num)

            if sample_type == 0:
                except_aids = np.array([self.aid[predict_eid]])
            else:
                mask_except = (self.ts[split_eid: session_end] - self.ts[session_start]) \
                        <= (self.max_predict_days * 24 * 3600 * 1000)
                mask_except = np.logical_and(
                    mask_except,
                    self.etype[split_eid: session_end] == sample_type
                )
                except_eids = np.argwhere(mask_except).flatten() + split_eid
                except_aids = np.unique(self.aid[except_eids])
                pass
            
            negative_mask = np.any(
                negative_aids[:, None] != except_aids[None, :],
                axis=1)
            break

        return split_eid, predict_eid, negative_aids, negative_mask, sample_type
        pass

    def __getitem__(self, index):

        split_eid, predict_eid, negative_aids, negative_mask, sample_type = self.sample_data()

        feat_session = self.extract_feat_session(split_eid)

        # remember: first is positive
        predict_aids = np.concatenate(
            (
                [self.aid[predict_eid]],
                negative_aids
            ))
        
        predict_mask = np.concatenate(
            (
                np.ones(shape=(1,)),
                negative_mask
            ))

        feat_article = self.extract_feat_article(predict_aids)

        label = np.zeros(shape=(len(predict_aids),))
        label[0] = 1
        
        return {
            'feat_session': feat_session.astype('int64'),
            'feat_article': feat_article.astype('int64'), 
            'label': label.astype('float32'),
            'predict_mask': predict_mask.astype('float32'),
            'event_type': sample_type
        }


    def __len__(self):
        return self.num
    pass


def collate_fn(datas):

    result = dict()

    if 'feat_session' in datas[0]:
        feat_session = [d['feat_session'] for d in datas]
        feat_session, session_mask = common_utils.collate_seq(feat_session)
        result['feat_session'] = torch.from_numpy(feat_session)
        result['session_mask'] = torch.from_numpy(session_mask)
        pass

    for key in datas[0].keys():
        if key != 'feat_session':
            result[key] = torch.from_numpy(np.stack([d[key] for d in datas], axis=0))
            pass
        pass

    return result


class OTTOPredictTestSessionEmbeddingDataset(OTTOBaseDataset):
    def __init__(self, max_session_size):
        super().__init__(max_session_size=max_session_size)
        self.num_test = global_data.data_info['num_test']
        self.num_train = global_data.data_info['num_train']
        pass

    def __getitem__(self, index):

        sid = self.num_train + index
        session_end = self.sid_index[sid + 1]

        feat_session = self.extract_feat_session(session_end - 1)

        
        return {
            'feat_session': feat_session
        }


    def __len__(self):
        return self.num_test
    pass


class OTTOPredictSessionEmbeddingDataset(OTTOBaseDataset):
    def __init__(self, session_splits, max_session_size):
        super().__init__(max_session_size=max_session_size)
        self.session_splits = session_splits
        pass

    def __getitem__(self, index):

        sid, split_id = self.session_splits[index]
        session_start = self.sid_index[sid]

        feat_session = self.extract_feat_session(session_start + split_id)

        
        return {
            'feat_session': feat_session
        }


    def __len__(self):
        return self.session_splits.shape[0]
    pass


class OTTOPredictArticleEmbeddingDataset(OTTOBaseDataset):
    def __init__(self, ):
        super().__init__(max_session_size=0)
        self.num = np.max(self.aid) + 1
        pass

    def __getitem__(self, index):

        aid = np.arange(index, index + 1)

        feat_article = self.extract_feat_article(aid)

        
        return {
            'feat_article': feat_article
        }


    def __len__(self):
        return self.num
    pass