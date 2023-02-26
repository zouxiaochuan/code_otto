import os
import constants

MAX_AID = 1855602
TIME_ENCODING_DIMS = [
    constants.time_encoding_max_minutes,
    constants.time_encoding_max_hours,
    constants.time_encoding_max_days
]

config = {
    'raw_data_path': os.path.expanduser('~/data/zouxiaochuan/017_otto/mid_data/raw_valid'),
    'mid_data_path': os.path.expanduser('~/data/zouxiaochuan/017_otto/mid_data/valid_data'),
    'model_save_path': os.path.expanduser('../mid_data/valid_model'),
    'max_predict_days': 7+1 ,
    'batch_size': 256,
    'num_data_workers': 4,
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'learning_rate_decay_rate': 0.99999999,
    'learning_rate_decay_epochs': 1,
    'weight_decay': 1e-5,
    'warmup_epochs': 1,
    'negative_num': 128,
    'time_encoding_dims': TIME_ENCODING_DIMS,
    'dims_session_cate': TIME_ENCODING_DIMS + [len(constants.etype_map)],
    'dims_article_cate': [MAX_AID+1],
    'hidden_size': 256,
    'num_layer_session': 4,
    'intermediate_size_session': 256,
    'attention_head_size_session': 32,
    'max_session_size': 64,
    'margin': 0.3,
}