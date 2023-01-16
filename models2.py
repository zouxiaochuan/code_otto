
import simple_transformer
import torch
import torch.nn as nn
import torch_utils


class OTTORecallModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.layer_embed_session = torch_utils.CategoryFeatureEmbedding(
            config['dims_session_cate'], config['hidden_size']
        )
        self.layer_encode_session = simple_transformer.MultiLayerTransformer(
            config['num_layer_session'], config['hidden_size'], reduction='mean',
            hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0,
            intermediate_size=config['intermediate_size_session'],
            attention_head_size=config['attention_head_size_session'],
        )

        self.session_bias = nn.Parameter(
            nn.init.normal_(torch.zeros(3, 1, config['hidden_size'])))

        self.layer_embed_article = torch_utils.CategoryFeatureEmbedding(
            config['dims_article_cate'], config['hidden_size']
        )


        self.layer_encode_article = nn.Sequential(
            nn.ReLU(),
            nn.Linear(config['hidden_size'], config['hidden_size'] * 3))

        self.hidden_size = config['hidden_size']
        pass


    def forward(self, data):
        feat_session = data['feat_session']
        session_mask = data['session_mask']
        feat_article = data['feat_article']
        event_type = data['event_type']

        x_session = self.forward_session(feat_session, session_mask, event_type)
        # x_session: [B, H]

        x_article = self.forward_article(feat_article, event_type)

        # x_article: [B, T, H]

        scores = torch.matmul(x_article, x_session[:, :, None]).squeeze(-1)
        # scores: [B, T]

        return scores

    def forward_session(self, feat_session, session_mask, event_type):
        # session_article = feat_session[:, :, :1]
        # feat_session = feat_session[:, :, 1:]
        
        # x_article = self.layer_embed_article(session_article)
        # x_article = self.forward_article(session_article, event_type)
        x_session = self.layer_embed_session(feat_session)
        # x_session = x_article + x_session
        # x_session: [B, T, H]

        # session_bias = self.session_bias[event_type]
        # session_bias: [B, 1, H]
        # x_session = torch.cat([session_bias, x_session], dim=1)
        # session_mask = torch.cat([torch.ones(x_session.shape[0], 1, device=session_mask.device), session_mask], dim=1)

        x_session = self.layer_encode_session(x_session, session_mask)
        # x_session = x_session * session_mask[:, :, None]
        # x_session = x_session.sum(dim=1) / session_mask.sum(dim=1, keepdim=True)
        return nn.functional.normalize(x_session, dim=-1)
    
    def forward_article(self, feat_article, event_type):
        # feat_article: [B, T, D]
        x_article_ = self.layer_embed_article(feat_article)
        # x_article = self.layer_encode_article(x_article_)
        # x_article = x_article.reshape(x_article.shape[0], x_article.shape[1], 3, self.hidden_size)
        # x_article = x_article[torch.arange(x_article.shape[0], device=x_article.device), :, event_type, :]
        # x_article = x_article + x_article_
        return nn.functional.normalize(x_article_, dim=-1)