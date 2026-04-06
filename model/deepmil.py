import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layers import *


EPS = 1e-6
__all__ = [
    "DeepMIL", "DeepMILEncoder"
]


#####################################################################################
#  Common deep MIL networks: Max-pooling, Mean-pooling, ABMIL
#####################################################################################


class DeepMIL(nn.Module):
    """
    Deep Multiple Instance Learning for Bag-level Task.

    Args:
        dim_in: input instance dimension.
        dim_emb: instance embedding dimension.
        num_cls: the number of class to predict.
        pooling: the type of MIL pooling, one of 'mean', 'max', and 'attention', default by attention pooling.
    """
    def __init__(self, dim_in=768, dim_emb=512, num_cls=2, dim_attn=384, num_feat_proj_layers=1, drop_rate=0.25, 
        pooling='attention', post_mil_layer='Identity', pred_head='default', **kwargs):
        super().__init__()
        assert pooling in ['mean', 'max', 'attention', 'gated_attention']
        assert pred_head in ['default']

        self.feat_proj = create_mlp(
            in_dim=dim_in,
            hid_dims=[dim_emb] * (num_feat_proj_layers - 1),
            dropout=drop_rate,
            out_dim=dim_emb,
            end_with_fc=False
        )
        
        if pooling == 'gated_attention':
            self.attention_net = Attn_Net_Gated(L=dim_emb, D=dim_attn, dropout=drop_rate)
        elif pooling == 'attention':
            self.attention_net = Attn_Net(L=dim_emb, D=dim_attn, dropout=drop_rate)
        else:
            self.attention_net = None

        # Residual branch for learning new knowledge from raw inputs
        if post_mil_layer == 'MLP':
            self.post_mil_layer = nn.Sequential(
                nn.Linear(dim_emb, dim_emb//4),
                nn.ReLU(),
                nn.Dropout(drop_rate),
                nn.Linear(dim_emb//4, dim_emb),
                nn.ReLU(),
            )
            print("[DeepMIL] added a post MIL layer (MLP).")
        else:
            self.post_mil_layer = nn.Identity()

        self.agg_method = pooling
        
        self.pred_head = nn.Linear(dim_emb, num_cls)

    def forward_attention_pooling(self, X, attn_mask=None):
        # X is B x K x C (K is the number of instances)
        # num_head = 1 for ABMIL
        A = self.attention_net(X)  # B x K x num_head 
        A = torch.transpose(A, -2, -1)  # B x num_head x K

        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        A = F.softmax(A, dim=-1)  # softmax over K (the last dim)
        M = torch.bmm(A, X).squeeze(dim=1) # B x num_head x C --> B x C

        return M, A.squeeze(dim=1) # B x C, B x K

    def forward_slide_representation(self, X):
        assert X.shape[0] == 1
        X = self.feat_proj(X)
        
        # global pooling: B x K x C -> B x C
        if 'attention' in self.agg_method:
            out_feat, attn = self.forward_attention_pooling(X)
            return out_feat, attn
        elif self.agg_method == 'mean':
            out_feat = torch.mean(X, dim=1)
            return out_feat
        elif self.agg_method == 'max':
            out_feat, _ = torch.max(X, dim=1)
            return out_feat
        else:
            raise NotImplementedError("Not Implemented!")

    def forward(self, X, ret_with_attn=False, ret_bag_feat=False):
        """
        X: initial bag features, with shape B x K x C
           where B = 1 for batch size, K is the instance size of this bag, and C is feature dimension.
        """
        slide_results = self.forward_slide_representation(X)
        if isinstance(slide_results, tuple):
            bag_feat, attn = slide_results
        else:
            bag_feat = slide_results
        
        out_feat = self.post_mil_layer(bag_feat)
        logit = self.pred_head(out_feat) # B x num_cls

        if ret_bag_feat:
            return logit, bag_feat.detach() # B x num_cls, B x dim_feat

        if ret_with_attn:
            return logit, attn.detach() # B x num_cls, B x K
        
        return logit


class DeepMILEncoder(nn.Module):
    """
    Deep Multiple Instance Learning Encoder.
    """
    def __init__(self, dim_in=768, dim_emb=512, dim_attn=384, num_feat_proj_layers=1, drop_rate=0.25, 
        pooling='attention', **kwargs):
        super().__init__()
        assert pooling in ['mean', 'max', 'attention', 'gated_attention']
        self.feat_proj = create_mlp(
            in_dim=dim_in,
            hid_dims=[dim_emb] * (num_feat_proj_layers - 1),
            dropout=drop_rate,
            out_dim=dim_emb,
            end_with_fc=False
        )
        if pooling == 'gated_attention':
            self.attention_net = Attn_Net_Gated(L=dim_emb, D=dim_attn, dropout=drop_rate)
        elif pooling == 'attention':
            self.attention_net = Attn_Net(L=dim_emb, D=dim_attn, dropout=drop_rate)
        else:
            self.attention_net = None
        self.agg_method = pooling
        # only used for possible pred_head construction
        self.cfg_pred_head = (dim_emb, kwargs['num_cls'])
        
    def forward_attention_pooling(self, X, attn_mask=None):
        # X is B x K x C (K is the number of instances)
        # num_head = 1 for ABMIL
        A = self.attention_net(X)  # B x K x num_head 
        A = torch.transpose(A, -2, -1)  # B x num_head x K

        if attn_mask is not None:
            A = A + (1 - attn_mask).unsqueeze(dim=1) * torch.finfo(A.dtype).min
        A = F.softmax(A, dim=-1)  # softmax over K (the last dim)
        M = torch.bmm(A, X).squeeze(dim=1) # B x num_head x C --> B x C

        return M, A.squeeze(dim=1) # B x C, B x K

    def forward_slide_representation(self, X):
        assert X.shape[0] == 1
        X = self.feat_proj(X)
        
        # global pooling: B x K x C -> B x C
        if 'attention' in self.agg_method:
            out_feat, attn = self.forward_attention_pooling(X)
            return out_feat, attn
        elif self.agg_method == 'mean':
            out_feat = torch.mean(X, dim=1)
            return out_feat
        elif self.agg_method == 'max':
            out_feat, _ = torch.max(X, dim=1)
            return out_feat
        else:
            raise NotImplementedError("Not Implemented!")

    def forward(self, X, ret_with_attn=False):
        """
        X: initial bag features, with shape B x K x C
           where B = 1 for batch size, K is the instance size of this bag, and C is feature dimension.
        """
        slide_results = self.forward_slide_representation(X)
        if isinstance(slide_results, tuple):
            bag_feat, attn = slide_results
        else:
            bag_feat = slide_results
        
        if ret_with_attn:
            return bag_feat, attn.detach() # B x C', B x K
        
        return bag_feat # B x C'
