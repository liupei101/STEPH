import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch import distributions as dist
from functorch import make_functional
from collections import OrderedDict

from model.layers import create_mlp
from model.merged_model import get_attr
from model.merged_model import MergedModel
from utils.type import StateDictType

EPS = 1e-9

class MixVecMergedModel(MergedModel):
    def __init__(
        self,
        compute_mix_weights='nn',
        initial_lambda_t=None,
        output_layer='relu',
        topk_task_vectors=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert 'ADA-mixvec' in self.method_compute_weights, "Expected ADA-mixvec in `method_compute_weights`"
        
        self.merge_weight = None
        print("[MixVecMergedModel] merge_weight is set to None; it is output by an adaptive network")
        
        self.topk_task_vectors = topk_task_vectors
        print(f"[MixVecMergedModel] topk_task_vectors = {topk_task_vectors}")

        self.target_task_vector = self._build_target_task_vector()
        
        dim_in, dim_emb, dim_out = 1536, 512, len(self.model_pool)
        drop_rate = 0.25

        # A shared MLP layer for Mean-MIL
        # <= 1: Linear | > 1 : Non-Linear
        num_emb_layers = 1
        self.regulate_layer_emb = create_mlp(
            in_dim=dim_in,
            hid_dims=[dim_emb] * (num_emb_layers - 1),
            dropout=drop_rate,
            out_dim=dim_emb,
            end_with_fc=False
        )

        # MLP head to compute input-conditional merging weights
        self.regulate_layer_prj = create_mlp(
            in_dim=dim_emb,
            hid_dims=[dim_emb],
            dropout=drop_rate,
            out_dim=dim_out,
            end_with_fc=True
        )

        # MLP head to compute input-conditional mixup ratio (if specified)
        # the output is converted to lambda_t by lambda_t = 1 / (1 + mixup_ratio)
        assert compute_mix_weights in ['nn', 'fnn', 'param', 'fix']
        self.method_compute_mix_weights = compute_mix_weights
        if compute_mix_weights == 'nn':
            self.vecmix_layer = create_mlp(
                in_dim=dim_emb,
                hid_dims=[dim_emb],
                dropout=drop_rate,
                out_dim=dim_out,
                end_with_fc=True
            )
        elif compute_mix_weights == 'fnn':
            self.vecmix_layer_emb = create_mlp(
                in_dim=dim_in,
                hid_dims=[dim_emb] * (num_emb_layers - 1),
                dropout=drop_rate,
                out_dim=dim_emb,
                end_with_fc=False
            )
            self.vecmix_layer = create_mlp(
                in_dim=dim_emb,
                hid_dims=[dim_emb],
                dropout=drop_rate,
                out_dim=dim_out,
                end_with_fc=True
            )
        else:
            if compute_mix_weights == 'param':
                self.vecmix_layer = nn.Parameter(
                    torch.zeros(dim_out, dtype=torch.float32),
                    requires_grad=True
                )
            else:
                assert initial_lambda_t is not None and len(initial_lambda_t) == dim_out
                initial_lambda_t = [max(EPS, lam) for lam in initial_lambda_t]
                initial_mix_ratios = [(1 - lam) / lam for lam in initial_lambda_t]
                self.vecmix_layer = nn.Parameter(
                    torch.tensor(initial_mix_ratios, dtype=torch.float32),
                    requires_grad=False
                )
            init_lambda_t = 1 / (1 + self.vecmix_layer.data.detach())
            print(f"[MixVecMergedModel] Mix weights are initialized to", init_lambda_t)

        if output_layer == 'sigmoid':
            # merging weight in [0, 1]
            self.output_layer = nn.Sigmoid()
        elif output_layer == 'softplus':
            # merging weight in [0, +inf]
            self.output_layer = nn.Softplus()
        elif output_layer == 'relu':
            # merging weight in [0, +inf]
            self.output_layer = nn.ReLU()
        else:
            raise NotImplementedError

        print(f"[MixVecMergedModel] compute_mix_weights = {compute_mix_weights}")
        print(f"[MixVecMergedModel] output_layer = {output_layer}")

    def apply_additional_task_vector(self, cur_model_params: StateDictType, merge_weight: Tensor):
        _merge_weight = merge_weight[0] # extract actual merging weights
        if self.topk_task_vectors is not None:
            _, sel_idx = torch.topk(_merge_weight, self.topk_task_vectors)
        else:
            sel_idx = [i for i in range(len(_merge_weight))]

        # NOTE: merge_weight with shape (2, dim_out)
        for _idx in sel_idx:
            weight, task_vector = merge_weight[:, _idx], self.task_vectors[_idx]
            task_vector_params = dict(task_vector.named_parameters())
            for i, name in enumerate(self.parameter_names):
                if name in self.parameter_names_to_merge:
                    w = weight[0] * weight[1]
                    task_transfer_vector = self.target_task_vector[name.replace('.', '_')] - task_vector_params[name]
                    cur_model_params[name] = cur_model_params[name] + w * task_transfer_vector

        return cur_model_params

    def get_merge_weight(self, X=None):
        # return a tensor with shape (2, dim_out)
        if X is None:
            null_out = torch.tensor([0, 0] * len(self.model_pool)).view(2, -1)
            return null_out

        with torch.no_grad():
            merge_weight = self.forward_regulate_layer(X)

        return merge_weight.detach.cpu()

    def forward_regulate_layer(self, X):
        _X = self.regulate_layer_emb(X) # [1, K, d] -> [1, K, d']
        mean_feat = torch.mean(_X, dim=1) # [1, K, d'] -> [1, d']

        logits = self.regulate_layer_prj(mean_feat) # [1, d'] -> [1, d_out]
        merge_weight = self.output_layer(logits) # [1, d_out] -> weights

        # here output vecmix_ratio: r = lamda_s / lambda_t = (1 - lambda_t) / lambda_t
        # so lambda_t = 1 / (1 + r), lambda_s = r / (1 + r)
        if self.method_compute_mix_weights == 'nn':
            vecmix_logits = self.vecmix_layer(mean_feat)
            vecmix_ratio = F.relu(vecmix_logits) # mixup weight > 0
        elif self.method_compute_mix_weights == 'fnn':
            _fnn_X = self.vecmix_layer_emb(X)
            fnn_mean_feat = torch.mean(_fnn_X, dim=1)
            vecmix_logits = self.vecmix_layer(fnn_mean_feat)
            vecmix_ratio = F.relu(vecmix_logits) # mixup weight > 0
        else:
            vecmix_ratio = self.vecmix_layer
            vecmix_ratio = vecmix_ratio.unsqueeze(0) # [1, d_out]
        # here convert ratio to lambda_t = 1 / (1 + r), i.e., the mixup coef of tau_t
        lambda_t = 1 / (1 + vecmix_ratio)

        return torch.cat([merge_weight, lambda_t], dim=0) # [2, d_out]

    def merge_and_unload(self):
        print("[MixVecMergedModel] merge_and_unload is not supported. Skipped...")
        return self

    def forward(self, *args, **kwargs):
        """
        Forward pass through the dynamically merged model.

        This method performs the forward pass by first ensuring the model parameters
        are merged according to the current merge weights, then applying the merged
        model to the input data.
        """
        X = args[0] # bag data with shape [1, K, d]
        merge_weight = self.forward_regulate_layer(X)

        forward_func, param = make_functional(self.target_model)
        _merged_params = self.merge_models(param, merge_weight, top_k=self.topk_task_vectors)

        _model_out = forward_func(_merged_params, *args, **kwargs)

        if self.task_ari_module_to_retrain is None or self.is_retrain_module_inside_model:
            model_out = _model_out
        else:
            # if retrain module is not inside the target model,
            # do forward pass here explicitly
            retrain_module = getattr(self, self.task_ari_module_to_retrain)
            model_out = retrain_module(_model_out)

        if self.training:
            return model_out, merge_weight

        return model_out
