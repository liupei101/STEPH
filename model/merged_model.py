from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Union

import torch
from torch import nn, Tensor
from functorch import make_functional
from collections import OrderedDict

from model.utils_merge import get_attr, get_module_from_model
from model.utils_merge import compute_task_vectors_with_iso_c
from model.utils_merge import compute_task_vectors_with_iso_cts
from model.utils_merge import compute_task_vectors_with_ties_merging
from model.model_pool import ModelPool
from model.layers import create_mlp
from utils.type import StateDictType, TorchModelType


__all__ = ["MergedModel", "DynamicMergedModel"]


class MergedModel(nn.Module, Generic[TorchModelType]):
    """
    A PyTorch module that dynamically merges multiple source models using learnable task-wise weights.

    This class implements a sophisticated model fusion approach where multiple task-specific models
    are combined with a target model using learnable weights. The fusion is performed
    using task vectors (differences between source and target models) that are weighted
    and applied to the source model's parameters.

    That is M_s_i + w_i * (M_t - M_s_i), where t is target, s_i is the i-th source model, 
    and w_i is the weight of model mixing.

    Args:
        model_pool (ModelPool): The ModelPool instance to provide the setting of target and source models.
            Defaults to None; otherwise, target_model, source_models, and source_model_init_coefs should
            be specified.
        null_model (TorchModelType): The null model (unfitted, only randomly intialized).
            Will be frozen and used as the M_null (theta_0) in merging.
        target_model (TorchModelType): The target model.
            Will be frozen and used as the target in merging.
        source_models (List[TorchModelType]): List of all source models.
            Must have the same architecture as the target model.
        source_model_init_coefs (List): Initial coefficients for each source model. Shape: (num_models,).
            These values become the starting point for learnable parameters.
        clamp_weights (bool, optional): Whether to clamp merge weights to [0, 1] range.
            Defaults to True. When True, ensures weights are non-negative and bounded.
        compute_weights (str): How to determine the weights for merging, one of ['FIX', 'ADA'].

    Attributes:
        merge_weight (nn.Parameter): Learnable weights (i.e., w_i).
        null_model (TorchModelType): The frozen null model.
        target_model (TorchModelType): The frozen target model.
        source_models (nn.ModuleList): The frozen source models.
    """
    def __init__(
        self,
        model_pool: Optional[ModelPool] = None,
        clamp_weights: bool = False,
        compute_weights: str = 'ADA',
        parameter_names_to_merge: List[str] = None,
        task_ari_module_to_retrain: Optional[str] = None,
        preprocess_task_vectors: Optional[str] = None,
        base_parameters_from: str = 'target',
        **kwargs
    ):
        super().__init__()
        self.model_pool = model_pool
        assert self.model_pool is not None
        self.clamp_weights = clamp_weights
        self.method_compute_weights = compute_weights
        self.kwargs = kwargs

        null_model = self.model_pool.load_null_model()
        target_model = self.model_pool.load_target_model()
        source_models = self.model_pool.to_model_list()
        source_model_init_coefs = self.model_pool.get_init_coef_list()

        # null: M = M_0 + \sigma_i w_i * (M_s_i - M_0), where there exists a s_i = t;
        # target: M = M_t + \sigma_i w_i * (M_s_i - M_0), where t is (or not) in {s_i}.
        assert base_parameters_from in ['null', 'target']
        self.base_parameters_from = base_parameters_from

        # setup models and task vectors
        self.null_model = null_model.requires_grad_(False)
        self.target_model = target_model.requires_grad_(False)
        self.target_model_name = self.model_pool.target_model_name
        self.parameter_names = [k for k, v in self.target_model.named_parameters()]
        # pre-compute task vector: \tau_s = M_s - M_0
        for m in source_models:
            for param_name in self.parameter_names:
                param_names = param_name.split(".")
                get_attr(m, param_names).data = get_attr(m, param_names) - get_attr(self.null_model, param_names)
        self.task_vectors = nn.ModuleList([m.requires_grad_(False) for m in source_models])
        self.source_model_names = self.model_pool.model_names
        
        # setup parameters to merge
        if parameter_names_to_merge is None:
            self.parameter_names_to_merge = self.parameter_names
        else:
            self.parameter_names_to_merge = parameter_names_to_merge
            for name in parameter_names_to_merge:
                assert name in self.parameter_names, f"Parameter name ({name}) is not found in target model"

        self.method_preprocess_task_vectors = preprocess_task_vectors
        if preprocess_task_vectors == 'iso_c':
            # project each tau_s onto the subspace shared by tau_t and tau_s
            target_task_vector = self._build_target_task_vector()
            self.task_vectors = compute_task_vectors_with_iso_c(
                target_task_vector,
                self.task_vectors,
                self.parameter_names_to_merge
            )
            print("[MergedModel] Important Action: task vectors changed by Iso-C")
        elif preprocess_task_vectors == 'iso_cts':
            # project each tau_s onto the subspace shared by tau_t and tau_s
            target_task_vector = self._build_target_task_vector()
            self.task_vectors = compute_task_vectors_with_iso_cts(
                target_task_vector,
                self.task_vectors,
                self.parameter_names_to_merge,
                common_space_fraction=0.8
            )
            print("[MergedModel] Important Action: task vectors changed by Iso-CTS")
        elif preprocess_task_vectors == 'ties_merging':
            target_task_vector = self._build_target_task_vector()
            self.task_vectors = compute_task_vectors_with_ties_merging(
                target_task_vector,
                self.task_vectors,
                self.parameter_names_to_merge
            )
            print("[MergedModel] Important Action: task vectors changed by ties_merging")
        else:
            pass 

        # setup merging weights
        task_wise_weight = torch.tensor(
            source_model_init_coefs, dtype=torch.float32
        )
        self.task_ari_module_to_retrain = task_ari_module_to_retrain
        self.is_retrain_module_inside_model = True
        if 'ADA' in self.method_compute_weights:
            # reset equal weights by default
            if self.method_compute_weights in ['ADA-default', 'ADA-mixvec']:
                task_wise_weight = 1 / len(self.source_model_names) * torch.ones_like(task_wise_weight)
                print(f"[MergedModel] {self.method_compute_weights}: initial task_wise_weight =", task_wise_weight)
            self.merge_weight = nn.Parameter(task_wise_weight, requires_grad=True)
            # setup parameters to train from scratch
            if task_ari_module_to_retrain is not None:
                for module_to_retrain in task_ari_module_to_retrain:
                    retrain_module, is_retrain_module_inside_model = get_module_from_model(self.target_model, module_to_retrain)
                    self.is_retrain_module_inside_model = self.is_retrain_module_inside_model and is_retrain_module_inside_model
                    setattr(self, module_to_retrain, retrain_module)
                    print(f"[MergedModel] set a new module to retrain: {module_to_retrain}")
        else:
            if self.method_compute_weights == 'FIX-avg':
                task_wise_weight = 1 / len(self.source_model_names) * torch.ones_like(task_wise_weight)
                print("[MergedModel] FIX-avg: fixed task_wise_weight =", task_wise_weight)
            self.merge_weight = nn.Parameter(task_wise_weight, requires_grad=False)
            if task_ari_module_to_retrain is not None:
                print(f"[MergedModel] Warning: task_ari_module_to_retrain={task_ari_module_to_retrain} has no effect if method_compute_weights is not based on ADA")
                self.task_ari_module_to_retrain = None

        print(f"[MergedModel] initialized a MergedModel with method_compute_weights = {compute_weights}")
        print(f"[MergedModel] parameter_names_to_merge = {self.parameter_names_to_merge}")
        print(f"[MergedModel] base_parameters_from = {self.base_parameters_from}")

    def get_model_pool(self):
        return self.model_pool

    def get_merge_weight(self):
        return self.merge_weight.detach().cpu()

    def merge_and_unload(self):
        """
        Merge models and return the final merged model.

        Warning:
            This method modifies the target_model's parameters in-place.
            The original target model parameters will be lost.
        """
        merged_state_dict = self.merge_models(return_type='dict')
        self.target_model.load_state_dict(merged_state_dict, strict=True)
        print("[MergedModel] Warning of calling merge_and_unload: the target_model's parameters are modified in-place so original ones are lost.")
        return self.target_model

    def _build_target_task_vector(self):
        dict_target_task_vector = OrderedDict()
        for param_name in self.parameter_names:
            if param_name in self.parameter_names_to_merge:
                _names = param_name.split(".")
                attr_name = param_name.replace('.', '_')
                dict_target_task_vector[attr_name] = nn.Parameter(
                    get_attr(self.target_model, _names) - get_attr(self.null_model, _names),
                    requires_grad=False
                )
        target_task_vector = nn.ParameterDict(dict_target_task_vector)
        return target_task_vector

    def reload_trainable_parameters(self, cur_model_params: StateDictType):
        """
        Reload the parameters to be trained (i.e., self.task_ari_module_to_retrain)
        """
        if not self.is_retrain_module_inside_model or self.task_ari_module_to_retrain is None:
            return cur_model_params

        _final_model_params = dict()
        for i, name in enumerate(self.parameter_names):
            for module_to_retrain in self.task_ari_module_to_retrain:
                if name.startswith(module_to_retrain):
                    retrain_module = getattr(self, module_to_retrain)
                    attr_name = name[len(module_to_retrain):]
                    if len(attr_name) > 0:
                        assert attr_name[0] == '.'
                        _final_param = getattr(retrain_module, attr_name[1:])
                    else:
                        _final_param = retrain_module
                else:
                    _final_param = cur_model_params[name]
                _final_model_params[name] = _final_param
        return _final_model_params

    def apply_additional_task_vector(self, cur_model_params: StateDictType, merge_weight: Tensor):
        return cur_model_params

    def on_merge_models_end(self, cur_model_params: StateDictType, merge_weight: Tensor):
        applied_model_params = self.apply_additional_task_vector(cur_model_params, merge_weight)
        _final_model_params = self.reload_trainable_parameters(applied_model_params)
        return _final_model_params

    def merge_models(
        self, 
        target_params: Optional[Union[tuple, StateDictType]] = None,
        merge_weight: Optional[Tensor] = None,
        top_k: Optional[int] = None,
        return_type: Optional[str] = None
    ):
        # get a tuple of target parameters
        if target_params is None:
            print("[MergedModel] merging models: target_params is not given so that of current target model will be used.")
            _target_params = tuple(self.target_model.parameters())
        elif isinstance(target_params, dict):
            _target_params = tuple(target_params[name] for name in self.parameter_names)
        else:
            _target_params = target_params

        if merge_weight is None:
            merge_weight = self.merge_weight
        if merge_weight is None:
            raise ValueError("Found merge_weight is None")
        if self.clamp_weights:
            merge_weight = torch.clamp(merge_weight, min=0.0, max=1.0)

        # get base parameters
        if self.base_parameters_from == 'null':
            null_named_params = dict(self.null_model.named_parameters())
            _new_model_params = dict({name: null_named_params[name] for name in self.parameter_names})
        elif self.base_parameters_from == 'target':
            _new_model_params = dict({name: _target_params[i] for i, name in enumerate(self.parameter_names)})
        else:
            NotImplementedError

        # NOTE: if base parameters are not from target, the following code will fill
        #       the parameters (NOT in the merging list) with those of target model
        if self.base_parameters_from != 'target':
            for i, name in enumerate(self.parameter_names):
                if name not in self.parameter_names_to_merge:
                    _new_model_params[name] = _target_params[i]

        # apply task vectors
        if merge_weight.ndim == 2 and merge_weight.shape[0] == 2:
            _merge_weight = merge_weight[0] # extract the actual merging weights
        else:
            assert merge_weight.shape[0] == len(self.task_vectors)
            _merge_weight = merge_weight
        if top_k is not None:
            _, sel_idx = torch.topk(_merge_weight, top_k)
        else:
            sel_idx = [i for i in range(len(_merge_weight))]
        for _idx in sel_idx:
            weight, task_vector = _merge_weight[_idx], self.task_vectors[_idx]
            task_vector_params = dict(task_vector.named_parameters())
            for i, name in enumerate(self.parameter_names):
                w = weight if name in self.parameter_names_to_merge else 0
                _new_model_params[name] = _new_model_params[name] + w * task_vector_params[name]

        # return a Dict (excute all steps on the end of model merging)
        _final_model_params = self.on_merge_models_end(_new_model_params, merge_weight)

        if return_type == 'dict' or (return_type is None and isinstance(target_params, dict)):
            final_model_params = _final_model_params
        else:
            final_model_params = tuple(_final_model_params[name] for name in self.parameter_names)

        return final_model_params

    def forward(self, *args, **kwargs):
        """
        Forward pass through the dynamically merged model.

        This method performs the forward pass by first ensuring the model parameters
        are merged according to the current merge weights, then applying the merged
        model to the input data.
        """
        forward_func, param = make_functional(self.target_model)
        _merged_params = self.merge_models(param)

        _model_out = forward_func(_merged_params, *args, **kwargs)

        if self.task_ari_module_to_retrain is None or self.is_retrain_module_inside_model:
            model_out = _model_out
        else:
            # if retrain module is not inside the target model,
            # do forward pass here explicitly
            retrain_module = getattr(self, self.task_ari_module_to_retrain)
            model_out = retrain_module(_model_out)

        if self.training:
            return model_out, self.merge_weight

        return model_out


class DynamicMergedModel(MergedModel):
    def __init__(
        self,
        output_layer='relu',
        **kwargs
    ):
        super().__init__(**kwargs)
        assert 'ADA-dynamic' in self.method_compute_weights, "Expected ADA-dynamic in `method_compute_weights`"
        self.merge_weight = None
        print("[DynamicMergedModel] merge_weight is set to None; it is output by an adaptive network")

        # Network to compute input-conditional merging weights
        dim_in, dim_emb, dim_out = 1536, 512, len(self.model_pool)
        drop_rate = 0.25
        # <= 1: Linear | > 1 : Non-Linear
        num_emb_layers = 1
        self.regulate_layer_emb = create_mlp(
            in_dim=dim_in,
            hid_dims=[dim_emb] * (num_emb_layers - 1),
            dropout=drop_rate,
            out_dim=dim_emb,
            end_with_fc=False
        )
        self.regulate_layer_prj = create_mlp(
            in_dim=dim_emb,
            hid_dims=[dim_emb],
            dropout=drop_rate,
            out_dim=dim_out,
            end_with_fc=True
        )

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

        print(f"[DynamicMergedModel] output_layer = {output_layer}")

    def get_merge_weight(self, X=None):
        if X is None:
            return torch.tensor([0] * len(self.model_pool))

        with torch.no_grad():
            merge_weight = self.forward_regulate_layer(X)

        return merge_weight.detach.cpu()

    def forward_regulate_layer(self, X):
        _X = self.regulate_layer_emb(X) # [1, K, d] -> [1, K, d']
        mean_feat = torch.mean(_X, dim=1) # [1, K, d'] -> [1, d']
        logits = self.regulate_layer_prj(mean_feat) # [1, d'] -> [1, d_out]
        merge_weight = self.output_layer(logits) # [1, d_out] -> weights

        return merge_weight.squeeze(0)

    def merge_and_unload(self):
        print("[DynamicMergedModel] merge_weight is None. Skipped...")
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
        _merged_params = self.merge_models(param, merge_weight)

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
