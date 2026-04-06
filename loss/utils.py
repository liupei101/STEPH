import inspect
from typing import Optional
from functools import partial
import torch
from torch import Tensor, nn

from . import loss_surv as SurvLoss
from . import loss_clf as ClfLoss
from .loss_clf import BinaryCrossEntropy, SoftTargetCrossEntropy


def load_loss(task, *args, **kws):
    if task in ['clf', 'sa', 'samtl']:
        assert 'loss_type' in kws, "The key of `loss_type` is not found in kws."
        loss_fn = dict()
        func_load_loss = load_clf_loss_func if task == 'clf' else load_surv_loss_func
        for loss_name in kws['loss_type']:
            loss_fn[loss_name] = func_load_loss(loss_name, **kws[loss_name])
        return loss_fn
    else:
        raise NotImplementedError(f"cannot recognize the task {task}.")

def load_clf_loss_func(loss_type:str, **loss_cfg):
    """
    loss_type (str): The name of the classification loss functions or classes defined in loss_clf.py
    loss_cfg (dict): The arguments to be specified for the loss function. 
    """
    if loss_type == 'BCE':
        target_loss_func = BinaryCrossEntropy(loss_cfg['smoothing'], target_threshold=loss_cfg['target_thresh'])
    elif loss_type == 'CE':
        target_loss_func = SoftTargetCrossEntropy(loss_cfg['smoothing'])
    else:
        loss_protype = getattr(ClfLoss, loss_type)
        if inspect.isclass(loss_protype):
            target_loss_func = loss_protype(**loss_cfg)
        elif inspect.isfunction(loss_protype):
            target_loss_func = partial(loss_protype, **loss_cfg)
        else:
            raise ValueError(f"{loss_type} is not found.")

    return target_loss_func

def load_surv_loss_func(loss_type:str, **loss_cfg):
    """
    loss_type (str): The name of the survival loss functions or classes defined in loss_surv.py
    loss_cfg (dict): The arguments of the survival function to be loaded. 
    """
    if loss_type == 'CE':
        target_loss_func = nn.CrossEntropyLoss()
    else:
        loss_protype = getattr(SurvLoss, loss_type)
        if inspect.isclass(loss_protype):
            target_loss_func = loss_protype(**loss_cfg)
        elif inspect.isfunction(loss_protype):
            target_loss_func = partial(loss_protype, **loss_cfg)
        else:
            raise ValueError(f"{loss_type} is not found.")

    return target_loss_func

def loss_reg_l1(coef):
    coef = .0 if coef is None else coef
    def func(model_params):
        if coef <= 1e-8:
            return 0.0
        else:
            return coef * sum([torch.abs(W).sum() for W in model_params])
    return func

def entropy_loss(logits: Tensor, eps: float = 1e-8, reduction=True) -> Tensor:
    """
    Compute the entropy loss of a set of logits.

    Args:
        logits (Tensor): The logits to compute the entropy loss of.
        eps (float): A small value to avoid log(0). Default is 1e-8.

    Returns:
        Tensor: The entropy loss of the logits.
    """
    # Ensure the logits tensor has 2 dimensions
    assert (
        logits.dim() == 2
    ), f"Expected logits to have 2 dimensions, found {logits.dim()}, {logits.size()=}"

    # Compute the softmax probabilities
    probs = torch.softmax(logits, dim=-1)

    # Compute the entropy loss
    if not reduction:
        return -torch.sum(probs * torch.log(probs + eps), dim=-1)
    
    return -torch.sum(probs * torch.log(probs + eps), dim=-1).mean()

def entropy_loss_with_hazards(
        hazards: Tensor,
        eps: float = 1e-8,
        reduction: bool = True,
        max_time_bins: Optional[int] = None
    ) -> Tensor:
    # Ensure the hazards tensor has 2 dimensions
    assert (
        hazards.dim() == 2
    ), f"Expected hazards to have 2 dimensions, found {hazards.dim()}."

    # convert it to survival function
    S = torch.cumprod(1 - hazards, dim=1)
    S_padded = torch.hstack((torch.ones((len(S), 1), dtype=torch.float32, device=S.device), S))
    
    # measure the entropy in FHT (first hitting time) distribution
    probs = hazards * S_padded[:, :-1]

    if max_time_bins is not None:
        pred_probs = probs[:, :max_time_bins] # [N, max_time_bins]
        rest_probs = 1 - torch.sum(pred_probs, dim=-1, keepdim=True) # [N, 1]
        final_probs = torch.hstack((pred_probs, rest_probs))
    else:
        final_probs = probs
    
    # Compute the entropy loss
    if not reduction:
        return -torch.sum(probs * torch.log(probs + eps), dim=-1)

    return -torch.sum(probs * torch.log(probs + eps), dim=-1).mean()

def bernoulli_entropy_loss_with_hazards(
        hazards: Tensor,
        eps: float = 1e-8,
        reduction: bool = True,
        max_time_bins: Optional[int] = None
    ) -> Tensor:
    # Ensure the hazards tensor has 2 dimensions
    assert (
        hazards.dim() == 2
    ), f"Expected hazards to have 2 dimensions, found {hazards.dim()}."

    if max_time_bins is not None:
        final_hazards = hazards[:, :max_time_bins] # [N, max_time_bins]
    else:
        final_hazards = hazards

    h = final_hazards.clamp(eps, 1 - eps)
    entropy_t = - (h * torch.log(h) + (1 - h) * torch.log(1 - h))
    # mean over the bernoulli entropy of h(t_i)
    entropy = torch.mean(entropy_t, dim=1)

    if not reduction:
        return entropy

    return entropy.mean()
