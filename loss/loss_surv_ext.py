"""Earth Mover Distance (Wasserstein distance p=1)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def wasserstein_loss(pred_dist, target_dist):
    return cdf_loss(pred_dist, target_dist, p=1)

def cdf_loss(pred_dist, target_dist, p=1, normalize_dist=True, ret_raw=False):
    assert pred_dist.shape == target_dist.shape, "Two input shapes do not match."
    if normalize_dist:
        pred_dist = pred_dist / (torch.sum(pred_dist, dim=-1, keepdim=True) + 1e-14)
        target_dist = target_dist / (torch.sum(target_dist, dim=-1, keepdim=True) + 1e-14)

    # make cdf with cumsum
    cdf_pred_dist = torch.cumsum(pred_dist, dim=-1)
    cdf_target_dist = torch.cumsum(target_dist, dim=-1)

    if p == 1:
        cdf_distance = torch.sum(torch.abs((cdf_pred_dist - cdf_target_dist)), dim=-1)
    elif p == 2:
        if not ret_raw:
            cdf_distance = torch.sqrt(torch.sum(torch.pow((cdf_pred_dist - cdf_target_dist), 2), dim=-1))
        else:
            cdf_distance = torch.sum(torch.pow((cdf_pred_dist - cdf_target_dist), 2), dim=-1)
    else:
        if not ret_raw:
            cdf_distance = torch.pow(torch.sum(torch.pow(torch.abs(cdf_pred_dist - cdf_target_dist), p), dim=-1), 1 / p)
        else:
            cdf_distance = torch.sum(torch.pow(torch.abs(cdf_pred_dist - cdf_target_dist), p), dim=-1)

    return cdf_distance

def convert_survival_label(t, e, n_bins):
    t, e = t.view(-1, 1), e.view(-1, 1)
    bsz = t.shape[0]
    t_vector = torch.full(
        (bsz, n_bins), 0,
        device=t.device, dtype=t.dtype
    ).scatter_(1, t, 1)
    
    for i in range(bsz):
        loc = t[i, 0] + 1
        if loc < n_bins:
            t_vector[i, loc:] = t_vector[i, loc:] + (1 - e[i, 0])

    return t_vector


class SurvEMD(nn.Module):
    """
    Earth Mover Distance^2 (Wasserstein distance p=2) for ordinal survival analysis.
    """
    def __init__(self, p=2, raw_distance=True, reduction='mean', **kws):
        super().__init__()
        self.p = p
        self.raw_distance = raw_distance
        self.reduction = reduction
        assert reduction in ['mean', 'sum', 'none']
        print(f"[SurvEMD] initialized a SurvEMD loss with p = {p}, raw_distance = {raw_distance}, and reduction = {reduction}.")

    def forward(self, y_hat, t, e, cur_logit_scale=10.0):
        """
        y_hat: torch.FloatTensor() with shape of [B, MAX_T], converted by softmax.
        t: torch.LongTensor() with shape of [B, ] or [B, 1]. It's a discrete time label.
        e: torch.FloatTensor() with shape of [B, ] or [B, 1]. 
            e = 1 for uncensored samples (with event), 
            e = 0 for censored samples (without event).
        cur_logit_scale: should be the value after applying self.logit_scale.exp().
        """
        bsz, n_bins = y_hat.shape[0], y_hat.shape[-1]
        
        if isinstance(cur_logit_scale, torch.Tensor):
            _logit_scale = cur_logit_scale.detach()
        else:
            _logit_scale = cur_logit_scale

        t = t.view(-1, 1).long()
        e = e.view(-1, 1).long()
        # convert time-to-event label
        target = convert_survival_label(t, e, n_bins) # [bsz, n_bins]
        target_dist = torch.softmax((2 * target - 1) * _logit_scale, dim=-1)

        # convert the predicted y_hat
        pred = (1 - e) * ((1 - target) * y_hat + target * _logit_scale) + e * y_hat
        pred_dist = torch.softmax(pred, dim=-1) # [bsz, n_bins]

        # [bsz, n_bins] <==> [bsz, n_bins]
        loss = cdf_loss(
            pred_dist, target_dist, 
            p=self.p, 
            normalize_dist=False, 
            ret_raw=self.raw_distance
        ) # [bsz, ]

        if self.reduction == 'mean': # default
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss