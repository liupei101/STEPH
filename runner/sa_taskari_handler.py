import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
import wandb
from tqdm import tqdm

from .sa_handler import SAHandler
from utils.func import load_cfg_src_ckpt, fetch_kws
from utils.func import seed_generator, seed_worker
from utils.func import add_prefix_to_filename
from utils.func_config import infer_parameter_names
from utils.func_config import read_initial_lambda_t
from loss.utils import entropy_loss_with_hazards
from loss.utils import bernoulli_entropy_loss_with_hazards
from model.utils import load_model
from model.model_pool import ModelPool
from model.merged_model import MergedModel, DynamicMergedModel
from model.merged_model_mixup import MixVecMergedModel


def construct_merged_model(model_pool: ModelPool, cfg):
    if 'task_ari_param_to_merge' not in cfg or cfg['task_ari_param_to_merge'] is None:
        parameter_names_to_merge = None
    else:
        func_infer_parameter_names = infer_parameter_names
        parameter_names_to_merge = func_infer_parameter_names(cfg['task_ari_param_to_merge'])

    if 'task_ari_param_to_exclude_norm' in cfg and cfg['task_ari_param_to_exclude_norm'] is False:
        parameter_names_to_merge = parameter_names_to_merge
    else:
        parameter_names_to_merge = [n for n in parameter_names_to_merge if 'norm.' not in n]
        print("[INFO] parameter names with norm are excluded")

    if 'task_ari_module_to_retrain' not in cfg or cfg['task_ari_module_to_retrain'] is None:
        task_ari_module_to_retrain = None
    else:
        task_ari_module_to_retrain = cfg['task_ari_module_to_retrain']
        if task_ari_module_to_retrain == 'pred_head':
            if cfg['deepmil_network'] == 'ABMIL':
                task_ari_module_to_retrain = ['pred_head']
                print("[INFO] ABMIL: set `task_ari_module_to_retrain` to (pred_head)")
            else:
                pass

    if 'task_ari_preprocess_task_vectors' not in cfg:
        task_ari_preprocess_task_vectors = None
    else:
        task_ari_preprocess_task_vectors = cfg['task_ari_preprocess_task_vectors']

    if 'mixvec' in cfg['task_ari_compute_weights']:
        if cfg['task_ari_compute_weights'] == 'ADA-mixvec':
            model_mixvec_class = MixVecMergedModel
        else:
            model_mixvec_class = None
        initial_lambda_t = read_initial_lambda_t(cfg)
        module = model_mixvec_class(
            model_pool=model_pool,
            clamp_weights=cfg['task_ari_clamp_weights'],
            compute_weights=cfg['task_ari_compute_weights'],
            parameter_names_to_merge=parameter_names_to_merge,
            task_ari_module_to_retrain=task_ari_module_to_retrain,
            preprocess_task_vectors=task_ari_preprocess_task_vectors,
            base_parameters_from=cfg['base_parameters_from'],
            output_layer=cfg['task_ari_output_layer'],
            compute_mix_weights=cfg['task_ari_compute_mix_weights'],
            initial_lambda_t=initial_lambda_t,
            topk_task_vectors=cfg['task_ari_topk_task_vectors'],
        )
    elif cfg['task_ari_compute_weights'] == 'ADA-dynamic':
        module = DynamicMergedModel(
            model_pool=model_pool,
            clamp_weights=cfg['task_ari_clamp_weights'],
            compute_weights=cfg['task_ari_compute_weights'],
            parameter_names_to_merge=parameter_names_to_merge,
            task_ari_module_to_retrain=task_ari_module_to_retrain,
            preprocess_task_vectors=task_ari_preprocess_task_vectors,
            base_parameters_from=cfg['base_parameters_from'],
            output_layer=cfg['task_ari_output_layer'],
        )
    else:
        module = MergedModel(
            model_pool=model_pool,
            clamp_weights=cfg['task_ari_clamp_weights'],
            compute_weights=cfg['task_ari_compute_weights'],
            parameter_names_to_merge=parameter_names_to_merge,
            task_ari_module_to_retrain=task_ari_module_to_retrain,
            preprocess_task_vectors=task_ari_preprocess_task_vectors,
            base_parameters_from=cfg['base_parameters_from'],
        )
    return module


class SATaskAriHandler(SAHandler):
    """
    This class handles the initialization (based on task arithmetic) and testing 
    of SA (Survival Analysis) models for WSIs.
    """
    def __init__(self, cfg):
        assert 'task_ari' in cfg and cfg['task_ari'] is True, "Expected `task_ari` = True."
        if 'loss_merge_weight' not in cfg:
            cfg['loss_merge_weight'] = 0.0
            print("[SATaskAriHandler] init: not found `loss_merge_weight` in cfg, set it to zero.")
        if 'loss_mix_weight' not in cfg:
            cfg['loss_mix_weight'] = 0.0
            print("[SATaskAriHandler] init: not found `loss_mix_weight` in cfg, set it to zero.")
        # run setup of cuda, seed, path, model, loss, optimizer
        # LR scheduler, evaluator, and evaluation metrics with 
        # the functions written to override those base ones. 
        super().__init__(cfg)

    @staticmethod
    def func_load_model(cfg):
        arch = cfg['arch']
        arch_cfg = fetch_kws(cfg, prefix=arch.lower())
        # This is the base model architecture used to train all task-specific models
        base_model = load_model(cfg['arch'], **arch_cfg)

        path_trg_ckpt = cfg['task_ari_trg_ckpt_path']
        excluded_datasets = cfg['dataset_name'] if 'task_ari_src_exlude_target' in cfg and cfg['task_ari_src_exlude_target'] is True else None
        print(f"[INFO] model setup: `task_ari_src_exlude_target` = {excluded_datasets}")
        cfg_src_ckpt = load_cfg_src_ckpt(
            cfg['task_ari_src_ckpt_path'].format("{}", cfg['task_ari_src_data_split_fold']), 
            datasets=cfg['task_ari_src_dataset_name'],
            coefficients=cfg['task_ari_src_init_coef'],
            excluded_datasets=excluded_datasets
        )
        # use the target model from the same fold to prevent data leakage
        if cfg['dataset_name'] in cfg_src_ckpt:
            cfg_src_ckpt[cfg['dataset_name']]['path_ckpt'] = path_trg_ckpt

        cfg_null_ckpt = {'name': 'null', 'path_ckpt': cfg['task_ari_null_ckpt_path']}
        cfg_trg_ckpt = {'name': cfg['dataset_name'], 'path_ckpt': path_trg_ckpt}
        model_pool = ModelPool(base_model, cfg_null_ckpt, cfg_trg_ckpt, cfg_src_ckpt)

        print(f"[SATaskAriHandler] model setup: it will construct a task arithmetic-based new model | target = {cfg_trg_ckpt}, source = {cfg_src_ckpt}.")
        new_merged_model = construct_merged_model(model_pool, cfg)

        return new_merged_model

    def calc_merge_weight_loss(self, merge_weight):
        z_loss = torch.logsumexp(merge_weight, dim=0)
        z_loss = torch.square(z_loss)

        if 'task_ari_topk_task_vectors' in self.cfg and self.cfg['task_ari_topk_task_vectors'] is not None:
            _, sel_idx = torch.topk(merge_weight, self.cfg['task_ari_topk_task_vectors'])
            mask_1 = F.one_hot(sel_idx, len(merge_weight)) # (topk, n_experts)
            density_1 = mask_1.float().mean(0) # (n_experts, )
            balance_loss = (merge_weight * density_1).sum()
            loss = z_loss + balance_loss
        else:
            loss = z_loss

        return loss

    def calc_mix_weight_loss(self, mix_weight, merge_weight):
        if 'task_ari_topk_task_vectors' in self.cfg and self.cfg['task_ari_topk_task_vectors'] is not None:
            _, sel_idx = torch.topk(merge_weight, self.cfg['task_ari_topk_task_vectors'])
            return torch.square(mix_weight[sel_idx]).mean()

        return torch.square(mix_weight).mean()

    def _update_network(self, xs, ys):
        """
        update the networks intialized from the MergeModel class
        """
        if isinstance(self.net, MixVecMergedModel) or isinstance(self.net, DefMixVecMergedModel):
            val_loss, val_preds = self._update_mixvec_network(xs, ys)
        else:
            val_loss, val_preds = self._update_normal_network(xs, ys)
        return val_loss, val_preds

    def _update_mixvec_network(self, xs, ys):
        n_sample = len(xs)
        y_hat = []
        batch_merge_weights = 0
        merge_weight_loss, mix_weight_loss = 0.0, 0.0

        for i in range(n_sample):
            X, ext_data = xs[i]
            X = X.cuda()
            pred, weights = self.net(X)
            y_hat.append(pred)
            assert weights.ndim == 2 and weights.shape[0] == 2
            # extract the actual merging weights
            merge_weight_loss += self.calc_merge_weight_loss(weights[0])
            mix_weight_loss += self.calc_mix_weight_loss(weights[1], weights[0])
            batch_merge_weights += weights.detach().cpu()

        self.optimizer.zero_grad()

        bag_preds = torch.cat(y_hat, dim=0) # [B, num_cls]
        bag_label = torch.cat(ys, dim=0) # [B, 2]
        pred_loss = self.calc_objective_loss(bag_preds, bag_label)

        aux_loss = self.cfg['loss_merge_weight'] * merge_weight_loss / n_sample
        aux_loss += self.cfg['loss_mix_weight'] * mix_weight_loss / n_sample
        pred_loss += aux_loss

        wandb.log({'train/aux_merge_weight_loss': merge_weight_loss.item() / n_sample})
        wandb.log({'train/aux_mix_weight_loss': mix_weight_loss.item() / n_sample})
        wandb.log({'train/aux_loss': aux_loss.item()})
        self._wandb_log_merging_weights(batch_merge_weights / n_sample)

        if isinstance(pred_loss, torch.Tensor) and pred_loss.requires_grad:
            pred_loss.backward()
            self.optimizer.step()
            self.steplr.step()
            val_loss = pred_loss.item()
        else:
            print("[batch train] warning: loss is not evaluated; skipped this batch training.")
            val_loss = 0

        val_preds = bag_preds.detach().cpu()
        return val_loss, val_preds

    def _update_normal_network(self, xs, ys):
        n_sample = len(xs)
        y_hat = []
        batch_merge_weights = 0
        merge_weight_loss = 0.0

        for i in range(n_sample):
            X, ext_data = xs[i]
            X = X.cuda()
            pred, weights = self.net(X)
            y_hat.append(pred)
            merge_weight_loss += self.calc_merge_weight_loss(weights)
            batch_merge_weights += weights.detach().cpu()

        self.optimizer.zero_grad()

        bag_preds = torch.cat(y_hat, dim=0) # [B, num_cls]
        bag_label = torch.cat(ys, dim=0) # [B, 2]
        pred_loss = self.calc_objective_loss(bag_preds, bag_label)

        aux_loss = self.cfg['loss_merge_weight'] * merge_weight_loss / n_sample
        pred_loss += aux_loss

        wandb.log({'train/aux_merge_weight_loss': merge_weight_loss.item() / n_sample})
        wandb.log({'train/aux_loss': aux_loss.item()})
        self._wandb_log_merging_weights(batch_merge_weights / n_sample)

        if isinstance(pred_loss, torch.Tensor) and pred_loss.requires_grad:
            pred_loss.backward()
            self.optimizer.step()
            self.steplr.step()
            val_loss = pred_loss.item()
        else:
            print("[batch train] warning: loss is not evaluated; skipped this batch training.")
            val_loss = 0

        val_preds = bag_preds.detach().cpu()
        return val_loss, val_preds

    def _wandb_log_merging_weights(self, weights=None):
        # current model is a MergedModel
        model_pool = self.net.get_model_pool()
        if weights is None:
            weights = self.net.get_merge_weight()
        if weights.ndim == 2 and weights.shape[0] == 2:
            wandb.log({f"TaskAri/merge_weight/{name}": w.item() for name, w in zip(model_pool.model_names, weights[0])})
            wandb.log({f"TaskAri/mix_weight/{name}": w.item() for name, w in zip(model_pool.model_names, weights[1])})
        else:
            wandb.log({f"TaskAri/merge_weight/{name}": w.item() for name, w in zip(model_pool.model_names, weights)})
