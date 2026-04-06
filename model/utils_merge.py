from typing import List
import copy
from collections import OrderedDict
import torch
from torch import nn, Tensor

from model.deepmil import DeepMIL, DeepMILEncoder


def get_attr(obj, names: List[str]):
    if len(names) == 1:
        return getattr(obj, names[0])
    else:
        return get_attr(getattr(obj, names[0]), names[1:])

def get_module_from_model(cur_model, module_name):
    if isinstance(cur_model, DeepMIL):
        for k, v in cur_model.named_parameters():
            if module_name == 'pred_head' and k == 'pred_head.weight':
                num_cls, dim_emb = tuple(v.shape)
                module = nn.Linear(dim_emb, num_cls)
                return module, True
    elif isinstance(cur_model, DeepMILEncoder):
        dim_emb, num_cls = cur_model.cfg_pred_head
        module = nn.Linear(dim_emb, num_cls)
        return module, False
    else:
        raise NotImplementedError("Only support for DeepMIL and DeepMILEncoder")

########################################################################################
#                        Iso-C (ICML 2025)
#  adapted from Github: iso-merging/blob/main/src/utils/iso.py
########################################################################################
@torch.no_grad()
def compute_task_vectors_with_iso_c(
    target_task_vector,
    task_vectors,
    parameter_names_to_project
):
    print("[INFO] start to compute task vectors with Iso-C...")
    tvs_params = [dict(tv.named_parameters()) for tv in task_vectors]
    for key in parameter_names_to_project:
        print(f"[INFO] processing a layer ({key})")
        
        tv_target = target_task_vector[key.replace('.', '_')].data

        for i, _tv in enumerate(tvs_params):
            tv_source = _tv[key].data

            # task vector merged from target and each source
            mer_tv = tv_target + tv_source

            if 'weight' not in key:
                merged_value = mer_tv / 2
            else:
                U, S, V = torch.linalg.svd(mer_tv, full_matrices=False)
                S_mean = torch.ones_like(S) * S.mean()

                merged_value = torch.linalg.multi_dot(
                    (U, torch.diag(S_mean), V)
                )

            # write it back to the task vector
            get_attr(task_vectors[i], key.split(".")).data = merged_value

    return task_vectors

@torch.no_grad()
def compute_task_vectors_with_iso_cts(
    target_task_vector,
    task_vectors,
    parameter_names_to_project,
    common_space_fraction=0.8
):
    print("[INFO] start to compute task vectors with Iso-CTS...")
    tvs_params = [dict(tv.named_parameters()) for tv in task_vectors]
    for key in parameter_names_to_project:
        print(f"[INFO] processing a layer ({key})")
        
        tv_target = target_task_vector[key.replace('.', '_')].data

        for i, _tv in enumerate(tvs_params):
            tv_source = _tv[key].data
            shape_ = tv_source.shape

            # task vector merged from target and each source
            combined_w = tv_target + tv_source
            num_tasks = 2

            if 'weight' not in key:
                merged_value = combined_w / 2
            else:
                ### Calculate the common space size (making sure that task specific space is equally divisible) ###
                common_space_index_s = int(min(shape_) * common_space_fraction)
                _task_specific_total_space_index_s = round((min(shape_) - common_space_index_s) / num_tasks) * num_tasks
                common_space_index_s = min(shape_) - _task_specific_total_space_index_s

                u, s, v = torch.linalg.svd(combined_w, full_matrices=False)
                common_space_u = u[:, :common_space_index_s]
                common_space_s = s[:common_space_index_s]
                common_space_v = v[:common_space_index_s, :]
                ###################################################################
                
                ### Calculate task specific space ###
                n_dims_per_task = int((min(shape_) - common_space_index_s) / num_tasks)
                cur_tvs = [tv_target, tv_source]
                for i, cur_tv in enumerate(cur_tvs):
                    w = cur_tv

                    # calculate the projection onto task specific space to remove the common space
                    w_ts = w - common_space_u @ common_space_u.T @ w
                    u_ts, s_ts, v_ts = torch.linalg.svd(w_ts, full_matrices=False)            
                    
                    if i == 0:
                        combined_space_u = torch.zeros_like(u_ts)
                        combined_space_s = torch.zeros_like(s_ts)
                        combined_space_v = torch.zeros_like(v_ts)
                        
                    combined_space_u[:, i * n_dims_per_task : (i + 1) * n_dims_per_task] = u_ts[:, :n_dims_per_task]
                    combined_space_s[i * n_dims_per_task : (i + 1) * n_dims_per_task] = s_ts[:n_dims_per_task]
                    combined_space_v[i * n_dims_per_task : (i + 1) * n_dims_per_task, :] = v_ts[:n_dims_per_task, :]
                ###################################################################
                
                combined_space_u[:, num_tasks * n_dims_per_task : num_tasks * n_dims_per_task + common_space_index_s] = common_space_u
                combined_space_s[num_tasks * n_dims_per_task : num_tasks * n_dims_per_task + common_space_index_s] = common_space_s
                combined_space_v[num_tasks * n_dims_per_task : num_tasks * n_dims_per_task + common_space_index_s, :] = common_space_v
                
                ### Orthogonalize combined_space_u and combined_space_v ###
                try:
                    u_combined_space_u, s_combined_space_u, v_combined_space_u = torch.linalg.svd(combined_space_u, full_matrices=False)
                    combined_space_u = u_combined_space_u @ v_combined_space_u
                except Exception as e:
                    print("Error occurs in the SVD of combined_space_u: undo SVD and use the original combined_space_u")
                try:
                    u_combined_space_v, s_combined_space_v, v_combined_space_v = torch.linalg.svd(combined_space_v, full_matrices=False)
                    combined_space_v = u_combined_space_v @ v_combined_space_v
                except Exception as e:
                    print("Error occurs in the SVD of combined_space_v: undo SVD and use the original combined_space_v")
                ###################################################################
                
                combined_space_s = torch.ones_like(combined_space_s) * combined_space_s.mean()
                        
                merged_value = torch.linalg.multi_dot(
                    (
                        combined_space_u,
                        torch.diag(combined_space_s),
                        combined_space_v,
                    )
                )

            # write it back to the task vector
            get_attr(task_vectors[i], key.split(".")).data = merged_value

    return task_vectors

########################################################################################
#                        TIES-Merging (NeurIPS 2023)
########################################################################################
def topk_values_mask(M, K=0.7, return_mask=False):
    if K > 1:
        K /= 100

    original_shape = M.shape
    if M.dim() == 1:
        M = M.unsqueeze(0)

    n, d = M.shape
    k = int(d * K)
    k = d - k  # Keep top k elements instead of bottom k elements

    # Find the k-th smallest element by magnitude for each row
    kth_values, _ = M.abs().kthvalue(k, dim=1, keepdim=True)
    # Create a mask tensor with True for the top k elements in each row
    mask = M.abs() >= kth_values
    final_mask = mask.squeeze() if original_shape == M.squeeze().shape else mask

    if return_mask:
        return M * final_mask, final_mask.float().mean(dim=1), final_mask
    return M * final_mask, final_mask.float().mean(dim=1)

def resolve_zero_signs(sign_to_mult, method="majority"):
    majority_sign = torch.sign(sign_to_mult.sum())

    if method == "majority":
        sign_to_mult[sign_to_mult == 0] = majority_sign
    elif method == "minority":
        sign_to_mult[sign_to_mult == 0] = -1 * majority_sign
    return sign_to_mult

def resolve_sign(v: Tensor):
    sign_to_mult = torch.sign(v.sum(dim=0))
    sign_to_mult = resolve_zero_signs(sign_to_mult, "majority")
    return sign_to_mult

def disjoint_merge(v: Tensor, merge_func: str, sign_to_mult):
    merge_func = merge_func.split("-")[-1]

    # If sign is provided then we select the corresponding entries and aggregate.
    if sign_to_mult is not None:
        rows_to_keep = torch.where(sign_to_mult.unsqueeze(0) > 0, v > 0, v < 0)
        selected_entries = v * rows_to_keep
    # Else we select all non-zero entries and aggregate.
    else:
        rows_to_keep = v != 0
        selected_entries = v * rows_to_keep

    if merge_func == "mean":
        non_zero_counts = (selected_entries != 0).sum(dim=0).float()
        disjoint_aggs = torch.sum(selected_entries, dim=0) / torch.clamp(
            non_zero_counts, min=1
        )
    elif merge_func == "sum":
        disjoint_aggs = torch.sum(selected_entries, dim=0)
    elif merge_func == "max":
        disjoint_aggs = selected_entries.abs().max(dim=0)[0]
        disjoint_aggs *= sign_to_mult
    else:
        raise ValueError(f"Merge method {merge_func} is not defined.")

    return disjoint_aggs

def ties_merging(flat_task_checks, reset_thresh=None, merge_func="sum"):
    all_checks = flat_task_checks.clone()
    updated_checks, *_ = topk_values_mask(all_checks, K=reset_thresh, return_mask=False)
    print("RESOLVING SIGN")
    final_signs = resolve_sign(updated_checks)
    assert final_signs is not None

    print(f"Disjoint AGGREGATION: {merge_func}")
    merged_tv = disjoint_merge(updated_checks, merge_func, final_signs)

    return merged_tv

def parameter_to_vector(list_tensors):
    return nn.utils.parameters_to_vector(
        [value.reshape(-1) for value in list_tensors]
    )

def vector_to_parameter(flat_vector, ref_state_tuple):
    reference_tuple = copy.deepcopy(ref_state_tuple)
    sorted_reference_dict = OrderedDict(reference_tuple)
    # create a shared state dict using the reference dict
    nn.utils.vector_to_parameters(flat_vector, sorted_reference_dict.values())
    return sorted_reference_dict

@torch.no_grad()
def compute_task_vectors_with_ties_merging(
    target_task_vector,
    task_vectors,
    parameter_names_to_merge
):
    print("[INFO] start to merge each task vector with the target using TIES-Merging...")

    flat_target_tv = parameter_to_vector([target_task_vector[k.replace('.', '_')].data for k in parameter_names_to_merge])

    for i in range(len(task_vectors)):
        tv_params = dict(task_vectors[i].named_parameters())
        flat_source_tv = parameter_to_vector([tv_params[k].data for k in parameter_names_to_merge])
        
        tv_flat_checks = torch.vstack([flat_target_tv, flat_source_tv])
        merged_tv = ties_merging(tv_flat_checks, reset_thresh=20, merge_func='sum')

        # write the merged tv back to the task vector
        ref_state_tuple = [(k, tv_params[k].data) for k in parameter_names_to_merge]
        merged_state_dict = vector_to_parameter(merged_tv, ref_state_tuple)
        for k in parameter_names_to_merge:
            get_attr(task_vectors[i], k.split(".")).data = merged_state_dict[k]

    return task_vectors

########################################################################################
#                                Task Vector Projection
########################################################################################
@torch.no_grad()
def compute_rank(S, rank_threshold=0.95):
    # S: torch.Tensor of singular values with shape (n,)
    S_squared = S.pow(2)
    norm_ratio = torch.sqrt(torch.cumsum(S_squared / S_squared.sum(), dim=0))
    rank = torch.argmax((norm_ratio > rank_threshold).float())
    return rank + 1

@torch.no_grad()
def project_task_vectors_onto_shared_space(
    target_task_vector,
    task_vectors,
    parameter_names_to_project,
    rank_threshold=0.95
):
    print("[INFO] start to project task vectors onto shared space...")
    tvs_params = [dict(tv.named_parameters()) for tv in task_vectors]
    for key in parameter_names_to_project:
        if 'weight' not in key:
            print(f"[INFO] skipped a layer ({key})")
            continue
        print(f"[INFO] projecting a layer ({key})")
        
        # compute SVD for tau_t
        tv_target = target_task_vector[key.replace('.', '_')].data

        for i, _tv in enumerate(tvs_params):
            tv_source = _tv[key].data
            # construct the common subspace of tau_t and tau_s
            mer_tv = tv_target + tv_source
            U_cm, S_cm, V_cm = torch.linalg.svd(mer_tv, full_matrices=False)

            # project tau_s onto the common subspace
            proj_tv_onto_share = torch.linalg.multi_dot((U_cm, U_cm.T, tv_source))

            # write it to the task vector
            get_attr(task_vectors[i], key.split(".")).data = proj_tv_onto_share

    return task_vectors

@torch.no_grad()
def project_task_vectors_onto_shared_space_v0(
    target_task_vector, task_vectors,
    parameter_names_to_project,
    rank_threshold=0.95,
    optimize_shared_space=False
):
    print("[INFO] start to project task vectors onto shared space...")
    tvs_params = [dict(tv.named_parameters()) for tv in task_vectors]
    for key in parameter_names_to_project:
        if 'weight' not in key:
            print(f"[INFO] skipped a layer ({key})")
            continue
        print(f"[INFO] projecting a layer ({key})")
        
        # compute SVD for tau_t
        tv_target = target_task_vector[key.replace('.', '_')].data
        U, S, V = torch.linalg.svd(tv_target, full_matrices=False)
        sum_rel_rank = compute_rank(S, rank_threshold=rank_threshold)
        U_sum_k = U[:, :sum_rel_rank]

        for i, _tv in enumerate(tvs_params):
            tv = _tv[key].data
            # construct the subspace shared by tau_t and tau_s
            U_tv, S_tv, V_tv = torch.linalg.svd(tv, full_matrices=False)
            sum_rel_rank_tv = compute_rank(S_tv, rank_threshold=rank_threshold)
            _U_share = torch.cat([U_sum_k, U_tv[:, :sum_rel_rank_tv]], dim=1)

            if optimize_shared_space:
                # optimize the shared space by cleanning the duplicates
                U_share, S_share, V_share = torch.linalg.svd(_U_share, full_matrices=False)
                sum_rel_rank_share = compute_rank(S_share, rank_threshold=rank_threshold)
                B_share = U_share[:, :sum_rel_rank_share]
            else:
                U_share, S_share, V_share = torch.linalg.svd(_U_share, full_matrices=False)
                B_share = U_share

            # project tau_s onto the shared subspace
            proj_tv_onto_share = torch.linalg.multi_dot((B_share, B_share.T, tv))

            # write it to the task vector
            get_attr(task_vectors[i], key.split(".")).data = proj_tv_onto_share

    return task_vectors
