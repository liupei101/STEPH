from .func import parse_str_dims, fill_placeholder

# strictly ordered, DO NOT CHANGE THIS
DATASET_LIST = [
    'tcga_blca', 'tcga_brca', 'tcga_cesc', 'tcga_coadread', 'tcga_gbmlgg', 
    'tcga_hnsc', 'tcga_kipan', 'tcga_lihc', 'tcga_lung', 'tcga_sarc', 
    'tcga_skcm', 'tcga_stes', 'tcga_ucec'
]
MAP_TO_NEAREST_TRANSFER = {
    'tcga_blca': 'tcga_hnsc',
    'tcga_brca': 'tcga_gbmlgg',
    'tcga_cesc': 'tcga_stes',
    'tcga_coadread': 'tcga_stes',
    'tcga_gbmlgg': 'tcga_kipan',
    'tcga_hnsc': 'tcga_blca',
    'tcga_kipan': 'tcga_gbmlgg',
    'tcga_lihc': 'tcga_sarc',
    'tcga_lung': 'tcga_kipan',
    'tcga_sarc': 'tcga_lihc',
    'tcga_skcm': 'tcga_stes',
    'tcga_stes': 'tcga_cesc',
    'tcga_ucec': 'tcga_sarc',
}

MAP_TO_MIDDLE_TRANSFER = {
    'tcga_blca': 'tcga_gbmlgg',
    'tcga_brca': 'tcga_coadread',
    'tcga_cesc': 'tcga_coadread',
    'tcga_coadread': 'tcga_brca',
    'tcga_gbmlgg': 'tcga_lihc',
    'tcga_hnsc': 'tcga_gbmlgg',
    'tcga_kipan': 'tcga_blca',
    'tcga_lihc': 'tcga_skcm',
    'tcga_lung': 'tcga_stes',
    'tcga_sarc': 'tcga_kipan',
    'tcga_skcm': 'tcga_lihc',
    'tcga_stes': 'tcga_brca',
    'tcga_ucec': 'tcga_coadread',
}

MAP_TO_FAREST_TRANSFER = {
    'tcga_blca': 'tcga_brca',
    'tcga_brca': 'tcga_skcm',
    'tcga_cesc': 'tcga_kipan',
    'tcga_coadread': 'tcga_gbmlgg',
    'tcga_gbmlgg': 'tcga_stes',
    'tcga_hnsc': 'tcga_sarc',
    'tcga_kipan': 'tcga_skcm',
    'tcga_lihc': 'tcga_stes',
    'tcga_lung': 'tcga_skcm',
    'tcga_sarc': 'tcga_skcm',
    'tcga_skcm': 'tcga_gbmlgg',
    'tcga_stes': 'tcga_sarc',
    'tcga_ucec': 'tcga_blca',
}

def convert_to_abbr(key):
    ABBR_MAPS = {
        'data_split_fold': 'fold',
        'dataset_name': 'data',
        'decoder_num_feat_proj_layers': 'num_prj',
        'deepmil_post_mil_layer': 'post',
        'init_ckpt_src_dataset': 'wsrc',
        'init_ckpt_src_dataset_fold': 'wsrc_fold',
        'task_ari_src_dataset_name': 'tasrc',
        'task_ari_src_data_split_fold': 'tasrc_fold',
        'task_ari_src_init_coef': 'tasrc_init_w',
        'task_ari_src_exlude_target': 'exd_tar',
        'task_ari_compute_weights': 'merge_w',
        'task_ari_model_covar': 'covar',
        'task_ari_adaptive_mix': 'adamix',
        'task_ari_topk_task_vectors': 'topk',
        'loss_merge_weight': 'l_merw',
        'loss_mix_weight': 'l_mixw',
        'loss_kl_div': 'kl',
        'task_ari_param_to_exclude_norm': 'train_norm'
    }

    if key in ABBR_MAPS.keys():
        print(f"[info] abbreviate {key} as {ABBR_MAPS[key]}.")
        return ABBR_MAPS[key]
    else:
        return key

def ignore_it_in_save_path(key, value):
    IGNORE_LIST = dict()

    if key in IGNORE_LIST.keys():
        judge_func = IGNORE_LIST[key]
        return judge_func(value)

    return False

def infer_parameter_names(short_name):
    if short_name == 'mil_encoder':
        return [
            'feat_proj.0.weight',
            'feat_proj.0.bias',
            'attention_net.attention_a.0.weight',
            'attention_net.attention_a.0.bias',
            'attention_net.attention_b.0.weight',
            'attention_net.attention_b.0.bias',
            'attention_net.attention_c.weight',
            'attention_net.attention_c.bias'
        ]
    elif short_name == 'pred_head':
        return [
            'pred_head.weight',
            'pred_head.bias'
        ]
    elif short_name in ['all', 'mil_encoder+pred_head']:
        return [
            'feat_proj.0.weight',
            'feat_proj.0.bias',
            'attention_net.attention_a.0.weight',
            'attention_net.attention_a.0.bias',
            'attention_net.attention_b.0.weight',
            'attention_net.attention_b.0.bias',
            'attention_net.attention_c.weight',
            'attention_net.attention_c.bias',
            'pred_head.weight',
            'pred_head.bias'
        ]
    else:
        raise ValueError(f"{short_name} is not in infer lists")

def read_initial_lambda_t(cfg):
    if 'task_ari_initial_lambda_t' in cfg and cfg['task_ari_initial_lambda_t'] is not None:
        if 'task_ari_src_exlude_target' in cfg and cfg['task_ari_src_exlude_target'] is True:
            initial_lambda_t = []
            for _dataset, _lam in zip(cfg['task_ari_src_dataset_name'].split('-'), cfg['task_ari_initial_lambda_t'].split('-')):
                if _dataset == cfg['dataset_name']:
                    continue
                initial_lambda_t.append(float(_lam))
        else:
            initial_lambda_t = [float(_lam) for _lam in cfg['task_ari_initial_lambda_t'].split('-')]
    else:
        initial_lambda_t = None
    return initial_lambda_t

def get_exp_datasets(cfg):
    """
    Return a list of datasets that will be used in this experiment.
    """
    if 'dataset_name' not in cfg or cfg['dataset_name'] is None:
        # Multiple datasets for this experiment
        assert 'dataset_list' in cfg and 'dataset_chosen_index' in cfg, "Failed to find datasets for experiments."
        dataset_list = parse_str_dims(cfg['dataset_list'], dtype=str)
        dataset_chosen_index = parse_str_dims(cfg['dataset_chosen_index'], dtype=int)
        dataset_chosen_index = sorted(dataset_chosen_index)
        return [dataset_list[i] for i in dataset_chosen_index]
    else:
        # One dataset for this experiment
        assert isinstance(cfg['dataset_name'], str)
        return [cfg['dataset_name']]

def fill_placeholder_in_cfg(cfg):
    # for placeholder = {dataset}
    if 'dataset_name' in cfg:
        dataset_name = cfg['dataset_name']
        temp_keys = [
            'save_path', 'path_patch', 'path_coord', 'path_cluster', 'path_graph', 
            'path_table', 'data_split_path', 
            'task_ari_trg_ckpt_path', 'task_ari_task_inputs_path',
            'task_ari_src_dataset_name', 'path_wsi_rep'
        ]
        for temp_key in temp_keys:
            if temp_key in cfg:
                cfg[temp_key] = fill_placeholder(cfg[temp_key], dataset_name, ind="{dataset}")

    # for placeholder = {fold}
    if 'data_split_fold' in cfg and cfg['data_split_fold'] is not None:
        data_split_fold = cfg['data_split_fold']
        temp_keys = [
            'data_split_path', 'path_table_data_split', 'transfer_path_feat', 'task_ari_task_inputs_path',
            'transfer_load_ckpt_path', 'transfer_path_self_feat', 'task_ari_trg_ckpt_path', 'path_wsi_rep'
        ]
        for temp_key in temp_keys:
            if temp_key in cfg:
                cfg[temp_key] = fill_placeholder(cfg[temp_key], data_split_fold, ind="{fold}")

    # for the key: `init_ckpt_path`
    if 'init_ckpt' in cfg and cfg['init_ckpt']:
        cfg['init_ckpt_path'] = fill_placeholder(cfg['init_ckpt_path'], cfg['init_ckpt_src_dataset'], ind="{source_dataset}")
        cfg['init_ckpt_path'] = fill_placeholder(cfg['init_ckpt_path'], cfg['init_ckpt_src_dataset_fold'], ind="{source_fold}")
    
    return cfg

def check_necessary_columns_in_label_dataframe(columns):
    for col in ['patient_id', 'pathology_id', 'project', 'dataset', 'dataset_id', 't', 'e']:
        assert col in columns, f"Column named {col} is not found in label dataframe."

def is_valid_run_cfg(cfg):
    if 'test' in cfg and cfg['test']:
        if 'task_ari' in cfg and cfg['task_ari']:
            if 'task_ari_src_dataset_name' in cfg and cfg['task_ari_src_dataset_name'] == cfg['dataset_name']:
                print("[INFO] found the same dataset name for the source and the target task (at task_ari mode).")
                return False

    if 'init_ckpt' in cfg and cfg['init_ckpt']:
        if cfg['init_ckpt_src_dataset'] == cfg['dataset_name']:
            print("[INFO] found the same dataset name in fine-tuning from transferred ckpt.")
            return False

    return True