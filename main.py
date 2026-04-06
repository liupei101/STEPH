"""
This is the entry file to run all experiments
"""
import os.path as osp
import argparse
import time
import wandb

from runner import BaseHandler, SAHandler, SATaskAriHandler
from utils.io import load_config_from_yaml, print_config
from utils.func import args_grid, fetch_kws
from utils.func_config import is_valid_run_cfg
from utils.func_config import convert_to_abbr, ignore_it_in_save_path

def login_wandb(run_remote=False):
    if run_remote:
        # IP of your remote wandb server
        host = 'http://wandb-local:8080'
    else:
        # IP of your local wandb server (or a docker container)
        host = 'http://127.0.0.1:8080'
    login_code = wandb.login(
        key='key',
        relogin=True,
        host=host,
        force=True,
        verify=True,
    )
    return login_code

def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-f', required=True, type=str, help='Path to the config file.')
    parser.add_argument('--handler', '-d', type=str, choices=['CLF', 'SA', 'SATA'], default='SA', help='Handler for running experiments.')
    parser.add_argument('--multi_run', action='store_true', help='If have multiple runs.')
    parser.add_argument('--run_remote', action='store_true', help='If run experiments in a remote server.')
    parser.add_argument('--sleep', type=int, default=0, help='If sleep how many seconds between two runs, only valid in a multi_run mode.')
    parser.add_argument('--cfg_dataset_name', type=str, default=None, help='This field overrides the config and forces to run a specific dataset.')
    parser.add_argument('--cfg_loss_merge_weight', type=float, default=None, help='This field overrides the config and forces to set loss_merge_weight.')
    args = vars(parser.parse_args())
    return args

def main(handler, config):
    if not is_valid_run_cfg(config):
        print("[Warning] skipped this run with config:", config)
        return

    login_wandb()

    model = handler(config)
    if config['test']:
        metrics = model.exec_test()
    else:
        metrics = model.exec()
    print('[INFO] Metrics:', metrics)

def is_duplicated_run(config):
    if config['test'] is True:
        finished_file_test = osp.join(config['test_save_path'], 'test_mode_metrics-last.txt')
        return osp.exists(finished_file_test)
    else:
        finished_file = osp.join(config['save_path'], 'train_metrics-last.txt')
        return osp.exists(finished_file)

def multi_run_main(handler, config, sleep=0, run_remote=False):

    hyperparams = []
    for k, v in config.items():
        if isinstance(v, list):
            hyperparams.append(k)

    if config['data_split_fold'] is None:
        configs = args_grid(config, loop_preference=['dataset_name'])
    else:
        configs = args_grid(config, loop_preference=['data_split_fold', 'dataset_name'])

    login_wandb(run_remote)

    for cur_cfg in configs:
        print('\n')
        for k in hyperparams:
            abbr_key, abbr_value = convert_to_abbr(k), convert_to_abbr(cur_cfg[k])
            
            if ignore_it_in_save_path(k, cur_cfg[k]):
                print(f"[INFO] `{k}` is ignored and will not be added to `save_path`.")
                continue

            cur_cfg['save_path'] += '-{}_{}'.format(abbr_key, abbr_value)
            if cur_cfg['test']:
                cur_cfg['test_save_path'] += '-{}_{}'.format(abbr_key, abbr_value)

        if not is_valid_run_cfg(cur_cfg):
            print("[Warning] skipped this run with config:", cur_cfg)
            continue

        if is_duplicated_run(cur_cfg):
            print("[Warning] skipped this duplicated run with config:", cur_cfg)
            continue

        model = handler(cur_cfg)
        if cur_cfg['test']:
            print(cur_cfg['test_save_path'])
            metrics = model.exec_test()
        else:
            print(cur_cfg['save_path'])
            metrics = model.exec()

        print('[INFO] Metrics:', metrics)

        time.sleep(sleep)

if __name__ == '__main__':
    cfg = get_cmd_args()
    cfg_to_override = fetch_kws(cfg, prefix='cfg')
    print("CFG to override:", cfg_to_override)
    config = load_config_from_yaml(cfg['config'], **cfg_to_override)
    print_config(config)
    
    if cfg['handler'] == 'CLF':
        # for classification models
        handler = BaseHandler
    elif cfg['handler'] == 'SA':
        # for survival analysis (SA) models
        handler = SAHandler
    elif cfg['handler'] == 'SATA':
        # for model merging-based SA models
        handler = SATaskAriHandler
    else:
        raise RuntimeError(f"Expected `CLF`, `SA`, or `SATA`, but got {cfg['handler']}")

    if cfg['multi_run']:
        multi_run_main(handler, config, sleep=cfg['sleep'], run_remote=cfg['run_remote'])
    else:
        main(handler, config)
