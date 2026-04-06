"""
Class for bag-style dataloader
"""
from typing import Union, Optional
import os.path as osp
import lmdb
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset

from utils.io import retrieve_from_table_clf
from utils.io import read_patch_data, read_patch_coord
from utils.func import sampling_data, random_mask_instance
from utils.func import agg_dict, fill_placeholder
from .label_converter import MetaSurvData


class WSIPatchSurv(Dataset):
    r"""A WSI dataset class for survival prediction tasks (patient-level generally).

    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        mode (string): 'patch', or 'cluster'.
        meta_data: label information of all samples in the dataset.
        read_format (string): The suffix name or format of the file storing patch feature.
    Return:
        index: The index of current item in the whole dataset, used to retrieval patient ID.
        (feats, extra_data): Patch features and extra data.
        label: It contains typical survival labels, 'last follow-up time' and 'event status';
            event = 1 -> w/ event, called uncensored one; event = 0 -> w/o event, called censored one.
    """
    def __init__(self, patient_ids: list, patch_path: str, mode:str, meta_data:Union[list,MetaSurvData],
        read_format:str='pt', sampling_ratio:Union[None,float,int]=None, sampling_seed=42, **kws):
        super().__init__()
        if sampling_ratio is not None:
            print("[dataset] Patient-level sampling with ratio ({}) and seed ({})".format(sampling_ratio, sampling_seed))
            patient_ids, pid_left = sampling_data(patient_ids, sampling_ratio, seed=sampling_seed)
            print("[dataset] Sampled {} patients, left {} patients".format(len(patient_ids), len(pid_left)))

        assert mode in ['patch', 'cluster']
        self.mode = mode
        if self.mode == 'cluster':
            assert 'cluster_path' in kws
        if self.mode == 'patch':
            assert 'coord_path' in kws
        self.kws = kws
        
        self.pids, self.pid2info = meta_data.collect_info_by_pids(
            patient_ids, target_columns=['pathology_id', 'y_t', 'y_e', 'project', 'dataset', 'dataset_id']
        )

        self.meta_data = meta_data
        self.uid = self.pids

        self.read_path = patch_path
        self.read_format = read_format
        if self.read_format == 'lmdb':
            self.lmdb_env = None
            with lmdb.open(self.read_path, readonly=True, lock=False) as env:
                with env.begin(write=False) as txn:
                    keys = pickle.loads(txn.get(b'__keys__'))
                    # lmdb_keys = slide_id
                    self.lmdb_keys = [osp.splitext(k)[0] for k in keys]
            print(f"[WSIPatchSurv] use LMDB dataset: loaded {len(self.lmdb_keys)} pt files from {self.read_path}.")

        self.summary()

    def summary(self):
        print(f"[Dataset] WSIPatchSurv: in {self.mode} mode, avaiable patients count {self.__len__()}.")

    def _init_lmdb_env(self):
        """Lazy loading LMDB in worker"""
        assert self.read_format == 'lmdb'
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                self.read_path, readonly=True, lock=False, 
                readahead=True, meminit=False
            )

    def get_meta_data(self):
        return self.meta_data

    def get_patient_info(self):
        return self.pids, self.pid2info

    def get_feat_read_path(self, pid, sid):
        if "{project}" in self.read_path:
            cur_read_path = self.read_path.replace("{project}", self.pid2info[pid]['project'])
        return osp.join(cur_read_path, sid + '.' + self.read_format)

    def read_patch_data_from_patient(self, pid):
        feats = []
        sids = self.pid2info[pid]['pathology_id']
        for sid in sids:
            if self.read_format == 'lmdb':
                self._init_lmdb_env()
                assert sid in self.lmdb_keys
                key_sid = sid.encode('utf-8')
                with self.lmdb_env.begin(write=False) as txn:
                    buf = txn.get(key_sid)
                    if buf is None:
                        raise KeyError(f"LMDBDataset: missing key {key}")
                    tensor = pickle.loads(buf)
                feats.append(tensor)
            else:
                path = self.get_feat_read_path(pid, sid)
                # expected patch_data: [1, K, d]
                feats.append(read_patch_data(path, dtype='torch'))
        # [1, sum_K, d] -> [sum_K, d]
        all_feats = torch.cat(feats, dim=1).squeeze(0).float()
        return all_feats

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, index):
        pid   = self.pids[index]
        info  = self.pid2info[pid]
        sids  = info['pathology_id']
        label = [info['y_t'], info['y_e']]
        # get all data from one patient
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor(label).to(torch.float)

        if self.mode == 'patch':
            feats = self.read_patch_data_from_patient(pid)
            extra_data = torch.Tensor([info['dataset_id']]).int() # 0 if there is only one dataset
            return index, (feats, extra_data), label

        elif self.mode == 'cluster':
            cids = np.load(osp.join(self.kws['cluster_path'], '{}.npy'.format(pid)))
            feats = []
            for sid in sids:
                full_path = self.get_feat_read_path(pid, sid)
                if not osp.exists(full_path):
                    raise ValueError(f"[WSIPatchSurv] not found slide {sid} in {full_path}.")
                feats.append(read_patch_data(full_path, dtype='torch'))
            feats = torch.cat(feats, dim=0).to(torch.float)
            cids = torch.Tensor(cids)
            assert cids.shape[0] == feats.shape[0]
            return index, (feats, cids), label

        else:
            pass
            return None


class WSIPatchClf(Dataset):
    r"""A WSI dataset class for classification tasks (slide-level in general).
    
    Args:
        patient_ids (list): A list of patients (string) to be included in dataset.
        patch_path (string): The root path of WSI patch features. 
        table_path (string): The path of table with dataset labels, which has to be included. 
        mode (string): 'patch', 'cluster', or 'graph'.
        read_format (string): The suffix name or format of the file storing patch feature.
    """
    def __init__(self, patient_ids: list, patch_path: str, table_path: str, label_path:Union[None,str]=None,
        read_format:str='pt', ratio_sampling:Union[None,float,int]=None, ratio_mask=None, mode='patch', **kws):
        super(WSIPatchClf, self).__init__()
        if ratio_sampling is not None:
            assert ratio_sampling > 0 and ratio_sampling < 1.0
            print("[dataset] patient-level sampling with ratio_sampling = {}".format(ratio_sampling))
            patient_ids, pid_left = sampling_data(patient_ids, ratio_sampling)
            print("[dataset] sampled {} patients, left {} patients".format(len(patient_ids), len(pid_left)))
        if ratio_mask is not None and ratio_mask > 1e-5:
            assert ratio_mask <= 1, 'The argument ratio_mask must be not greater than 1.'
            assert mode == 'patch', 'Only support a patch mode for instance masking.'
            self.ratio_mask = ratio_mask
            print("[dataset] masking instances with ratio_mask = {}".format(ratio_mask))
        else:
            self.ratio_mask = None

        self.read_path = patch_path
        self.label_path = label_path
        self.has_patch_label = (label_path is not None) and len(label_path) > 0
        
        info = ['sid', 'sid2pid', 'sid2label']
        self.sids, self.sid2pid, self.sid2label = retrieve_from_table_clf(
            patient_ids, table_path, ret=info, level='slide')
        self.uid = self.sids
        
        assert mode in ['patch', 'cluster']
        self.mode = mode
        self.read_format = read_format
        self.kws = kws
        if self.mode == 'cluster':
            assert 'cluster_path' in kws
        if self.mode == 'patch':
            assert 'coord_path' in kws
        self.summary()

    def summary(self):
        print(f"[dataset] in {self.mode} mode, avaiable WSIs count {self.__len__()}")
        if not self.has_patch_label:
            print("[dataset] the patch-level label is not avaiable, derived by slide label.")

    def __len__(self):
        return len(self.sids)

    def __getitem__(self, index):
        sid   = self.sids[index]
        pid   = self.sid2pid[sid]
        label = self.sid2label[sid]
        # get patches from one slide
        index = torch.Tensor([index]).to(torch.int)
        label = torch.Tensor([label]).to(torch.long)

        if self.mode == 'patch':
            full_path = osp.join(self.read_path, sid + '.' + self.read_format)
            feats = read_patch_data(full_path, dtype='torch').to(torch.float)
            # if masking patches
            if self.ratio_mask:
                feats = random_mask_instance(feats, self.ratio_mask, scale=1, mask_way='mask_zero')
            full_coord = osp.join(self.kws['coord_path'],  sid + '.h5')
            coors = read_patch_coord(full_coord, dtype='torch')
            if self.has_patch_label:
                path = osp.join(self.label_path, sid + '.npy')
                patch_label = read_patch_data(path, dtype='torch', key='label').to(torch.long)
            else:
                patch_label = label * torch.ones(feats.shape[0]).to(torch.long)
            assert patch_label.shape[0] == feats.shape[0]
            assert coors.shape[0] == feats.shape[0]
            return index, (feats, coors), (label, patch_label)
        else:
            pass
            return None