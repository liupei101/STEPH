from abc import ABC
from copy import deepcopy
from typing import Dict, List, Optional, Union
from collections import OrderedDict

import torch
from torch import nn

__all__ = ["ModelPool"]


class ModelPool(ABC):

    model_names = None
    null_model_name = None
    target_model_name = None

    def __init__(
        self,
        base_model: nn.Module,
        cfg_null_model: Dict,
        cfg_target_model: Dict,
        cfg_model_pool: Dict[str, Dict]
    ):
        """
        Initialize the ModelPool with the given configuration.

        Args:
            base_model (nn.Module): the network architecture of all fitted models.
            cfg_null_model (Dict): the configuration of the null model (unfitted).
            cfg_target_model (Dict): the configuration of the target model.
            cfg_model_pool (Dict): the configuration of all fitted models.
        """
        super().__init__()
        self.base_model = base_model
        self.cfg_model_pool = cfg_model_pool
        self.model_names = [_ for _ in self.cfg_model_pool.keys()]
        
        self.cfg_null_model = cfg_null_model
        self.null_model_name = cfg_null_model['name']
        print(f"[ModelPool] A null model ({self.null_model_name}) is specified.")

        self.cfg_target_model = cfg_target_model
        self.target_model_name = cfg_target_model['name']
        print(f"[ModelPool] A target model ({self.target_model_name}) is specified.")

    def __len__(self) -> int:
        """
        Return the number of source models in the model pool.

        Returns:
            int: The number of source models in the model pool.
        """
        return len(self.model_names)

    def get_model_config(self, model_name: str) -> Dict:
        """
        Retrieves the configuration for a specific model from the model pool.

        Args:
            model_name (str): The name of the model for which to retrieve the configuration.

        Returns:
            dict: The configuration dictionary for the specified model.
        """
        return self.cfg_model_pool[model_name]

    def get_null_model_config(self) -> Dict:
        return self.cfg_null_model

    def get_target_model_config(self) -> Dict:
        return self.cfg_target_model

    def load_model_from_path(
        self,
        path_ckpt: str,
        ckpt_key: Optional[str] = None,
        copy:bool = True
    ) -> nn.Module:
        """
        Load the model's state dict to the base model according to given path
        """
        model_data = torch.load(path_ckpt)
        if ckpt_key is not None:
            model_state_dict = model_data[ckpt_key]
        else:
            model_state_dict = model_data
        assert isinstance(model_state_dict, OrderedDict), f"Expected an OrderedDict loaded from {path_ckpt}."

        # load state_dict to the base model
        load_ret = self.base_model.load_state_dict(model_state_dict, strict=False)
        print("[ModelPool] loaded state dict:", load_ret)
        if copy:
            model = deepcopy(self.base_model)

        return model

    def load_model(self, model_name: str, copy=True) -> nn.Module:
        """
        Load the model from the model pool.

        Args:
            model_config (str | DictConfig): The configuration dictionary for the model to load.
            copy (bool): Whether to return a copy of the model.

        Returns:
            nn.Module: The loaded model.
        """
        model_cfg = self.get_model_config(model_name)
        model = self.load_model_from_path(
            model_cfg['path_ckpt'],
            ckpt_key='model',
            copy=copy
        )
        return model
        
    def load_target_model(self, copy=True):
        model = self.load_model_from_path(
            self.cfg_target_model['path_ckpt'],
            ckpt_key='model',
            copy=copy
        )
        return model

    def load_null_model(self, copy=True):
        model = self.load_model_from_path(
            self.cfg_null_model['path_ckpt'],
            ckpt_key='model',
            copy=copy
        )
        return model

    def models(self):
        """
        Generator that yields source models from the model pool.

        Yields:
            nn.Module: The next source model in the model pool.
        """
        for model_name in self.model_names:
            yield self.load_model(model_name)

    def named_models(self):
        """
        Generator that yields source model names and source models from the model pool.

        Yields:
            tuple: A tuple containing the source model name and the source model.
        """
        for model_name in self.model_names:
            yield model_name, self.load_model(model_name)

    def get_train_dataset(self, model_name: str):
        """
        Get the training dataset for the model.

        Args:
            model_name (str): The name of the model for which to get the training dataset.

        Returns:
            Any: The training dataset for the model.
        """
        raise NotImplementedError

    def get_test_dataset(self, model_name: str):
        """
        Get the testing dataset for the model.

        Args:
            model_name (str): The name of the model for which to get the testing dataset.

        Returns:
            Any: The testing dataset for the model.
        """
        raise NotImplementedError

    def setup_taskpool(self, taskpool):
        """
        Setup the taskpool before evaluation.
        Such as setting the fabric, processor, tokenizer, etc.

        Args:
            taskpool (Any): The taskpool to setup.
        """
        pass

    def to_model_list(self) -> List[nn.Module]:
        """
        Convert the model pool to a list of models.

        Returns:
            list: A list of models.
        """
        return [self.load_model(m) for m in self.model_names]

    def to_model_dict(self) -> Dict[str, nn.Module]:
        """
        Convert the model pool to a dictionary of models.

        Returns:
            dict: A dictionary of models.
        """
        return {m: self.load_model(m) for m in self.model_names}

    def get_init_coef_list(self) -> List[Union[int, float]]:
        """
        Return a list of models' initial coefficients.

        Returns:
            list: A list of initial coefficients.
        """
        return [self.cfg_model_pool[m]['init_coef'] for m in self.model_names]

    def get_init_coef_dict(self) -> Dict[str, Union[int, float]]:
        """
        Return a dict of models' initial coefficients.

        Returns:
            dict: A dictionary of initial coefficients.
        """
        return {m: self.cfg_model_pool[m]['init_coef'] for m in self.model_names}
