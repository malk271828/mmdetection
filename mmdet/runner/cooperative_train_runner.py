import os.path as osp
import platform
import shutil
import time
import warnings
from pprint import pprint
from colorama import *
init()

import torch
from torchviz import make_dot

import mmcv
from mmcv.runner import EpochBasedRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info

# custom class
from mmcv.runner.hooks import OptimizerHook
from mmdet.core.utils.coteach_utils import CoteachingOptimizerHook

@RUNNERS.register_module()
class CooperativeTrainRunner(EpochBasedRunner):
    """Cooperative Train Runner.
    This runner train models cooperatively.
    """
    def __init__(self,
                 models: list(),
                 batch_processor=None,
                 optimizers=None,
                 work_dir=None,
                 logger=None,
                 meta=None,
                 max_iters=None,
                 max_epochs=None):
        if len(models)!=len(optimizers):
            raise Exception("number of models and optimizers must be equal {0}!={1}".format(len(models), len(optimizers)))
        if len(models) != len(set(models)) or len(optimizers) != len(set(optimizers)):
            raise Exception("models or optimizers include same instance(s)")

        self.batch_processor = batch_processor
        # init with the first model and optimizer
        super().__init__(models[0],
            optimizer=optimizers[0],
            work_dir=work_dir,
            logger=logger,
            meta=meta)
        self.models = models
        self.optimizers = optimizers

    def run_iter(self, data_batch, train_mode, **kwargs):
        """
        References
        ----------
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/detectors/base.py
        """
        verbose = 0
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            opt_hook = None
            for hook in self._hooks:
                if isinstance(hook, OptimizerHook):
                    opt_hook = hook
            if isinstance(opt_hook, CoteachingOptimizerHook):
                outputs = [model.train_step(data_batch, optimizer, **kwargs) for model, optimizer in zip(self.models, self.optimizers)]
                if verbose > 1:
                    for i, (model, output) in enumerate(zip(self.models, outputs)):
                        file_path = "model_{0}.png".format(i)
                        if not osp.exists(file_path):
                            make_dot(output, params=dict(model.named_parameters())).render(file_path, format="png")
                        print(Fore.CYAN + "output computational graph : {0}".format(file_path) + Style.RESET_ALL)
            else:
                raise Exception("expected optimizer type is CooperativeOptimizerHook. But got: {0}".format(type(opt_hook)))
        else:
            outputs = [model.val_step(data_batch, optimizer, **kwargs) for model, optimizer in zip(self.models, self.optimizers)]
        if not isinstance(outputs[0], dict):
            raise TypeError('"batch_processor()" or "model.train_step()"'
                            'and "model.val_step()" must return a dict')
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs
    
    def train(self, data_loader, **kwargs):
        for model in self.models:
            model.train()
        super().train(data_loader, **kwargs)
