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

from mmcv.parallel import MMDataParallel

# custom class
from mmcv.runner.hooks import OptimizerHook
from mmdet.core.utils.coteach_utils import CoteachingOptimizerHook, DistillationOptimizerHook

class CustomDataParallel(MMDataParallel):
    def forward_dummy(self, *inputs, **kwargs):
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        return self.module.forward_dummy(inputs[0][0]["img"])

@RUNNERS.register_module()
class CooperativeTrainRunner(EpochBasedRunner):
    """Cooperative Train Runner.
    This runner train models cooperatively.
    """
    def __init__(self,
                 models: list(),
                 batch_processor=None,
                 optimizers=None,
                 dr_config=None,
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

                # model visualization
                if verbose > 1:
                    for i, (model, output) in enumerate(zip(self.models, outputs)):
                        file_path = "model_{0}.png".format(i)
                        if not osp.exists(file_path):
                            make_dot(output, params=dict(model.named_parameters())).render(file_path, format="png")
                        print(Fore.CYAN + "output computational graph : {0}".format(file_path) + Style.RESET_ALL)

                # check output type
                if not isinstance(outputs[0], dict):
                    raise TypeError('"batch_processor()" or "model.train_step()"'
                                    'and "model.val_step()" must return a dict')

            elif isinstance(opt_hook, DistillationOptimizerHook):
                outputs = [model.train_step(data_batch, optimizer, **kwargs) for model, optimizer in zip(self.models, self.optimizers)]
                logits = [model.forward_dummy(data_batch) for model, optimizer in zip(self.models, self.optimizers)]
            else:
                raise Exception("expected optimizer type is either CooperativeOptimizerHook or DistillationOptimizerHook. But got: {0}".format(type(opt_hook)))
        else:
            outputs = [model.val_step(data_batch, optimizer, **kwargs) for model, optimizer in zip(self.models, self.optimizers)]

        # register losses to log_buffer
        if isinstance(opt_hook, DistillationOptimizerHook):
            idx_str = ["student", "teacher"]
        else:
            idx_str = ["1", "2"]
        for i, output in enumerate(outputs):
            output["log_vars"]["loss_cls_"+idx_str[i]] = output["log_vars"].pop("loss_cls")
            output["log_vars"]["loss_bbox_"+idx_str[i]] = output["log_vars"].pop("loss_bbox")
            if 'log_vars' in output:
                self.log_buffer.update(output['log_vars'], output['num_samples'])
        self.outputs = outputs
        self.logits = logits

    def train(self, data_loader, **kwargs):
        for model in self.models:
            model.train()
        super().train(data_loader, **kwargs)
