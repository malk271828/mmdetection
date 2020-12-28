import os.path as osp
import platform
import shutil
import time
import warnings
import numpy as np
from rich.progress import track
from rich import pretty, print
pretty.install()

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
            optimizer={ "opt"+str(i+1): optimizer for i, optimizer in enumerate(optimizers)},
            work_dir=work_dir,
            logger=logger,
            meta=meta)
        self.models = models
        self.optimizers = optimizers
        self.opt_hook = None

    def run_iter(self, data_batch, train_mode, **kwargs):
        """
        References
        ----------
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/models/detectors/base.py
        """
        verbose = 0
        if not self.opt_hook:
            # search the hook inherited from OptimizerHook
            for hook in self._hooks:
                if isinstance(hook, OptimizerHook):
                    self.opt_hook = hook
        if self.batch_processor is not None:
            outputs = self.batch_processor(
                self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            # construct rate_schedule
            if not hasattr(self, "rate_schedule"):
                dr_config = self.opt_hook.dr_config
                if dr_config:
                    # define drop rate schedule from config
                    self.rate_schedule = np.ones(self.max_epochs) * dr_config.max_drop_rate
                    self.rate_schedule[:dr_config.num_gradual] = np.linspace(0, dr_config.max_drop_rate, dr_config.num_gradual)
                else:
                    # define from default settings
                    self.rate_schedule = np.linspace(0, 0.8, self.max_epochs)

            if isinstance(self.opt_hook, CoteachingOptimizerHook):
                outputs = [model.train_step(data_batch, optimizer, **kwargs) for model, optimizer in zip(self.models, self.optimizers)]
                if self.opt_hook.coteaching_method == "naive" or self.opt_hook.coteaching_method == "per-object":
                    # [Han2018] and [Chadwick2019]
                    zip_loss0 = zip(outputs[0]["losses"]["loss_cls"], outputs[0]["losses"]["loss_bbox"])
                    zip_loss1 = zip(outputs[1]["losses"]["loss_cls"], outputs[1]["losses"]["loss_bbox"])

                    loss0 = torch.cat([loss_cls + loss_bbox for loss_cls, loss_bbox in zip_loss0])
                    loss1 = torch.cat([loss_cls + loss_bbox for loss_cls, loss_bbox in zip_loss1])

                    ind0_sorted = torch.argsort(loss0)
                    loss0_sorted = loss0[ind0_sorted]
                    ind1_sorted = torch.argsort(loss1)
                    #loss1_sorted = loss1[ind1_sorted]

                    drop_rate = self.rate_schedule[self.epoch]
                    remember_rate = 1 - self.rate_schedule[self.epoch]
                    num_remember = int(remember_rate * len(loss0_sorted))

                    # pure_ratio_1 = np.sum(noise_or_not[ind[ind0_sorted[:num_remember]]])/float(num_remember)
                    # pure_ratio_2 = np.sum(noise_or_not[ind[ind1_sorted[:num_remember]]])/float(num_remember)

                    ind0_update=ind0_sorted[:num_remember]
                    ind1_update=ind1_sorted[:num_remember]

                    # exchange data sample index
                    loss0_update = loss0[ind1_update]
                    loss1_update = loss1[ind0_update]

                    # pack
                    self.losses_update = [torch.sum(loss0_update)/num_remember, torch.sum(loss1_update)/num_remember]

                # model visualization
                if verbose > 1:
                    for i, (model, output) in enumerate(zip(self.models, outputs)):
                        file_path = "model_{0}.png".format(i)
                        if not osp.exists(file_path):
                            make_dot(output, params=dict(model.named_parameters())).render(file_path, format="png")
                        print("[cyan] output computational graph : {0} [/cyan]".format(file_path))

                # check output type
                if not isinstance(outputs[0], dict):
                    raise TypeError('"batch_processor()" or "model.train_step()"'
                                    'and "model.val_step()" must return a dict')
                
                # update log buffer for co-teaching
                self.log_buffer.update({"drop_rate": drop_rate})

            elif isinstance(self.opt_hook, DistillationOptimizerHook):
                outputs = [model.train_step(data_batch, optimizer, **kwargs) for model, optimizer in zip(self.models, self.optimizers)]
                self.logits = [model.forward_dummy(data_batch) for model, optimizer in zip(self.models, self.optimizers)]
            else:
                raise Exception("expected optimizer type is either CooperativeOptimizerHook or DistillationOptimizerHook. But got: {0}".format(type(opt_hook)))
        else:
            outputs = [model.val_step(data_batch, optimizer, **kwargs) for model, optimizer in zip(self.models, self.optimizers)]

        # register losses to log_buffer
        if isinstance(self.opt_hook, DistillationOptimizerHook):
            idx_str = ["student", "teacher"]
        else:
            idx_str = ["1", "2"]
        for i, output in enumerate(outputs):
            output["log_vars"]["loss_"+idx_str[i]] = output["log_vars"].pop("loss")
            output["log_vars"]["loss_cls_"+idx_str[i]] = output["log_vars"].pop("loss_cls")
            output["log_vars"]["loss_bbox_"+idx_str[i]] = output["log_vars"].pop("loss_bbox")
            if 'log_vars' in output:
                self.log_buffer.update(output['log_vars'], output['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        for model in self.models:
            model.train()
        super().train(data_loader, **kwargs)
