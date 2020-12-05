import warnings
import numpy as np
from colorama import *
init()

import torch

from mmdet.core import get_classes
from mmcv.runner import load_checkpoint
from mmcv.runner.hooks import HOOKS, OptimizerHook
from mmdet.utils import get_root_logger

# mmdet.models cannot be imported here
# set path and import build_detector directly as a workaround
# https://stackoverflow.com/questions/13763985/how-to-import-a-module-but-ignoring-the-packages-init-py
import sys
sys.path.append("mmdet/models")
from builder import build_detector

@HOOKS.register_module()
class CoteachingOptimizerHook(OptimizerHook):
    """
    co-teaching logic

    References
    ----------
    https://mmdetection.readthedocs.io/en/v2.5.0/tutorials/customize_runtime.html#customize-self-implemented-optimizer
    https://github.com/bhanML/Co-teaching/blob/master/loss.py
    """
    def __init__(self, grad_clip=None, coteaching_method=None, dr_config=None):
        self.grad_clip = grad_clip
        self.coteaching_method = coteaching_method
        self.dr_config = dr_config

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        if not hasattr(self, "rate_schedule"):
            if self.dr_config:
                # define drop rate schedule
                self.rate_schedule = np.ones(runner.max_epochs) * self.dr_config.max_drop_rate
                self.rate_schedule[:self.dr_config.num_gradual] = np.linspace(0, self.dr_config.max_drop_rate, self.dr_config.num_gradual)
            else:
                self.rate_schedule = np.linspace(0, 0.8, runner.max_epochs)

        if not (hasattr(runner, "models") and isinstance(runner.models, list)):
            runner.logger.warning("runner.models attribute must be list type in CoteachOptimizerHook. But got {0}".format(type(runner.models)))

        # co-teaching logic
        zip_loss0 = zip(runner.outputs[0]["losses"]["loss_cls"], runner.outputs[0]["losses"]["loss_bbox"])
        zip_loss1 = zip(runner.outputs[1]["losses"]["loss_cls"], runner.outputs[1]["losses"]["loss_bbox"])
        loss0 = torch.cat([loss_cls + loss_bbox for loss_cls, loss_bbox in zip_loss0])
        loss1 = torch.cat([loss_cls + loss_bbox for loss_cls, loss_bbox in zip_loss1])

        ind0_sorted = torch.argsort(loss0)
        loss0_sorted = loss0[ind0_sorted]
        ind1_sorted = torch.argsort(loss1)
        #loss1_sorted = loss1[ind1_sorted]

        drop_rate = self.rate_schedule[runner.epoch]
        remember_rate = 1 - self.rate_schedule[runner.epoch]
        num_remember = int(remember_rate * len(loss0_sorted))

        # pure_ratio_1 = np.sum(noise_or_not[ind[ind0_sorted[:num_remember]]])/float(num_remember)
        # pure_ratio_2 = np.sum(noise_or_not[ind[ind1_sorted[:num_remember]]])/float(num_remember)

        ind0_update=ind0_sorted[:num_remember]
        ind1_update=ind1_sorted[:num_remember]

        # exchange data sample index
        loss0_update = loss0[ind1_update]
        loss1_update = loss1[ind0_update]

        # pack
        loss_updates = [torch.sum(loss0_update/num_remember), torch.sum(loss1_update/num_remember)]

        for optimizer, loss in zip(runner.optimizers, loss_updates):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # update log buffer
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

        # update log buffer for co-teaching
        runner.log_buffer.update({"drop_rate": drop_rate})

@HOOKS.register_module()
class DistillationOptimizerHook(OptimizerHook):
    """
    References
    ----------
    https://mmdetection.readthedocs.io/en/v2.5.0/tutorials/customize_runtime.html#customize-self-implemented-optimizer
    """
    def __init__(self, grad_clip=None, distillation_method=None, distill_config=None):
        self.grad_clip = grad_clip
        self.distillation_method = distillation_method
        self.distill_config = distill_config
        self.teacher_model = None
        self.verbose = 0

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def before_train_iter(self, runner):
        if self.teacher_model is None:
            # init detector
            self.teacher_model = build_detector(self.distill_config, checkpoint=self.distill_config.checkpoint)

            # same logic to init_detector API
            if checkpoint is not None:
                    map_loc = "cpu"
                    checkpoint = load_checkpoint(self.teacher_model, checkpoint, map_location=map_loc)
                    if 'CLASSES' in checkpoint['meta']:
                        self.teacher_model.CLASSES = checkpoint['meta']['CLASSES']
                    else:
                        warnings.simplefilter('once')
                        warnings.warn('Class names are not saved in the checkpoint\'s '
                                    'meta data, use COCO classes by default.')
                        self.teacher_model.CLASSES = get_classes('coco')
            self.teacher_model.cfg = self.distill_config  # save the config in the model for convenience
            self.teacher_model.to(runner.device)
            self.teacher_model.eval()

            if self.verbose > 0:
                print(Fore.CYAN + "teacher model is loaded : {0}".format(checkpoint) + Style.RESET_ALL)

            # register models to runner
            runner.models.append(self.teacher_model)

    def after_train_iter(self, runner):
        if not (hasattr(runner, "models") and isinstance(runner.models, list)):
            runner.logger.warning("runner.models attribute must be list type in DistillationOptimizerHook. But got {0}".format(type(runner.models)))
        for optimizer, output in zip(runner.optimizers, runner.outputs):
            optimizer.zero_grad()
            output["loss"].backward()
            optimizer.step()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

