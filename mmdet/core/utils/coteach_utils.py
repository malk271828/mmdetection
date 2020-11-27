import numpy as np

import torch

from mmcv.runner.hooks import HOOKS, OptimizerHook
from mmdet.utils import get_root_logger

@HOOKS.register_module()
class CoteachingOptimizerHook(OptimizerHook):
    """
    References
    ----------
    https://mmdetection.readthedocs.io/en/v2.5.0/tutorials/customize_runtime.html#customize-self-implemented-optimizer

    https://github.com/bhanML/Co-teaching/blob/master/loss.py
    """
    def __init__(self, grad_clip=None, cooperative_method=None, dr_config=None):
        self.grad_clip = grad_clip
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

        remember_rate = 1 - self.rate_schedule[self.epoch]
        num_remember = int(remember_rate * len(loss0_sorted))

        # pure_ratio_1 = np.sum(noise_or_not[ind[ind0_sorted[:num_remember]]])/float(num_remember)
        # pure_ratio_2 = np.sum(noise_or_not[ind[ind1_sorted[:num_remember]]])/float(num_remember)

        ind0_update=ind0_sorted[:num_remember]
        ind1_update=ind1_sorted[:num_remember]

        # exchange
        loss0_update = loss0[ind1_update]
        loss1_update = loss1[ind0_update]

        # pack
        loss_updates = [torch.sum(loss0_update/num_remember), torch.sum(loss1_update/num_remember)]

        for optimizer, loss in zip(runner.optimizers, loss_updates):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])


@HOOKS.register_module()
class DistillationOptimizerHook(OptimizerHook):
    """
    References
    ----------
    https://mmdetection.readthedocs.io/en/v2.5.0/tutorials/customize_runtime.html#customize-self-implemented-optimizer
    """
    def __init__(self, grad_clip=None, cooperative_method=None, dr_config=None):
        self.grad_clip = grad_clip

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        if not (hasattr(runner, "models") and isinstance(runner.models, list)):
            runner.logger.warning("runner.models attribute must be list type in CoteachOptimizerHook. But got {0}".format(type(runner.models)))
        for optimizer, output in zip(runner.optimizers, runner.outputs):
            optimizer.zero_grad()
            output["loss"].backward()
        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])
        for optimizer in runner.optimizers:
            optimizer.step()

