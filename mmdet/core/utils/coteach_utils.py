import warnings
import numpy as np
from rich.progress import track
from rich import pretty, print
pretty.install()

from mmdet.core import get_classes
from mmcv.runner import load_checkpoint
from mmcv.runner.hooks import HOOKS, OptimizerHook
from mmdet.utils import get_root_logger

# mmdet.models cannot be imported here
# set path and import build_detector directly as a workaround
# https://stackoverflow.com/questions/13763985/how-to-import-a-module-but-ignoring-the-packages-init-py
# import sys
# sys.path.append("mmdet/models")
# from builder import build_detector

@HOOKS.register_module()
class CoteachingOptimizerHook(OptimizerHook):
    """
    co-teaching logic

    naive co-teaching

    References
    ----------
    https://mmdetection.readthedocs.io/en/v2.5.0/tutorials/customize_runtime.html#customize-self-implemented-optimizer
    https://github.com/bhanML/Co-teaching/blob/master/loss.py

    [Chadwick2019]
    Chadwick, Simon, and Paul Newman. "Training object detectors with noisy data." 2019 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2019.
    https://arxiv.org/pdf/1905.07202.pdf
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
        if not (hasattr(runner, "models") and isinstance(runner.models, list)):
            runner.logger.warning("runner.models attribute must be list type in CoteachOptimizerHook. But got {0}".format(type(runner.models)))

        for optimizer, loss in zip(runner.optimizers, runner.losses_update):
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

        # extract distillation config
        self.alpha = distill_config.alpha
        self.beta = distill_config.beta
        self.gamma = distill_config.gamma
        self.loss_wts_hard = distill_config.loss_wts_hard
        self.loss_wts_soft = distill_config.loss_wts_soft
        self.temperature = distill_config.temperature
        self.use_focal = distill_config.use_focal
        self.use_adaptive = distill_config.use_adaptive

    def clip_grads(self, params):
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        if len(params) > 0:
            return clip_grad.clip_grad_norm_(params, **self.grad_clip)

    def after_train_iter(self, runner):
        if not (hasattr(runner, "models") and isinstance(runner.models, list)):
            runner.logger.warning("runner.models attribute must be list type in DistillationOptimizerHook. But got {0}".format(type(runner.models)))

        # It is assumed that opt1 is an optimizer for student model
        runner.optimizer["opt1"].zero_grad()
        runner.overall_loss.backward()
        runner.optimizer["opt1"].step()

        if self.grad_clip is not None:
            grad_norm = self.clip_grads(runner.model.parameters())
            if grad_norm is not None:
                # Add grad norm to the logger
                runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                         runner.outputs['num_samples'])

