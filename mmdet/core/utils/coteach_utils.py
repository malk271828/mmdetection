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
    def __init__(self, grad_clip=None, cooperative_method=None):
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


@HOOKS.register_module()
class DistillationOptimizerHook(OptimizerHook):
    """
    References
    ----------
    https://mmdetection.readthedocs.io/en/v2.5.0/tutorials/customize_runtime.html#customize-self-implemented-optimizer
    """
    def __init__(self, grad_clip=None, cooperative_method=None):
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

