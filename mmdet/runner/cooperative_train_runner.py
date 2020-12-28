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
import torch.nn.functional as F
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

    EPSILON = 1.0e-10

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

            outputs = [model.train_step(data_batch, optimizer, **kwargs) for model, optimizer in zip(self.models, self.optimizers)]

            if isinstance(self.opt_hook, CoteachingOptimizerHook):
                if self.opt_hook.coteaching_method == "naive" or self.opt_hook.coteaching_method == "per-object":
                    # naive Co-teaching [Han2018]
                    # per-object co-teaching [Chadwick2019]
                    zip_loss0 = zip(outputs[0]["losses"]["loss_cls"], outputs[0]["losses"]["loss_bbox"])
                    zip_loss1 = zip(outputs[1]["losses"]["loss_cls"], outputs[1]["losses"]["loss_bbox"])

                    loss0 = torch.cat([loss_cls + loss_bbox for loss_cls, loss_bbox in zip_loss0])
                    loss1 = torch.cat([loss_cls + loss_bbox for loss_cls, loss_bbox in zip_loss1])

                    if self.opt_hook.coteaching_method == "naive":
                        argsort_dim = -1
                    else:
                        argsort_dim = 1
                    ind0_sorted = torch.argsort(loss0, argsort_dim)
                    loss0_sorted = loss0.gather(argsort_dim, ind0_sorted)
                    ind1_sorted = torch.argsort(loss1, argsort_dim)
                    #loss1_sorted = loss1[ind1_sorted]

                    drop_rate = self.rate_schedule[self.epoch]
                    remember_rate = 1 - self.rate_schedule[self.epoch]

                    if self.opt_hook.coteaching_method == "naive":
                        num_remember = int(remember_rate * len(loss0_sorted))
                        ind0_update=ind0_sorted[:num_remember]
                        ind1_update=ind1_sorted[:num_remember]

                        # exchange data sample index
                        loss0_update = loss0[ind1_update]
                        loss1_update = loss1[ind0_update]

                        # pack
                        self.losses_update = [torch.sum(loss0_update)/num_remember, torch.sum(loss1_update)/num_remember]
                    else:
                        num_remember = int(remember_rate * len(loss0_sorted[argsort_dim]))
                        divisor = len(loss0_sorted[0])
                        ind0_update=ind0_sorted[:, :num_remember]
                        ind1_update=ind1_sorted[:, :num_remember]

                        # exchange data sample index
                        loss0_update = loss0[:, ind1_update]
                        loss1_update = loss1[:, ind0_update]                        

                        # pack
                        self.losses_update = [torch.sum(loss0_update)/divisor, torch.sum(loss1_update)/divisor]

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
                # logits value is required because log probability should be computed with a temperature parameter.
                logits = [model.forward_dummy(data_batch) for model, optimizer in zip(self.models, self.optimizers)]

                # distinguish student/teacher loss and optimizer
                student_optimizer = self.optimizers[0]
                (student_cls_logits, student_bbox_logits), (teacher_cls_logits, teacher_bbox_logits) = logits
                student_losses = outputs[0]
                student_cls_logits = torch.cat([logit.flatten() for logit in student_cls_logits])

                # Calculate distillation loss
                soft_log_probs = F.log_softmax(student_cls_logits.reshape(-1, self.opt_hook.num_classes) / self.opt_hook.temperature, dim=1)
                soft_targets = F.softmax(teacher_cls_logits.reshape(-1, self.opt_hook.num_classes) / self.opt_hook.temperature, dim=1)
                soft_kl_div = F.kl_div(soft_log_probs, soft_targets.detach(), reduction="none")

                if self.opt_hook.use_focal:
                    # compute focal term

                    # https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/focal.html#FocalLoss
                    if self.opt_hook.use_adaptive:
                        # Normal entropy is calculated by multiplying probability and log-probability
                        # https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510
                        if self.verbose > 0:
                            print("use automated adaptative distillation")
                        soft_log_targets = F.log_softmax(teacher_cls_logits.reshape(-1, self.opt_hook.num_classes) / self.opt_hook.temperature, dim=1)
                        entropy_target = torch.mean(- soft_targets * soft_log_targets, dim=1).unsqueeze(1).expand(-1, self.opt_hook.num_classes)
                        soft_distance = soft_kl_div + self.opt_hook.beta * entropy_target
                    else:
                        if abs(self.opt_hook.alpha) < self.EPSILON or abs(self.opt_hook.alpha - 1.0) < self.EPSILON:
                            if self.verbose > 0:
                                print("Alpha-balancing is disabled")
                            soft_distance = - soft_log_probs * soft_targets.detach()
                        else:
                            if self.verbose > 0:
                                print("Alpha-balancing is enabled")
                            soft_distance = - (np.log(self.opt_hook.alpha) + soft_log_probs) * (1 - self.opt_hook.alpha) * soft_targets.detach()
                    focal_term = torch.pow(1. - torch.exp( - soft_distance), self.opt_hook.gamma)

                    if self.opt_hook.normalized:
                        norm_factor = 1.0 / (focal_term.sum() + 1e-5)
                    else:
                        norm_factor = 1.0
                    if self.verbose > 0:
                        print("focal_term shape:{0} range[{1}, {2}]".format(focal_term.shape, torch.min(focal_term), torch.max(focal_term)))
                        print("norm_factor: {0}".format(norm_factor))
                    focal_distillation_loss = focal_term.reshape(-1, self.opt_hook.num_classes) * norm_factor * soft_kl_div

                    sum_focal_distillation_loss = focal_distillation_loss.sum() / num_pos
                    sum_classification_loss = classification_loss.sum()
                    sum_regression_loss = regression_loss.sum()
                    self.overall_loss = self.loss_wts.distill * sum_focal_distillation_loss + self.loss_wts.student * (sum_regression_loss + sum_classification_loss)
                else:
                    self.overall_loss = 0.3 * student_losses.sum() + 0.7 * soft_kl_div

            else:
                raise Exception("expected optimizer type is either CooperativeOptimizerHook or DistillationOptimizerHook. But got: {0}".format(type(self.opt_hook)))
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
