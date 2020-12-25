import random
from copy import deepcopy
from colorama import *
init()

import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer, load_checkpoint)
from mmcv.utils import build_from_cfg

from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import get_root_logger
from mmdet.runner import CooperativeTrainRunner, CustomDataParallel

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(model,
                   dataset,
                   cfg,
                   distributed=False,
                   validate=False,
                   timestamp=None,
                   meta=None,
                   verbose=0):
    logger = get_root_logger(cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if 'imgs_per_gpu' in cfg.data:
        logger.warning('"imgs_per_gpu" is deprecated in MMDet V2.0. '
                       'Please use "samples_per_gpu" instead')
        if 'samples_per_gpu' in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and '
                f'"samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"'
                f'={cfg.data.imgs_per_gpu} is used in this experiments')
        else:
            logger.warning(
                'Automatically set "samples_per_gpu"="imgs_per_gpu"='
                f'{cfg.data.imgs_per_gpu} in this experiments')
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # cfg.gpus will be ignored if distributed
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed) for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        # Sets the `find_unused_parameters` parameter in
        # torch.nn.parallel.DistributedDataParallel
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
    else:
        models = model if isinstance(model, (list, tuple)) else [model]

        # load teacher model
        if hasattr(cfg.optimizer_config, "distill_config"):
            distill_config = cfg.optimizer_config.distill_config
            teacher_model = build_detector(distill_config.model,
                train_cfg=distill_config.train_cfg, test_cfg=distill_config.test_cfg)

            # same logic to init_detector API
            if distill_config.checkpoint is not None:
                    map_loc = "cpu"
                    checkpoint = load_checkpoint(teacher_model, distill_config.checkpoint, map_location=map_loc)
                    if 'CLASSES' in checkpoint['meta']:
                        teacher_model.CLASSES = checkpoint['meta']['CLASSES']
                    else:
                        warnings.simplefilter('once')
                        warnings.warn('Class names are not saved in the checkpoint\'s '
                                    'meta data, use COCO classes by default.')
                        teacher_model.CLASSES = get_classes('coco')
            teacher_model.cfg = distill_config  # save the config in the model for convenience
            teacher_model.to("cpu")
            teacher_model.eval()

            if verbose > 0:
                print(Fore.CYAN + "teacher model is loaded : {0}".format(distill_config.checkpoint) + Style.RESET_ALL)

            # append teacher model to list
            models.append(teacher_model)

            models = [CustomDataParallel(
                model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids) for model in models]
            model = models[0]
        elif cfg.coteaching_method=="per-object":
            models = [CustomDataParallel(
                model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids) for model in models]
            model = models[0]
        else:
            models = [MMDataParallel(
                model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids) for model in models]
            model = models[0]

    # build runner
    if "type" in cfg.optimizer_config.keys():
        if cfg.optimizer_config["type"] in ["CoteachingOptimizerHook", "DistillationOptimizerHook"]:            
            optimizers = [build_optimizer(model, cfg.optimizer) for model in models]
            runner = CooperativeTrainRunner(
                models=models,
                optimizers=optimizers,
                work_dir=cfg.work_dir,
                logger=logger,
                meta=meta)
        else:
            raise Exception("unknown OptimizerHook type={0}".format(cfg.optimizer_config["type"]))
    else:
        optimizer = build_optimizer(model, cfg.optimizer)
        runner = EpochBasedRunner(
            model,
            optimizer=optimizer,
            work_dir=cfg.work_dir,
            logger=logger,
            meta=meta)
    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config,
                                   cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    if distributed:
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(
                cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    # user-defined hooks
    if cfg.get('custom_hooks', None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(custom_hooks, list), \
            f'custom_hooks expect list type, but got {type(custom_hooks)}'
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(hook_cfg, dict), \
                'Each item in custom_hooks expects dict type, but got ' \
                f'{type(hook_cfg)}'
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
