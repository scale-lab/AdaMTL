# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import AverageMeter

from config import get_config
from models import build_model, build_mtl_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper
from utils import get_matching_tokens_stats, get_block_select_stats, get_tokens_select_stats, get_tokens_select_stats_per_task, save_imgs_mtl

from mtl_loss_schemes import MultiTaskLoss, get_loss, ControllersLoss, MultiObjectiveMultiTaskLoss
from evaluation.evaluate_utils import PerformanceMeter, get_output
from ptflops import get_model_complexity_info


def parse_option():
    parser = argparse.ArgumentParser(
        'Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True,
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int,
                        help="batch size for single GPU")
    parser.add_argument('--ckpt-freq', type=int, default=5,
                        help="checkpoint saving frequency")
    parser.add_argument('--eval-freq', type=int, default=5,
                        help="model evaluation frequency")
    parser.add_argument('--epochs', type=int, default=300,
                        help="number of epochs to train")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true',
                        help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true',
                        help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--name', type=str, help='override model name')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')
    
    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm',
                        action='store_true', help='Use fused layernorm.')
    # overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    # MTL Config
    parser.add_argument('--tasks', type=str, default='depth',
                        help='Enable adaptive MTL, defaults to depth est.')
    parser.add_argument(
        '--nyud', type=str, help='specify the path to load NYUD, replaces --data-path')
    parser.add_argument(
        '--pascal', type=str, help='specify the path to load PASCAL, replaces --data-path and --nyud')
    parser.add_argument('--eval-training-freq', type=int,
                        help='calculate performance score on the training dataset')
    parser.add_argument('--resume-backbone',
                        help='resume checkpoint into the backbone')
    parser.add_argument(
        '--resume-teacher', help='resume teacher checkpoint for knowledge distillation')
    parser.add_argument('--freeze-backbone',
                        action='store_true', help='Freeze encoder layers.')

    # Efficiency configs
    parser.add_argument("--blocks_efficiency_target", type=float,
                        help="Target activated blocks percentage.")
    parser.add_argument("--tokens_efficiency_target", type=float,
                        help="Target activated tokens percentage.")
    parser.add_argument("--efficiency_weight", type=float,
                        help="Efficiency weight during training.")
    parser.add_argument("--pertask_overlap_weight", type=float,
                        help="Weight for pertask overlap during pertask policy training.")
    parser.add_argument('--skip_initial_validation', action='store_true',
                        help='Skip running validation at the start')
    parser.add_argument('--combined_fine_tune_only', action='store_true',
                        help='Skip any steps before the combined fine tune')
    parser.add_argument('--compute_flops', action='store_true',
                        help='Compute Flops while evaluating.')
    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    _, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(
        config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    is_kd = config.TRAIN.TRAIN_MODE == 'k_distil'
    teacher = None
    model = build_model(config)
    if config.MTL:
        model = build_mtl_model(model, config)
    if is_kd:
        teacher = build_model(config, is_teacher=True)
        teacher = build_mtl_model(teacher, config, is_teacher=True)
        teacher.cuda()

    n_parameters = sum(p.numel() for p in model.parameters())
    logger.info(f"number of params: {n_parameters / 1e6} M")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GMACs: {flops / 1e9}")

    model.cuda()

    macs, params = get_model_complexity_info(model, (3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), as_strings=False,
                                             print_per_layer_stat=False, verbose=False)

    logger.info(f"ptflops GMACS = {macs / 1e9} and params = {params/1e6} M")

    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    is_per_task = config.TRAIN.POLICY == 'per_task'

    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(
            data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(
            config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(
            smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    if config.MTL:
        loss_ft = torch.nn.ModuleDict(
            {task: get_loss(config['TASKS_CONFIG'], task) for task in config.TASKS})
        all_loss_weights = {
            'depth': 1.0,
            'semseg': 1.0,
            'human_parts': 2.0,
            'sal': 5.0,
            'edge': 50.0,
            'normals': 10.0,
        }
        loss_weights = {}
        for t in config.TASKS:
            loss_weights[t] = all_loss_weights[t]

        if config.TRAIN.MTL_MULTI_OBJECTIVE_TRAIN:
            criterion = MultiObjectiveMultiTaskLoss(config.TASKS, loss_ft, loss_weights,
                                                    efficiency_weight=config.TRAIN.EFFICIENCY_WEIGHT,
                                                    blocks_efficiency_target=config.TRAIN.BLOCKS_EFFICIENCY_TARGET,
                                                    tokens_efficiency_target=config.TRAIN.TOKENS_EFFICIENCY_TARGET,
                                                    per_layer_loss=config.TRAIN.PER_LAYER_EFFICIENCY_LOSS,
                                                    policy=config.TRAIN.POLICY,
                                                    ada_tokens=config.TRAIN.ADA_TOKENS,
                                                    ada_blocks=config.TRAIN.ADA_BLOCKS,
                                                    weighted_tokens=config.TRAIN.WEIGHTED_TOKENS_LOSS,
                                                    mixed_efficiency_loss=config.TRAIN.MIXED_PER_LAYER_EFFICIENCY_LOSS,
                                                    pertask_overlap_weight=config.TRAIN.PERTASK_OVERLAP_WEIGHT,
                                                    is_kd=config.TRAIN.TRAIN_MODE == 'k_distil',
                                                    kd_weight=config.TRAIN.KD_LOSS_WEIGHT)
        else:
            criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)

    if config.TRAIN.CONTROLLERS_PRETRAIN:
        criterion = ControllersLoss(policy=config.TRAIN.POLICY)

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(
                    f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(
                f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(
            config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)

        if not config.SKIP_INITIAL_EVAL:
            validate(config, data_loader_val, model)
        if config.EVAL_MODE:
            return

    if config.MODEL.RESUME_BACKBONE:
        max_accuracy = load_checkpoint(
            config, model_without_ddp.backbone, optimizer, lr_scheduler, loss_scaler, logger, True)
        if not config.SKIP_INITIAL_EVAL:
            validate(config, data_loader_val, model)
        if config.EVAL_MODE:
            return

    if teacher is not None:
        print("loading teacher.......")
        load_checkpoint(config, teacher, optimizer, lr_scheduler,
                        loss_scaler, logger, quiet=True)

    # TODO: make it work with MTL
    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        if not config.SKIP_INITIAL_EVAL:
            acc1, _, _ = validate(config, data_loader_val, model)

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()
    is_per_task = config.TRAIN.POLICY == 'per_task'
    if is_per_task:
        task_epochs = config.TRAIN.TASK_EPOCHS
        if config.TRAIN.COMBINED_FINE_TUNE:
            task_epochs = max(task_epochs, config.TRAIN.EPOCHS)
    total_epochs = 0
    # No per task and KD for now
    assert not (is_per_task and (config.TRAIN.TRAIN_MODE == 'k_distil'))

    if is_per_task and not config.TRAIN.CONTROLLERS_PRETRAIN:
        if not args.combined_fine_tune_only:
            if config.TRAIN.ALTERNATING_TASK_TRAINING:
                for epoch in range(config.TRAIN.START_EPOCH, task_epochs):
                    if not config.MTL:
                        data_loader_train.sampler.set_epoch(epoch)
                    task = config.TASKS[epoch % len(config.TASKS)]
                    print("Training task", task)
                    for task_it in config.TASKS:
                        if task == task_it:
                            print("Unfreezing", task_it)
                            model.unfreeze_task(task_it)
                        else:
                            print("Freezing", task_it)
                            model.freeze_task(task_it)
                    n_parameters = sum(p.numel()
                                    for p in model.parameters() if p.requires_grad)
                    logger.info(f"number of params: {n_parameters}")

                    train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                                    loss_scaler, task=task)
                    if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                        save_path = save_checkpoint(config, total_epochs, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                                        logger)
                    if not config.MTL:
                        acc1, _, _ = validate(
                            config, data_loader_val, model)
                        max_accuracy = max(max_accuracy, acc1)
                    elif (epoch % config.EVAL_FREQ == 0):
                        validate(config, data_loader_val, model)
                    total_epochs = total_epochs + 1
            else:
                for task in config.TASKS:  # Train policy networks
                    logger.info(f"Freezing all")
                    model.freeze_all()
                    n_parameters = sum(p.numel()
                                    for p in model.parameters() if p.requires_grad)
                    logger.info(f"number of params: {n_parameters}")

                    logger.info(f"Unfreezing controller: {task}")
                    model.backbone.unfreeze_controllers(task)
                    n_parameters = sum(p.numel()
                                    for p in model.parameters() if p.requires_grad)
                    logger.info(f"number of params: {n_parameters}")

                    logger.info(f"Unfreezing decoders: {task}")
                    model.unfreeze_task(task)
                    n_parameters = sum(p.numel()
                                    for p in model.parameters() if p.requires_grad)
                    logger.info(f"number of params: {n_parameters}")

                    logger.info(f"Training policy network for task: {task}")
                    for epoch in range(config.TRAIN.START_EPOCH, task_epochs):
                        if not config.MTL:
                            data_loader_train.sampler.set_epoch(epoch)

                        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                                        loss_scaler, task=task)
                        if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                            save_path = save_checkpoint(config, total_epochs, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                                            logger)
                        if not config.MTL:
                            acc1, _, _ = validate(
                                config, data_loader_val, model)
                            max_accuracy = max(max_accuracy, acc1)
                        elif (epoch % config.EVAL_FREQ == 0):
                            validate(config, data_loader_val, model)
                        total_epochs = total_epochs + 1
        if config.TRAIN.COMBINED_FINE_TUNE:
            # Combined policy training
            logger.info(f"Unfreezing all")
            model.unfreeze_all()
            n_parameters = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
            logger.info(f"number of params: {n_parameters}")

            for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
                logger.info(f"Training combined policy network")
                if not config.MTL:
                    data_loader_train.sampler.set_epoch(epoch)

                train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                                loss_scaler)
                if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                    save_path = save_checkpoint(config, total_epochs, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                                    logger)
                if not config.MTL:
                    acc1, _, _ = validate(config, data_loader_val, model)
                    logger.info(
                        f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
                    max_accuracy = max(max_accuracy, acc1)
                    logger.info(f'Max accuracy: {max_accuracy:.2f}%')
                elif (epoch % config.EVAL_FREQ == 0):
                    validate(config, data_loader_val, model)
                total_epochs = total_epochs + 1
    else:
        for epoch in range(config.TRAIN.EPOCHS):
            if not config.MTL:
                data_loader_train.sampler.set_epoch(epoch)

            train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                            loss_scaler, teacher=teacher)
            if (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_path = save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                                logger)
            if epoch % config.EVAL_FREQ == 0:
                if config.MTL:
                    validate(config, data_loader_val, model)
                else:
                    acc1, _, _ = validate(config, data_loader_val, model)
                    max_accuracy = max(max_accuracy, acc1)

    # final eval
    validate(config, data_loader_val, model)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))
    if config.TRAIN.CONTROLLERS_PRETRAIN:
        del model
        logger.info("Adaptive evaluation")
        config.defrost()
        config.TRAIN.ADAPTIVE = True
        config.TRAIN.HARD_GUMBEL = True
        config.MODEL.RESUME = save_path
        config.freeze()
        logger.info(f"Loading last checkpoint {save_path}")
        model = build_mtl_model(build_model(config), config)
        load_checkpoint(
            config, model, optimizer, lr_scheduler, loss_scaler, logger)
        n_parameters = sum(p.numel() for p in model.parameters())
        logger.info(f"number of params: {n_parameters / 1e6} M")
        if hasattr(model, 'flops'):
            flops = model.flops()
            logger.info(f"number of GMACs: {flops / 1e9}")
        model.cuda()
        macs, params = macs, params = get_model_complexity_info(
            model, (3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE), as_strings=False, print_per_layer_stat=False, verbose=False)
        logger.info(
            f"ptflops GMACS = {macs / 1e9} and params = {params/1e6} M")
        validate(config, data_loader_val, model)


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler, task=None, teacher=None):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()
    model_loss_meter = AverageMeter()
    efficiency_loss_meter = AverageMeter()
    kd_loss_meter = AverageMeter()

    performance_meter = PerformanceMeter(config, 'PASCALContext' if config.get(
        'DATA', {}).get('PASCAL', False) else 'NYUD')

    start = time.time()
    end = time.time()

    for idx, batch in enumerate(data_loader):
        if not config.MTL:
            samples, targets = batch
            samples = samples.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)
        else:
            samples = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(
                non_blocking=True) for task in config.TASKS}

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
        teacher_outputs = None
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            if config.TRAIN.CONTROLLERS_PRETRAIN or config.TRAIN.MTL_MULTI_OBJECTIVE_TRAIN:
                outputs, NBs, NTs, nTsPerTask, nBsPerTask = model(
                    samples, return_activation_stats=True, task=task)
                if teacher is not None:
                    teacher_outputs = teacher(samples)
            else:
                outputs = model(samples)

            if config.TRAIN.CONTROLLERS_PRETRAIN:
                loss = criterion(nBsPerTask if config.TRAIN.POLICY == 'per_task' else NBs,
                                 nTsPerTask if config.TRAIN.POLICY == 'per_task' else NTs)

            elif config.TRAIN.MTL_MULTI_OBJECTIVE_TRAIN:
                loss_task = task if (
                    config.TRAIN.ALTERNATING_TASK_TRAINING and config.TRAIN.USE_SINGLE_TASK_LOSS) else None
                loss, loss_model, loss_efficiency, loss_dict, kd_loss = criterion(
                    pred=outputs, gt=targets, Dbs=NBs, DTs=NTs,
                    DTs_pertask=nTsPerTask, teacher_pred=teacher_outputs, task=loss_task)
            else:
                loss, loss_dict = criterion(outputs, targets)

        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update(
                (epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if not config.MTL:
            loss_meter.update(loss.item(), targets.size(0))
        else:
            if config.TRAIN.MTL_MULTI_OBJECTIVE_TRAIN:
                model_loss_meter.update(loss_model.item())
                efficiency_loss_meter.update(loss_efficiency.item())
                if kd_loss is not None:
                    kd_loss_meter.update(kd_loss.item())

            loss_meter.update(loss.item())

        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
            if config.TRAIN.MTL_MULTI_OBJECTIVE_TRAIN:
                logger.info(
                    f'model loss {model_loss_meter.val:.4f} ({model_loss_meter.avg:.4f})\t'
                    f'efficiency loss {efficiency_loss_meter.val:.4f} ({efficiency_loss_meter.avg:.4f})')
                if config.TRAIN.TRAIN_MODE == 'k_distil':
                    logger.info(
                        f'kd loss {kd_loss_meter.val:.4f}')

    if config.EVAL_TRAINING is not None and (epoch % config.EVAL_TRAINING == 0):
        print("Training Eval:")
        performance_meter.update(
            {t: get_output(outputs[t], t) for t in config.TASKS}, targets)

        _ = performance_meter.get_score(verbose=True)

    epoch_time = time.time() - start
    logger.info(
        f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model):
    """ Evaluate model in an online fashion without storing the predictions to disk """
    tasks = config.TASKS
    performance_meter = PerformanceMeter(config, 'PASCALContext' if config.get(
        'DATA', {}).get('PASCAL', False) else 'NYUD')
    loss_meter = AverageMeter()

    loss_ft = torch.nn.ModuleDict(
        {task: get_loss(config['TASKS_CONFIG'], task) for task in config.TASKS})
    all_loss_weights = {
        'depth': 1.0,
        'semseg': 1.0,
        'human_parts': 2.0,
        'sal': 5.0,
        'edge': 50.0,
        'normals': 10.0,
    }
    loss_weights = {}
    for t in config.TASKS:
        loss_weights[t] = all_loss_weights[t]
    criterion = MultiTaskLoss(config.TASKS, loss_ft, loss_weights)

    model.eval()
    all_blocks = 0
    activated_blocks = 0
    all_tokens = 0.0
    activated_tokens = 0.0
    matching_tokens = 0.0
    total_flops = 0
    num_val_points = 0
    logger.info("Start eval")
    start = time.time()
    outputs_batch = {task: [] for task in config.TASKS}
    labels_batch = {task: [] for task in config.TASKS}
    for i, batch in enumerate(data_loader):
        # Forward pass
        logger.debug(f"Image ID = {batch['meta']['image']}")
        images = batch['image'].cuda(non_blocking=True)
        targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}

        if config.TRAIN.MTL_MULTI_OBJECTIVE_TRAIN or config.TRAIN.CONTROLLERS_PRETRAIN:
            output, num_blocks, num_tokens, nTsPerTask, nBsPerTask = model(
                images, return_activation_stats=True)
            for t in output:
                outputs_batch[t].append(output[t])
                labels_batch[t].append(targets[t])
            ab, tb = get_block_select_stats(num_blocks)
            at, tt, at_per_layer = get_tokens_select_stats(num_tokens)
            if config.TRAIN.POLICY == 'per_task':
                at_per_layer_per_task = get_tokens_select_stats_per_task(
                    nTsPerTask, config.TASKS)
                matching_tokens += get_matching_tokens_stats(
                    nTsPerTask, config.TASKS)

            logger.debug(
                f"Activated Tokens Percentage = {(100*at/tt).item():.2f}, Activated Blocks Percentage = {(100*ab/tb).item():.2f}")
            if args.compute_flops:
                flops, flops_per_layer = model.flops(
                    images, logger=logger, detailed=True)
                total_flops += flops
                num_val_points += 1
                logger.debug(f"Total flops {flops/1e6}")

                def flatten(l):
                    return [item for sublist in l for item in sublist]
                logger.debug(f"Flops per layer = {flatten(flops_per_layer)}")
            if config.TRAIN.POLICY == 'per_task':
                logger.debug(
                    f"Activated Tokens per Layer per task {at_per_layer_per_task}")
            logger.debug(f"Activated Tokens per Layer {at_per_layer}")

            activated_tokens += at
            all_tokens += tt
            activated_blocks += ab
            all_blocks += tb

        else:
            output = model(images)
            for t in output:
                outputs_batch[t].append(output[t])
                labels_batch[t].append(targets[t])

        if len(outputs_batch[config.TASKS[0]]) > 0 and len(outputs_batch[config.TASKS[0]]) % config.DATA.BATCH_SIZE == 0:
            output_batch_tesnor = {task: torch.cat(
                task_batch, dim=0) for task, task_batch in outputs_batch.items()}
            label_batch_tesnor = {task: torch.cat(
                task_batch, dim=0) for task, task_batch in labels_batch.items()}

            # Measure performance
            with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
                loss, loss_dict = criterion(
                    output_batch_tesnor, label_batch_tesnor)
                loss_meter.update(loss.item())
            processed_output = {t: get_output(
                output_batch_tesnor[t], t) for t in tasks}
            performance_meter.update(processed_output, label_batch_tesnor)

            outputs_batch = {task: [] for task in config.TASKS}
            labels_batch = {task: [] for task in config.TASKS}

    if len(outputs_batch[config.TASKS[0]]) > 0:
        output_batch_tesnor = {task: torch.cat(
            task_batch, dim=0) for task, task_batch in outputs_batch.items()}
        label_batch_tesnor = {task: torch.cat(
            task_batch, dim=0) for task, task_batch in labels_batch.items()}

        # Measure performance
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            loss, loss_dict = criterion(
                output_batch_tesnor, label_batch_tesnor)
            loss_meter.update(loss.item())
        processed_output = {t: get_output(
            output_batch_tesnor[t], t) for t in tasks}
        performance_meter.update(processed_output, label_batch_tesnor)

        # save_imgs_mtl(images, targets, processed_output, "adamtl", id=batch['meta']['image'][0])

    logger.info(f"val loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t")

    if config.TRAIN.ADA_TOKENS or config.TRAIN.ADA_BLOCKS:
        logger.info(f"Activated blocks {100*activated_blocks/all_blocks}%\t")
        logger.info(
            f"% Activated tokens {100.0*activated_tokens/all_tokens}%\t")
        logger.info(f"# Activated tokens {activated_tokens}/{all_tokens}\t")
        logger.info(
            f"% Matching tokens {(100.0*matching_tokens/all_tokens)}%\t")
        logger.info(f"# Matching tokens {matching_tokens}/{all_tokens}\t")
        if args.compute_flops:
            logger.info(
                f"# Total Flops {(total_flops/num_val_points)/1e9}GFLOPS\t")

    eval_results = performance_meter.get_score(verbose=True)
    epoch_time = time.time() - start
    logger.info(
        f"eval takes {datetime.timedelta(seconds=int(epoch_time))}")

    return eval_results


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(
            f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")


    seed = config.SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * \
        config.DATA.BATCH_SIZE / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * \
        config.DATA.BATCH_SIZE / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * \
        config.DATA.BATCH_SIZE / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    if config.TRAIN.CONTROLLERS_PRETRAIN:
        config.TRAIN.WARMUP_EPOCHS = 0
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)
