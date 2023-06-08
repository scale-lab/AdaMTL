# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
from torch._six import inf
import errno

from PIL import Image
import numpy as np


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger, backbone=False, quiet=False):
    logger.info(
        f"==============> Resuming form {config.MODEL.RESUME}....................")
    resume_path = config.MODEL.RESUME if not backbone else config.MODEL.RESUME_BACKBONE
    if resume_path.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            resume_path, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(resume_path, map_location='cpu')

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    if not quiet:
        logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        if not config.TRAIN.CONTROLLERS_PRETRAIN:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        logger.info(
            f"=> loaded successfully '{resume_path}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_pretrained(config, model, logger):
    logger.info(
        f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']

    # delete relative_position_index since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_position_index" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete relative_coords_table since we always re-init it
    relative_position_index_keys = [
        k for k in state_dict.keys() if "relative_coords_table" in k]
    for k in relative_position_index_keys:
        del state_dict[k]

    # delete attn_mask since we always re-init it
    attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
    for k in attn_mask_keys:
        del state_dict[k]

    # bicubic interpolate relative_position_bias_table if not match
    relative_position_bias_table_keys = [
        k for k in state_dict.keys() if "relative_position_bias_table" in k]
    for k in relative_position_bias_table_keys:
        relative_position_bias_table_pretrained = state_dict[k]
        relative_position_bias_table_current = model.state_dict()[k]
        L1, nH1 = relative_position_bias_table_pretrained.size()
        L2, nH2 = relative_position_bias_table_current.size()
        if nH1 != nH2:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                # bicubic interpolate relative_position_bias_table if not match
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                    relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
                    mode='bicubic')
                state_dict[k] = relative_position_bias_table_pretrained_resized.view(
                    nH2, L2).permute(1, 0)

    # bicubic interpolate absolute_pos_embed if not match
    absolute_pos_embed_keys = [
        k for k in state_dict.keys() if "absolute_pos_embed" in k]
    for k in absolute_pos_embed_keys:
        # dpe
        absolute_pos_embed_pretrained = state_dict[k]
        absolute_pos_embed_current = model.state_dict()[k]
        _, L1, C1 = absolute_pos_embed_pretrained.size()
        _, L2, C2 = absolute_pos_embed_current.size()
        if C1 != C1:
            logger.warning(f"Error in loading {k}, passing......")
        else:
            if L1 != L2:
                S1 = int(L1 ** 0.5)
                S2 = int(L2 ** 0.5)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(
                    -1, S1, S1, C1)
                absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(
                    0, 3, 1, 2)
                absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
                    absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(
                    0, 2, 3, 1)
                absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(
                    1, 2)
                state_dict[k] = absolute_pos_embed_pretrained_resized

    # check classifier, if not match, then re-init classifier to zero
    head_bias_pretrained = state_dict['head.bias']
    Nc1 = head_bias_pretrained.shape[0]
    Nc2 = model.head.bias.shape[0]
    if (Nc1 != Nc2):
        if Nc1 == 21841 and Nc2 == 1000:
            logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
            map22kto1k_path = f'data/map22kto1k.txt'
            with open(map22kto1k_path) as f:
                map22kto1k = f.readlines()
            map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
            state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
            state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
        else:
            torch.nn.init.constant_(model.head.bias, 0.)
            torch.nn.init.constant_(model.head.weight, 0.)
            del state_dict['head.weight']
            del state_dict['head.bias']
            logger.warning(
                f"Error in loading classifier head, re-init classifier head to 0")

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'scaler': loss_scaler.state_dict(),
                  'epoch': epoch,
                  'config': config}

    if config.TRAIN.CONTROLLERS_PRETRAIN:
        save_state = {
            'model': save_state['model']
        }

    save_name = f'ckpt_epoch_{epoch}.pth' if not config.TRAIN.CONTROLLERS_PRETRAIN else f'ckpt_epoch_{epoch}_pretrain.pth'
    save_path = os.path.join(config.OUTPUT, save_name)
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    return save_path


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d)
                                for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def ampscaler_get_grad_norm(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device)
                         for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(),
                                                        norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                # unscale the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def tens2image(tens, transpose=False):
    """Converts tensor with 2 or 3 dimensions to numpy array"""
    im = tens.cpu().detach().numpy()

    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    elif im.shape[-1] == 1:
        im = np.squeeze(im)
    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)
    if transpose:
        if im.ndim == 3:
            im = im.transpose((1, 2, 0))
    return im


def normalize(arr, t_min=0, t_max=255):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = arr.max() - arr.min()
    for i in arr:
        temp = (((i - arr.min())*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    res = np.array(norm_arr)
    return res


def save_imgs_mtl(batch_imgs, batch_labels, batch_predictions, path, id):
    import torchvision

    imgs = tens2image(batch_imgs, transpose=True)
    labels = {task: tens2image(label, transpose=True) for task, label in batch_labels.items()}
    predictions = {task: tens2image(prediction) for task, prediction in batch_predictions.items()}

    Image.fromarray(normalize(imgs, 0, 255).astype(np.uint8)).save(f'{path}/{id}_img.png')
    
    for task in labels.keys():
        if task == "semseg":
            print(np.sum(labels[task] != 255))
            labels[task] = labels[task] != 255
            predictions[task] = predictions[task] != 225
            batch_imgs = 255*(batch_imgs-torch.min(batch_imgs))/(torch.max(batch_imgs)-torch.min(batch_imgs))
            semseg = torchvision.utils.draw_segmentation_masks(batch_imgs[0].cpu().detach().to(torch.uint8), \
                                                                batch_predictions[task][0].to(torch.bool), colors="blue", alpha=0.5)
            Image.fromarray(semseg.numpy().transpose((1, 2, 0))).save(f'{path}/{id}_{task}_pred.png')
            semseg = torchvision.utils.draw_segmentation_masks(batch_imgs[0].cpu().detach().to(torch.uint8), \
                                                                batch_labels[task][0].to(torch.bool), colors="blue", alpha=0.5)
            Image.fromarray(semseg.numpy().transpose((1, 2, 0))).save(f'{path}/{id}_{task}_gt.png')
        else:
            labels[task] = normalize(labels[task], 0, 255)
            predictions[task] = normalize(predictions[task], 0, 255)    

            Image.fromarray(labels[task].astype(np.uint8)).save(f'{path}/{id}_{task}_gt.png')
            Image.fromarray(predictions[task].astype(np.uint8)).save(f'{path}/{id}_{task}_pred.png')

def get_block_select_stats(Nbs):
    num_activated_blocks = None
    num_total_blocks = 0
    
    for nb in Nbs:
        if num_activated_blocks is None:
            num_activated_blocks = torch.sum(nb, dim=-1)
        else:
            num_activated_blocks += torch.sum(nb, dim=-1)
        num_total_blocks += nb.shape[-1]

    return num_activated_blocks, num_total_blocks

def get_tokens_select_stats(NTs, weighted=False):
    weights = [96, 192, 384, 768] # token sizes for swin_t, TODO: make generic
    num_activated_tokens = None
    num_activated_tokens_per_layer = []
    num_total_tokens = 0
    
    for Nt, weight in zip(NTs, weights):
        if not weighted:
            weight = 1
            
        for head in Nt:
            if num_activated_tokens is None:
                num_activated_tokens = weight*torch.sum(head, dim=(1, 2))
            else:
                num_activated_tokens += weight*torch.sum(head, dim=(1, 2))

            num_activated_tokens_per_layer.append(weight*torch.sum(head, dim=(1, 2))/head.shape[1])
            num_total_tokens += weight*head.shape[1]

    return num_activated_tokens, num_total_tokens, num_activated_tokens_per_layer

def get_tokens_select_stats_per_task(DTs_pertask, tasks):
    num_activated_tokens_per_layer_per_task = {task:[] for task in tasks}
    
    for Dt in DTs_pertask: # stages
        for head in Dt: # blocks 
            for task in tasks: # tasks
                num_activated_tokens_per_layer_per_task[task].append(torch.sum(head[task], dim=(1, 2))/head[task].shape[1])
    return num_activated_tokens_per_layer_per_task

def get_matching_tokens_stats(DTs_pertask, tasks):
    num_matching_tokens = None
    
    for Dt in DTs_pertask: # stages
        for head in Dt: # blocks 
            tokens_match_mask = None
            for i in range(0, len(tasks)): # tasks
                if tokens_match_mask is None:
                    tokens_match_mask = head[tasks[i-1]] == head[tasks[i]]
                else:
                    tokens_match_mask &= head[tasks[i-1]] == head[tasks[i]]
            
            if num_matching_tokens is None:
                num_matching_tokens = torch.sum(tokens_match_mask, dim=(1, 2))
            else:
                num_matching_tokens += torch.sum(tokens_match_mask, dim=(1, 2))

    if num_matching_tokens is None:
        return 0
    else:
        return num_matching_tokens