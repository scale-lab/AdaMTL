#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np
from utils import get_matching_tokens_stats, get_block_select_stats, get_tokens_select_stats


class SoftMaxwithLoss(Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    def __init__(self):
        super(SoftMaxwithLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=255)

    def forward(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        # label shape batch_size x 1 x h x w
        label = label[:, 0, :, :].long()
        loss = self.criterion(self.softmax(out), label)

        return loss


class BalancedCrossEntropyLoss(Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())
        labels = torch.ge(label, 0.5).float()

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class BinaryCrossEntropyLoss(Module):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """

    def __init__(self, size_average=True, batch_average=True):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())

        labels = torch.ge(label, 0.5).float()

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)
        final_loss = loss_pos + loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class DepthLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.  
    """

    def __init__(self, loss='l1'):
        super(DepthLoss, self).__init__()
        if loss == 'l1':
            self.loss = nn.L1Loss()

        else:
            raise NotImplementedError(
                'Loss {} currently not supported in DepthLoss'.format(loss))

    def forward(self, out, label):
        mask = (label != 255)
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class NormalsLoss(Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """

    def __init__(self, size_average=True, normalize=False, norm=1):
        super(NormalsLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            # print('Using L1 loss for surface normals')
            self.loss_func = F.l1_loss
        elif norm == 2:
            # print('Using L2 loss for surface normals')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, out, label, ignore_label=255):
        assert not label.requires_grad
        mask = (label != ignore_label)
        n_valid = torch.sum(mask).item()

        if self.normalize is not None:
            out_norm = self.normalize(out)
            loss = self.loss_func(torch.masked_select(
                out_norm, mask), torch.masked_select(label, mask), reduction='sum')
        else:
            loss = self.loss_func(torch.masked_select(
                out, mask), torch.masked_select(label, mask), reduction='sum')

        if self.size_average:
            if ignore_label:
                ret_loss = torch.div(loss, max(n_valid, 1e-6))
                return ret_loss
            else:
                ret_loss = torch.div(loss, float(np.prod(label.size())))
                return ret_loss

        return loss


class SingleTaskLoss(nn.Module):
    def __init__(self, loss_ft, task):
        super(SingleTaskLoss, self).__init__()
        self.loss_ft = loss_ft
        self.task = task

    def forward(self, pred, gt):
        out = {self.task: self.loss_ft(pred[self.task], gt[self.task])}
        out['total'] = out[self.task]
        return out


class MultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict):
        super(MultiTaskLoss, self).__init__()
        assert (set(tasks) == set(loss_ft.keys()))
        assert (set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        out = {
            task: self.loss_ft[task](pred[task], gt[task]) for task in self.tasks
        }
        out['total'] = torch.sum(torch.stack(
            [self.loss_weights[t] * out[t] for t in self.tasks]))
        return out['total'], out


class PADNetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict,
                 loss_weights: dict):
        super(PADNetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        total = 0.
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]

        # Losses initial task predictions (deepsup)
        for task in self.auxilary_tasks:
            pred_ = F.interpolate(
                pred['initial_%s' % (task)], img_size, mode='bilinear')
            gt_ = gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out['deepsup_%s' % (task)] = loss_
            total += self.loss_weights[task] * loss_

        # Losses at output
        for task in self.tasks:
            pred_, gt_ = pred[task], gt[task]
            loss_ = self.loss_ft[task](pred_, gt_)
            out[task] = loss_
            total += self.loss_weights[task] * loss_

        out['total'] = total

        return out['total'], out


class MTINetLoss(nn.Module):
    def __init__(self, tasks: list, auxilary_tasks: list, loss_ft: nn.ModuleDict,
                 loss_weights: dict):
        super(MTINetLoss, self).__init__()
        self.tasks = tasks
        self.auxilary_tasks = auxilary_tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights

    def forward(self, pred, gt):
        total = 0.
        out = {}
        img_size = gt[self.tasks[0]].size()[-2:]

        # Losses initial task predictions at multiple scales (deepsup)
        for scale in range(4):
            pred_scale = pred['deep_supervision']['scale_%s' % (scale)]
            pred_scale = {t: F.interpolate(
                pred_scale[t], img_size, mode='bilinear') for t in self.auxilary_tasks}
            losses_scale = {t: self.loss_ft[t](
                pred_scale[t], gt[t]) for t in self.auxilary_tasks}
            for k, v in losses_scale.items():
                out['scale_%d_%s' % (scale, k)] = v
                total += self.loss_weights[k] * v

        # Losses at output
        losses_out = {task: self.loss_ft[task](
            pred[task], gt[task]) for task in self.tasks}
        for k, v in losses_out.items():
            out[k] = v
            total += self.loss_weights[k] * v

        out['total'] = total

        return out['total'], out


""" 
    Loss functions 
"""


def get_loss(config, task=None):
    """ Return loss function for a specific task """
    if task == 'edge':
        criterion = BalancedCrossEntropyLoss(
            size_average=True, pos_weight=config['edge_w'])

    elif task == 'semseg' or task == 'human_parts':
        criterion = SoftMaxwithLoss()

    elif task == 'normals':
        criterion = NormalsLoss(normalize=True, size_average=True, norm=1)

    elif task == 'sal':
        criterion = BalancedCrossEntropyLoss(size_average=True)

    elif task == 'depth':
        criterion = DepthLoss('l1')

    else:
        raise NotImplementedError('Undefined Loss: Choose a task among '
                                  'edge, semseg, human_parts, sal, depth, or normals')

    return criterion


class ControllersLoss(nn.Module):
    def __init__(self, policy='per_task'):
        super(ControllersLoss, self).__init__()
        self.policy = policy
        self.loss_fct = nn.L1Loss()

    def forward(self, Nbs, NTs):
        loss = 0
        if self.policy == 'per_task':
            # activated blocks loss
            for Nb in Nbs:
                for mask in Nb.values():
                    loss += self.loss_fct(mask, torch.ones_like(mask))

            # activated tokens loss
            for _NTs in NTs:
                for NT in _NTs:
                    for mask in NT.values():
                        loss += self.loss_fct(mask, torch.ones_like(mask))
        else:
            # activated blocks loss
            for Nb in Nbs:
                loss += self.loss_fct(Nb, torch.ones_like(Nb))

            # activated tokens loss
            for _NTs in NTs:
                for NT in _NTs:
                    loss += self.loss_fct(NT, torch.ones_like(NT))

        return loss


class MultiObjectiveMultiTaskLoss(nn.Module):
    def __init__(self, tasks: list, loss_ft: nn.ModuleDict, loss_weights: dict,
                 efficiency_weight: float = 1, tokens_efficiency_target: float = 0.5,
                 blocks_efficiency_target: float = 0.9,
                 policy="single", ada_blocks: bool = False, ada_tokens: bool = False,
                 pertask_overlap_weight: float = 0, per_layer_loss=False, weighted_tokens=False,
                 mixed_efficiency_loss: bool = False, is_kd=False, kd_weight=0.1):
        super(MultiObjectiveMultiTaskLoss, self).__init__()
        assert (set(tasks) == set(loss_ft.keys()))
        assert (set(tasks) == set(loss_weights.keys()))
        self.tasks = tasks
        self.loss_ft = loss_ft
        self.loss_weights = loss_weights
        # weight of efficiency loss compared to model loss
        self.efficiency_weight = efficiency_weight
        # target % of the blocks to be executed
        self.tokens_efficiency_target = tokens_efficiency_target
        self.blocks_efficiency_target = blocks_efficiency_target
        self.pertask_overlap_weight = pertask_overlap_weight
        self.weighted_tokens = weighted_tokens
        self.per_layer_loss = per_layer_loss
        self.mixed_efficiency_loss = mixed_efficiency_loss
        self.policy = policy
        self.ada_blocks = ada_blocks
        self.ada_tokens = ada_tokens

        self.is_kd = is_kd
        self.kd_weight = kd_weight
        self.efficiency_loss_fct = nn.MSELoss()
        self.kd_loss_fct = nn.KLDivLoss()

    def forward(self, pred, gt, Dbs, DTs, DTs_pertask, teacher_pred=None, task=None):
        # DTs is a list of lists with an activated tokens mask for each layer in each stage
        # Note that: swin transformer consists of 4 stages each has a various number of layers/blocks
        out = {
            task: self.loss_ft[task](pred[task], gt[task]) for task in (self.tasks if task is None else [task])
        }
        student_loss = out['total'] = torch.sum(torch.stack(
            [self.loss_weights[t] * out[t] for t in (self.tasks if task is None else [task])]))

        num_activated_blocks, num_total_blocks = get_block_select_stats(Dbs)

        if self.policy == "single":
            num_activated_tokens, num_total_tokens, per_layer_tokens = get_tokens_select_stats(
                DTs, weighted=self.weighted_tokens)
            per_layer_tokens = torch.stack(per_layer_tokens).permute(1, 0).to()

        elif self.policy == "per_task":
            if task is None:
                num_activated_tokens, num_total_tokens, per_layer_tokens = get_tokens_select_stats(
                    DTs, weighted=self.weighted_tokens)
                num_matching_tokens = get_matching_tokens_stats(
                    DTs_pertask, self.tasks)
                per_layer_tokens = torch.stack(
                    per_layer_tokens).permute(1, 0).to()
            else:
                # This is activated when USE_SINGLE_TASK_LOSS is true
                for Dt in DTs_pertask:
                    for head in Dt:
                        num_activated_tokens += torch.sum(
                            head[task], dim=(1, 2))
                        num_total_tokens += head[task].shape[1]

        else:
            raise ValueError(f"{self.policy} is not a defined mode")

        percentage_activated_layers = num_activated_blocks/num_total_blocks
        percentage_activated_tokens = num_activated_tokens/num_total_tokens

        efficiency_loss = 0
        if self.ada_blocks:
            # layers loss
            efficiency_loss += self.efficiency_loss_fct(percentage_activated_layers,
                                                        self.blocks_efficiency_target*torch.ones_like(percentage_activated_layers))
        if self.ada_tokens:
            # tokens loss
            if self.mixed_efficiency_loss:
                efficiency_loss += 0.5*self.efficiency_loss_fct(per_layer_tokens,
                                                                self.tokens_efficiency_target*torch.ones_like(per_layer_tokens))
                efficiency_loss += 0.5*self.efficiency_loss_fct(percentage_activated_tokens,
                                                                self.tokens_efficiency_target*torch.ones_like(percentage_activated_tokens))
            elif self.per_layer_loss:
                efficiency_loss += self.efficiency_loss_fct(per_layer_tokens,
                                                            self.tokens_efficiency_target*torch.ones_like(per_layer_tokens))
            else:
                efficiency_loss += self.efficiency_loss_fct(percentage_activated_tokens,
                                                            self.tokens_efficiency_target*torch.ones_like(percentage_activated_tokens))

            # tokens overlap loss
            if self.policy == "per_task" and abs(self.pertask_overlap_weight) > 0:
                percentage_matching_tokens = num_matching_tokens/num_total_tokens
                efficiency_loss += self.pertask_overlap_weight*self.efficiency_loss_fct(percentage_matching_tokens,
                                                                                        torch.ones_like(percentage_matching_tokens))

        kd_loss = None
        if self.is_kd:
            kd_loss_dict = {
                task: self.kd_loss_fct(torch.nn.functional.log_softmax(pred[task]), torch.nn.functional.softmax(teacher_pred[task])) for task in self.tasks
            }
            kd_loss = torch.sum(torch.stack(
                [self.loss_weights[t] * kd_loss_dict[t] for t in self.tasks]))
            loss = self.kd_weight * student_loss +\
                (1 - self.kd_weight) * kd_loss +\
                self.efficiency_weight*efficiency_loss
        else:
            loss = student_loss + self.efficiency_weight*efficiency_loss
        return loss, student_loss, self.efficiency_weight*efficiency_loss, out, kd_loss
