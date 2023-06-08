import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from models.swin_transformer import SwinTransformerBlock
import numpy as np


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Decoder(nn.Module):
    def __init__(self, decoder_dim, dims, num_outputs, patch_res):
        super(Decoder, self).__init__()
        self.dims = dims
        self.decoder_dim = decoder_dim
        self.num_outputs = num_outputs
        self.patch_res = patch_res

        downsample = [nn.Sequential(nn.Conv2d(dim, decoder_dim//2, 1, bias=False),
                                    nn.BatchNorm2d(decoder_dim//2)) for i, dim in enumerate(dims)]

        self.fuse = nn.ModuleList(
            [nn.Sequential(
                torch.nn.Unflatten(
                    1, (patch_res[len(dims) - 1 - i], patch_res[len(dims) - 1 - i])),
                Reshape(-1, dim, patch_res[len(dims) -
                        1 - i], patch_res[len(dims) - 1 - i]),
                BasicBlock(dim, decoder_dim//2, downsample=downsample[i]),
                BasicBlock(decoder_dim//2, decoder_dim//2),
                nn.Upsample(scale_factor=2 ** i if i !=
                            len(dims) - 1 else 2 ** (i - 1))
            )
                for i, dim in enumerate(dims)])

        self.segment = nn.Sequential(
            nn.Conv2d(len(dims) * decoder_dim//2, decoder_dim//2, 1),
            nn.BatchNorm2d(decoder_dim//2),
            nn.ReLU(),
            nn.Conv2d(decoder_dim//2, num_outputs, 1)
        )

    def forward(self, x):
        fused = [fuse(output) for output, fuse in zip(x, self.fuse)]
        fused = torch.cat(fused, dim=1)
        return self.segment(fused)

    def flops(self):
        from ptflops import get_model_complexity_info
        flops = 0
        # fuse
        for i, fuse_block in enumerate(self.fuse):
            input_dims = (self.patch_res[len(
                self.dims) - 1 - i]*self.patch_res[len(self.dims) - 1 - i], self.dims[i])
            macs, _ = get_model_complexity_info(fuse_block, input_dims, as_strings=False,
                                                print_per_layer_stat=False, verbose=False)
            flops += macs

        # segment
        input_dims = (4 * self.decoder_dim//2,
                      self.patch_res[-1], self.patch_res[-1])
        macs, _ = get_model_complexity_info(self.segment, input_dims, as_strings=False,
                                            print_per_layer_stat=False, verbose=False)
        flops += macs
        return flops


class MultiTaskSwin(nn.Module):
    def __init__(self, encoder, decoder_config, config):
        super(MultiTaskSwin, self).__init__()

        self.backbone = encoder
        self.num_outputs = config.TASKS_CONFIG.ALL_TASKS.NUM_OUTPUT
        self.tasks = config.TASKS
        self.decoders = nn.ModuleDict()
        self.embed_dim = decoder_config['embed_dim']
        self.num_decoders = len(decoder_config['depths'])
        dims = decoder_config['dims']
        decoder_dim = decoder_config['decoder_dim']
        for task in self.tasks:
            self.decoders[task] = Decoder(decoder_dim=decoder_dim, dims=dims,
                                          patch_res=decoder_config['patch_res'], num_outputs=self.num_outputs[task])

    def forward(self, x, return_activation_stats=False, task=None):
        out_size = x.size()[2:]
        if return_activation_stats:
            shared_representation, nBs, nTs, nTsPerTask, nBsPerTask = self.backbone(x, return_activation_stats=return_activation_stats,
                                                                        return_stages=True, task=task)
        else:
            shared_representation = self.backbone(x, return_stages=True)
        result = {
            task: F.interpolate(self.decoders[task](
                shared_representation), out_size, mode='bilinear')
            for task in self.tasks
        }
        if return_activation_stats:
            return result, nBs, nTs, nTsPerTask, nBsPerTask
        else:
            return result

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True

    def freeze_task(self, task):
        for param in self.decoders[task].parameters():
            param.requires_grad = False

    def unfreeze_task(self, task):
        for param in self.decoders[task].parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        for param in self.backbone():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone():
            param.requires_grad = True

    def flops(self, images=None, logger=None, detailed=False):
        flops = self.backbone.flops(images, logger=logger, detailed=detailed)
        if detailed:
            flops, flops_per_encoder_block = flops

        temp = flops

        for task in self.decoders:
            flops += self.decoders[task].flops()

        if logger:
            logger.debug(
                f"Encoder FLOPS = {temp/1e6} - Decoder FLOPS = {(flops-temp)/1e6}")

        if detailed:
            return flops, flops_per_encoder_block
        else:
            return flops
