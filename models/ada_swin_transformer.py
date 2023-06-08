import copy
from .swin_transformer import SwinTransformer, BasicLayer, \
    SwinTransformerBlock, window_partition, window_reverse
from .controllers import BlockSelect, TokensSelect
import torch.nn as nn
import torch
import typing
from functools import partial


class AdaSwinTransformerBlock(SwinTransformerBlock):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__(dim, input_resolution, num_heads, window_size=window_size, shift_size=shift_size,
                         mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop, attn_drop=attn_drop, drop_path=drop_path,
                         act_layer=act_layer, norm_layer=norm_layer, fused_window_process=fused_window_process)

    def forward(self, x, tokens_mask=None):
        # if no tokens is activated, skip the whole block
        if tokens_mask is not None and torch.sum(tokens_mask) == 0.:
            return x

        H, W = self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = torch.roll(
                    x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                # partition windows
                # nW*B, window_size, window_size, C
                x_windows = window_partition(shifted_x, self.window_size)
            else:
                x_windows = WindowProcess.apply(
                    x, B, H, W, C, -self.shift_size, self.window_size)
        else:
            shifted_x = x
            # partition windows
            # nW*B, window_size, window_size, C
            x_windows = window_partition(shifted_x, self.window_size)

        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            if not self.fused_window_process:
                shifted_x = window_reverse(
                    attn_windows, self.window_size, H, W)  # B H' W' C
                x = torch.roll(shifted_x, shifts=(
                    self.shift_size, self.shift_size), dims=(1, 2))
            else:
                x = WindowProcessReverse.apply(
                    attn_windows, B, H, W, C, self.shift_size, self.window_size)
        else:
            shifted_x = window_reverse(
                attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

        # FFN
        temp = self.norm2(x)
        if tokens_mask is not None:
            B, _, E_sz = temp.shape
            tokens_mask = tokens_mask.expand(-1, -1, E_sz)

            if self.training:
                # During training zero-out the non-chosen tokens to
                # keep the tensor size similar for batching/paralleization purposes
                selected_tokens = temp * tokens_mask
                processed_selected_tokens = self.mlp(selected_tokens)
                mask_inv = torch.ones_like(tokens_mask) - tokens_mask
                temp = (processed_selected_tokens *
                        tokens_mask) + (temp * mask_inv)
            else:
                tokens_mask = tokens_mask.bool()
                selected_tokens = temp[tokens_mask].view(B, -1, E_sz)
                processed_selected_tokens = self.mlp(selected_tokens)
                temp[tokens_mask] = processed_selected_tokens.view(-1)
        else:
            temp = self.mlp(temp)

        x = x + self.drop_path(temp)

        return x

    def flops(self, tokens_mask=None, logger=None):
        if tokens_mask is not None and torch.sum(tokens_mask) == 0.:
            return 0
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        if tokens_mask is None:
            flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        else:
            activated_tokens = torch.sum(tokens_mask).item()
            if logger:
                logger.debug(
                    f"\t\t\t% of activated tokens = {100*activated_tokens/(H*W):.2f} - {activated_tokens} - {H*W} - {tokens_mask.shape}")
            flops += 2 * activated_tokens * self.dim * self.dim * self.mlp_ratio

        # norm2
        flops += self.dim * H * W
        return flops


class AdaBasicLayer(BasicLayer):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 fused_window_process=False, ada_blocks=False, ada_tokens=False, policy_mode='single',
                 token_ctrl_size: int = 0, tasks: typing.Optional[list] = None, hard_gumbel: bool = True, random=False):
        super().__init__(dim, input_resolution, depth, num_heads, window_size,
                         mlp_ratio, qkv_bias, qk_scale, drop, attn_drop,
                         drop_path, norm_layer, downsample, use_checkpoint,
                         fused_window_process)

        self.ada_blocks = ada_blocks
        self.ada_tokens = ada_tokens
        self.policy_mode = policy_mode

        self.tasks = tasks
        self.token_ctrl_size = token_ctrl_size
        self.random = random
        self.hard_gumbel = hard_gumbel

        (W, H) = input_resolution

        if self.policy_mode == 'single':
            self.tokens_select = self._generate_token_select_single_policy(
                dim, W*H)
        elif self.policy_mode == 'per_task':
            assert len(tasks) > 0
            self.tokens_select = self._generate_token_select_per_task_policy(
                dim, W*H)

        # build blocks
        self.blocks = nn.ModuleList([
            AdaSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    shift_size=0 if (
                                        i % 2 == 0) else window_size // 2,
                                    mlp_ratio=mlp_ratio,
                                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=drop, attn_drop=attn_drop,
                                    drop_path=drop_path[i] if isinstance(
                                        drop_path, list) else drop_path,
                                    norm_layer=norm_layer,
                                    fused_window_process=fused_window_process)
            for i in range(depth)])

    def _generate_token_select_single_policy(self, dim, WH):
        if not self.ada_tokens:
            return None
        return TokensSelect(
            dim, WH, hidden_sz=self.token_ctrl_size, random=self.random)

    def _generate_token_select_per_task_policy(self, dim, WH):
        if not self.ada_tokens:
            return None
        return nn.ModuleDict({
            task: self._generate_token_select_single_policy(dim, WH)
            for task in self.tasks
        })

    def _generate_tokens_mask(self, x, task):
        curr_task_mask = None
        if self.policy_mode == 'single':
            tokens_mask = self.tokens_select(x)
        elif self.policy_mode == "per_task":
            if task is not None:
                tokens_mask = self.tokens_select[task](
                    x, hard=self.hard_gumbel)
                curr_task_mask = {}
                for tsk in self.tasks:
                    if tsk == task:
                        curr_task_mask[tsk] = tokens_mask
                    else:
                        curr_task_mask[tsk] = self.tokens_select[tsk](
                            x, hard=self.hard_gumbel)
            else:
                tokens_mask = None
                curr_task_mask = {}
                for tsk, ctrl in self.tokens_select.items():
                    if tokens_mask is None:
                        tokens_mask = ctrl(x, hard=self.hard_gumbel)
                        curr_task_mask[tsk] = tokens_mask
                    else:
                        # task is task name
                        task_mask = ctrl(x, hard=self.hard_gumbel)
                        curr_task_mask[tsk] = task_mask
                        tokens_mask = torch.clamp_max(
                            tokens_mask + task_mask, 1.0)
        else:
            raise ValueError(
                f"{self.policy_mode} must be one of \"single\", \"per_task\"")

        return tokens_mask, curr_task_mask

    def forward(self, x, layer_mask=None, adaptive=False, task=None):
        # "layer_mask" is a mask to choose the active layers
        activated_tokens = []
        task_masks = []
        if self.ada_tokens or self.ada_blocks:
            if self.training:
                _, P, E = x.shape
                # assert layer_mask.shape[1] == len(self.blocks)
                if layer_mask is not None:
                    layer_mask = layer_mask.unsqueeze(
                        -1).unsqueeze(-1).repeat(1, 1, P, E)
                for i, blk in enumerate(self.blocks):
                    tokens_mask = None
                    if self.ada_tokens:
                        tokens_mask, curr_task_mask = self._generate_tokens_mask(
                            x, task)
                        activated_tokens.append(tokens_mask)
                        task_masks.append(curr_task_mask)
                    else:
                        activated_tokens.append(
                            (torch.ones([1, x.shape[1], 1]).to(x.get_device())))

                    if adaptive:
                        if self.ada_blocks:
                            tmp = blk(x, tokens_mask)
                            mask_inv = torch.ones_like(
                                layer_mask[:, i, :, :]) - layer_mask[:, i, :, :]
                            x = (tmp * layer_mask[:, i, :, :]) + (x * mask_inv)
                        else:
                            tmp = blk(x, tokens_mask)
                    else:
                        x = blk(x, tokens_mask=None)
            else:
                assert x.shape[0] == 1, "Batch size has to be exactly 1 during inference"
                for i, blk in enumerate(self.blocks):
                    if self.ada_blocks and not layer_mask[0, i]:
                        activated_tokens.append(torch.zeros(
                            [1, x.shape[1], 1]).to(x.get_device()))
                        if self.policy_mode == 'per_task':
                            curr_task_mask = {}
                            for tsk in self.tasks:
                                curr_task_mask[tsk] = torch.zeros(
                                    [1, x.shape[1], 1]).to(x.get_device())
                            task_masks.append(curr_task_mask)
                        if adaptive:
                            continue

                    tokens_mask = None
                    if self.ada_tokens or self.ada_blocks:
                        if self.ada_tokens:
                            tokens_mask, curr_task_mask = self._generate_tokens_mask(
                                x, task)
                            task_masks.append(curr_task_mask)
                            activated_tokens.append(tokens_mask)
                        else:
                            activated_tokens.append(torch.ones(
                                [1, x.shape[1], 1]).to(x.get_device()))

                    if adaptive:
                        x = blk(x, tokens_mask)
                    else:
                        if self.use_checkpoint:
                            x = checkpoint.checkpoint(blk, x)
                        else:
                            x = blk(x)
        else:
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x, activated_tokens, task_masks

    def flops(self, x=None, layer_mask=None, logger=None):
        overhead_flops = 0
        flops_per_block_per_layer = []
        flops = 0
        if x is None:
            for i, blk in enumerate(self.blocks):
                temp = copy.deepcopy(flops)
                flops += blk.flops()
                if logger:
                    logger.debug(
                        f"\t\tStatic Block {i} flops = {(flops - temp)/1e6}")
        else:
            assert x.shape[
                0] == 1, f"Batch size has to be exactly 1 during inference, {x.shape}"
            for i, blk in enumerate(self.blocks):
                temp = flops

                if self.ada_blocks:
                    if layer_mask[0, i] == 0:
                        flops_per_block_per_layer.append(0)
                        continue

                tokens_mask = None
                if self.ada_tokens:
                    tokens_mask, _ = self._generate_tokens_mask(x, task=None)
                    if self.policy_mode == 'single':
                        overhead_flops += self.tokens_select.flops()
                    else:
                        for _, ctrl in self.tokens_select.items():
                            overhead_flops += ctrl.flops()

                    flops += blk.flops(tokens_mask, logger=logger)
                else:
                    flops += blk.flops()

                if logger:
                    logger.debug(
                        f"\t\tAdaptive Block {i} flops = {(flops - temp)/1e6}, {flops/1e6}")

                flops_per_block_per_layer.append((flops - temp)/1e6)
                x = blk(x, tokens_mask)

        flops += overhead_flops
        if logger:
            logger.debug(
                f"\tLayer - tokens controller overhead flops = {overhead_flops/1e6}")
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops, flops_per_block_per_layer


class AdaSwinTransformer(SwinTransformer):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, fused_window_process=False,
                 flatten_ft=False, policy_mode='single', controller_mode='per_model', tasks=[],
                 ada_blocks=False, ada_tokens=False, block_ctrl_size=100, token_ctrl_size=100,
                 task_attn=False, task_attn_num_heads=5, adaptive: bool = True, hard_gumbel: bool = True, random=False, **kwargs):

        # Note: The depth vector represents the number of stages and the depth of each stage
        #       For example: depths=[2, 2, 6, 2] means 4 stages where the first, second, third and
        #                    fourth stages have 2, 2, 6, 2, layers respectively
        self.policy_mode = policy_mode
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_size = patch_size
        self.depths = depths
        self.tasks = tasks
        self.token_ctrl_size = token_ctrl_size
        self.block_ctrl_size = block_ctrl_size
        self.adaptive = adaptive
        self.hard_gumbel = hard_gumbel
        self.ada_blocks = ada_blocks
        self.ada_tokens = ada_tokens
        self.random = random
        ayer_ctor = partial(AdaBasicLayer,
                            ada_tokens=self.ada_tokens,
                            ada_blocks=self.ada_blocks,
                            token_ctrl_size=self.token_ctrl_size,
                            tasks=tasks,
                            policy_mode=policy_mode,
                            window_size=window_size,
                            random=self.random,
                            hard_gumbel=self.hard_gumbel
                            )

        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes,
                         embed_dim=embed_dim, depths=depths, num_heads=num_heads,
                         window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
                         norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
                         use_checkpoint=use_checkpoint, fused_window_process=fused_window_process, basic_layer=ayer_ctor, **kwargs)

        if self.policy_mode == 'single':
            self.layers_controllers = self._generate_block_select_single_policy()
        elif self.policy_mode == 'per_task':
            assert len(tasks) > 0
            self.layers_controllers = self._generate_block_select_per_task_policy()

    def _generate_block_select_single_policy(self):
        if not self.ada_blocks:
            return None
        return nn.ModuleList([
            BlockSelect(
                num_embeddings=self.embed_dim*(2**idx),
                num_patches=int((self.img_size/(self.patch_size*(2**idx)))**2),
                num_blocks=self.depths[idx],
                hidden_sz=self.block_ctrl_size,
                random=self.random) for idx in range(len(self.depths))
        ])

    def _generate_block_select_per_task_policy(self):
        if not self.ada_blocks:
            return None
        return nn.ModuleDict({
            task: self._generate_block_select_single_policy()
            for task in self.tasks
        })

    def _generate_layer_mask(self, x, layer_idx, task):
        curr_task_mask = None
        if self.policy_mode == 'single':
            layer_mask = self.layers_controllers[layer_idx](
                x, hard=self.hard_gumbel)
        else:
            if task is not None:
                layer_mask = self.layers_controllers[task][layer_idx](
                    x, hard=self.hard_gumbel)
                curr_task_mask = {}
                for tsk in self.tasks:
                    if tsk == task:
                        curr_task_mask[tsk] = layer_mask
                    else:
                        curr_task_mask[tsk] = self.layers_controllers[tsk][layer_idx](
                            x)
            else:
                layer_mask = None
                curr_task_mask = {}
                for tsk, ctrl in self.layers_controllers.items():
                    if layer_mask is None:
                        layer_mask = ctrl[layer_idx](x, hard=self.hard_gumbel)
                        curr_task_mask[tsk] = layer_mask
                    else:
                        # task is task name
                        task_mask = ctrl[layer_idx](x, hard=self.hard_gumbel)
                        curr_task_mask[tsk] = task_mask
                        layer_mask = torch.clamp_max(
                            layer_mask + task_mask, 1.0)
        return layer_mask, curr_task_mask

    def forward_features(self, x, return_activation_stats=False, return_stages=False, flatten_ft=False, task=None):
        x = self.patch_embed(x)

        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        if return_stages:
            out = []

        activated_layers = []
        activated_layers_per_task = []
        activated_tokens = []
        activated_tokens_per_task = []

        # Compute the mask
        for idx, layer in enumerate(self.layers):
            if self.ada_blocks:
                layer_mask, layer_per_task = self._generate_layer_mask(
                    x, layer_idx=idx, task=task)
            else:
                layer_mask, layer_per_task = None, None
            x, activated_tokens_per_layer, activated_tokens_per_layer_per_task = layer(
                x, layer_mask, adaptive=self.adaptive, task=task)
            if self.ada_blocks:
                activated_layers.append(layer_mask)
                activated_layers_per_task.append(layer_per_task)
            else:
                activated_layers.append(torch.ones(
                    x.shape[0], self.depths[idx]).to(x.get_device()))
                activated_layers_per_task.append({tsk: torch.ones(
                    x.shape[0], self.depths[idx]).to(x.get_device()) for tsk in self.tasks})

            activated_tokens.append(activated_tokens_per_layer)
            activated_tokens_per_task.append(
                activated_tokens_per_layer_per_task)

            if return_stages:
                out.append(x)

        # x = self.norm(x)  # B L C
        if flatten_ft:
            x = self.avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
        if return_stages:
            x = out
        if return_activation_stats:
            return x, activated_layers, activated_tokens, activated_tokens_per_task, activated_layers_per_task
        return x

    def forward(self, x, return_activation_stats=False, return_stages=False, flatten_ft=False, task=None):
        if return_activation_stats:
            x, nBs, nTs, nTsPerTask, nBsPerTask = self.forward_features(x, return_activation_stats=return_activation_stats,
                                                                        return_stages=return_stages, flatten_ft=flatten_ft, task=task)
        else:
            x = self.forward_features(
                x, return_stages=return_stages, flatten_ft=flatten_ft, task=task)

        x = self.head(x)

        if return_activation_stats:
            return x, nBs, nTs, nTsPerTask, nBsPerTask
        else:
            return x

    def unfreeze_controllers(self, task=None):
        print("Unfreezing Controllers")
        if task is None:
            if self.policy_mode == 'per_task':
                for layers_controller in self.layers_controllers.values():
                    for ctrl in layers_controller:
                        ctrl.unfreeze_controllers()
            else:
                for layers_controller in self.layers_controllers:
                    layers_controller.unfreeze_controllers()

            for layer in self.layers:
                if self.policy_mode == 'per_task':
                    for ctrl in layer.tokens_select.values():
                        ctrl.unfreeze_controllers()
                else:
                    layer.tokens_select.unfreeze_controllers()
        else:
            for layers_controller in self.layers_controllers[task]:
                layers_controller.unfreeze_controllers()
            for layer in self.layers:
                layer.tokens_select[task].unfreeze_controllers()

    def freeze_controllers(self, task=None):
        print("Freezing Controllers")
        if task is None:
            if self.policy_mode == 'per_task':
                for layers_controller in self.layers_controllers.values():
                    for ctrl in layers_controller:
                        ctrl.freeze_controllers()
            else:
                for layers_controller in self.layers_controllers:
                    layers_controller.freeze_controllers()

            for layer in self.layers:
                if self.policy_mode == 'per_task':
                    for ctrl in layer.tokens_select.values():
                        ctrl.freeze_controllers()
                else:
                    layer.tokens_select.freeze_controllers()
        else:
            for layers_controller in self.layers_controllers[task]:
                layers_controller.freeze_controllers()
            layer.tokens_select[task].freeze_controllers()

    def flops(self, x=None, logger=None, detailed=False):
        flops = 0
        flops_per_block = []

        flops += self.patch_embed.flops()

        if x is not None:
            x = self.patch_embed(x)
            if self.ape:
                x = x + self.absolute_pos_embed
            x = self.pos_drop(x)

        for i, layer in enumerate(self.layers):
            temp = copy.deepcopy(flops)
            if x is not None:
                layer_mask, _ = self._generate_layer_mask(
                    x, layer_idx=i, task=None)
                layer_flops, layer_flops_per_block = layer.flops(
                    x=x, layer_mask=layer_mask, logger=logger)
                flops += layer_flops
                x, _, _ = layer(
                    x, layer_mask, adaptive=self.adaptive, task=None)
            else:
                layer_mask = None
                layer_flops, layer_flops_per_block = layer.flops(
                    x=x, layer_mask=layer_mask, logger=logger)
                flops += layer_flops

            if logger:
                logger.debug(f"\tLayer {i} flops = {(flops - temp)/1e6}")
            flops_per_block.append(layer_flops_per_block)

        flops += self.num_features * \
            self.patches_resolution[0] * \
            self.patches_resolution[1] // (2 ** self.num_layers)

        flops += self.num_features * self.num_classes

        if detailed:
            return flops, flops_per_block
        else:
            return flops
