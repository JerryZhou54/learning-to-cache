# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from einops import repeat
from models.router_models import STE


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################

def select_topk_and_remaining_tokens(x, token_weights, k, C):
    """
    Selects top-k and remaining tokens based on the token weights.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, C).
        token_weights (torch.Tensor): Weights tensor of shape (B, N).
        k (int): Number of top tokens to select.
        C (int): Number of channels.

    Returns:
        topk_x (torch.Tensor): Top-k tokens of shape (B, k, C).
        remaining_x (torch.Tensor): Remaining tokens of shape (B, N, C).
        topk_indices (torch.Tensor): Indices of top-k tokens of shape (B, k).
    """
    B, N, _ = x.shape
    topk_weights, topk_indices = torch.topk(torch.sigmoid(token_weights), k=k, sorted=False)
    sorted_indices, index = torch.sort(topk_indices, dim=1)

    # Get top-k tokens
    topk_x = x.gather(
        dim=1,
        index=repeat(sorted_indices, 'b t -> b t d', d=C)
    )

    # Get remaining tokens
    remaining_x = x.clone()
    remaining_x.scatter_(1, repeat(sorted_indices, 'b t -> b t d', d=C), torch.zeros_like(topk_x))

    return topk_weights, topk_x, remaining_x, sorted_indices, index

class STE_Ceil(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in):
        x = torch.ceil(x_in)
        return x
    
    @staticmethod
    def backward(ctx, g):
        return g, None

ste_ceil = STE_Ceil.apply

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, 
        routing=False, mod_ratio=0, diffrate=False, timewise=False,
        **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        ## MoD attributes
        self.routing = routing
        self.mod_ratio = mod_ratio
        self.diffrate = diffrate
        self.timewise = timewise
        if self.routing:
            self.mod_router = nn.Linear(hidden_size, 1, bias=False)
        if self.diffrate:
            self.kept_ratio_candidate = nn.Parameter(torch.arange(1, 0, -0.1).float())
            # self.kept_ratio_candidate = nn.Parameter(torch.arange(1, 0, -0.2).float())
            self.kept_ratio_candidate.requires_grad_(False)

            self.diff_mod_ratio = nn.Parameter(torch.tensor(1.0)) # modified
            self.diff_mod_ratio.requires_grad_(True)

    def find_soft_nearest_bins(self, kept_mod_ratio):
        # Calculate the absolute differences between diff_mod_ratio and each candidate value
        differences = torch.abs(self.kept_ratio_candidate - kept_mod_ratio)

        # Find the indices of the two smallest differences
        _, indices = torch.topk(differences, 2, largest=False)
        
        # Get the values corresponding to these indices
        nearest_bins = self.kept_ratio_candidate[indices]

        lower_bin, upper_bin = nearest_bins[0], nearest_bins[1]
        lower_weight = (upper_bin - kept_mod_ratio) / (upper_bin - lower_bin)
        upper_weight = 1.0 - lower_weight

        weights = torch.tensor([lower_weight.log(), upper_weight.log()])
        temperature = 1.0  
        soft_samples = F.gumbel_softmax(weights, tau=temperature, hard=True)
        selected_bin = torch.where(soft_samples[0] == 1.0, lower_bin, upper_bin)
        
        return selected_bin

    def find_nearest_bins(self, kept_mod_ratio):
        # Calculate the absolute differences between diff_mod_ratio and each candidate value
        differences = torch.abs(self.kept_ratio_candidate - kept_mod_ratio)

        # Find the indices of the two smallest differences
        _, indices = torch.topk(differences, 2, largest=False)

        # Get the values corresponding to these indices
        nearest_bins = self.kept_ratio_candidate[indices]

        return nearest_bins, indices

    def forward(self, x, c, reuse_attn=None, reuse_mlp=None, activate_mod_router=False):
        B, N, C = x.shape
        if self.routing and not self.diffrate and activate_mod_router:
            token_weights = self.mod_router(x).squeeze(2)

            capacity = int(self.mod_ratio * N)
            k = min(N, capacity)
            k = max(k, 1)
            topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)

            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
            topk_x = topk_x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(topk_x), shift_msa, scale_msa))
            out = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(topk_x), shift_mlp, scale_mlp))

            out = out * topk_weights.gather(dim=1, index=index).unsqueeze(2)
            out = out + topk_x
            
            # if self.training:
            #     if reuse_attn is not None:
            #         reuse_attn = reuse_attn.gather(dim=1, index=repeat(sorted_indices, 'b t -> b t d', d=C))
            #     if reuse_mlp is not None:
            #         reuse_mlp = reuse_mlp.gather(dim=1, index=repeat(sorted_indices, 'b t -> b t d', d=C))

            # if self.training:
            #     out, (attn_out, mlp_out) = self._forward(topk_x, c)
            # else:
            #     out, (attn_out, mlp_out) = self._forward(topk_x, c, reuse_att=reuse_attn, reuse_mlp=reuse_mlp)

            out = remaining_x.scatter_add(
                dim=1,
                index=repeat(sorted_indices, 'b t -> b t d', d=C),
                src=out
            )
            attn_out, mlp_out = None, None
        elif self.routing and self.diffrate and activate_mod_router:
            if self.training:
                kept_mod_ratio = torch.clamp(self.diff_mod_ratio, 0.1, 1.0)
                nearest_bins, indices = self.find_nearest_bins(kept_mod_ratio)

                lower_bin, upper_bin = nearest_bins[0], nearest_bins[1]
                lower_weight = (upper_bin - kept_mod_ratio) / (upper_bin - lower_bin)
                upper_weight = 1.0 - lower_weight

                # lower outputs
                capacity = ste_ceil(lower_bin * N).to(torch.int32)
                k = torch.min(capacity, torch.tensor(N, device=x.device))

                token_weights = self.mod_router(x).squeeze(2)
                topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)

                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
                topk_x = topk_x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(topk_x), shift_msa, scale_msa))
                lower_out = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(topk_x), shift_mlp, scale_mlp))

                lower_out = lower_out * topk_weights.gather(dim=1, index=index).unsqueeze(2)
                lower_out = lower_out + topk_x

                lower_out = remaining_x.scatter_add(
                    dim=1,
                    index=repeat(sorted_indices, 'b t -> b t d', d=C),
                    src=lower_out
                )

                # upper outputs

                capacity = ste_ceil(upper_bin * N).to(torch.int32)
                k = torch.min(capacity, torch.tensor(N, device=x.device))

                token_weights = self.mod_router(x).squeeze(2)
                topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)

                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
                topk_x = topk_x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(topk_x), shift_msa, scale_msa))
                upper_out = gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(topk_x), shift_mlp, scale_mlp))

                upper_out = upper_out * topk_weights.gather(dim=1, index=index).unsqueeze(2)
                upper_out = upper_out + topk_x

                upper_out = remaining_x.scatter_add(
                    dim=1,
                    index=repeat(sorted_indices, 'b t -> b t d', d=C),
                    src=upper_out
                )

                # Linear combination of the two outputs
                out = lower_weight * lower_out + upper_weight * upper_out
                return out, (None, None), kept_mod_ratio
            else:
                kept_mod_ratio = torch.clamp(self.diff_mod_ratio, 0.1, 1.0)
                if kept_mod_ratio < 1:
                    kept_mod_ratio = self.find_soft_nearest_bins(kept_mod_ratio)

                    capacity = ste_ceil(kept_mod_ratio*N).to(torch.int32) #kept_mod_ratio *
                    k = torch.min(capacity, torch.tensor(N, device=x.device))
                    token_weights = self.mod_router(x).squeeze(2)

                    topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)
                else:
                    out, (attn_out, mlp_out) = self._forward(x, c, reuse_att=reuse_attn, reuse_mlp=reuse_mlp)
                    return out, (attn_out, mlp_out), kept_mod_ratio

                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
                if reuse_attn is not None:
                    att_out = reuse_attn
                else:
                    att_out = self.attn(modulate(self.norm1(topk_x), shift_msa, scale_msa))
                topk_x = topk_x + gate_msa.unsqueeze(1) * att_out

                if reuse_mlp is not None:
                    mlp_out = reuse_mlp
                else:
                    mlp_out = self.mlp(modulate(self.norm2(topk_x), shift_mlp, scale_mlp))
                out = gate_mlp.unsqueeze(1) * mlp_out

                out = out * topk_weights.gather(dim=1, index=index).unsqueeze(2)
                out = out + topk_x

                out = remaining_x.scatter_add(
                    dim=1,
                    index=repeat(sorted_indices, 'b t -> b t d', d=C),
                    src=out
                )

                return out, (att_out, mlp_out), kept_mod_ratio
        else:
            out, (attn_out, mlp_out) = self._forward(x, c, reuse_att=reuse_attn, reuse_mlp=reuse_mlp)

        return out, (attn_out, mlp_out)


    def _forward(self, x, c, reuse_att=None, reuse_mlp=None):
        B, N, C = x.shape
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        if reuse_att is None:
            att_out = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        else:
            att_out = reuse_att
        x = x + gate_msa.unsqueeze(1) * att_out

        if reuse_mlp is None:
            mlp_out = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        else:
            mlp_out = reuse_mlp
        x = x + gate_mlp.unsqueeze(1) * mlp_out
        return x, (att_out, mlp_out)


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.depth=depth

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.diffrate = True
        self.target_ratio = 0.8
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, routing=True, mod_ratio=self.target_ratio, diffrate=self.diffrate) for _ in range(depth)
        ])
        if self.diffrate:
            self.kept_mod_ratios = [1.0] * depth

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        self.reset()
        
    
    def reset(self, start_timestep=20):
        self.cur_timestep = start_timestep-1
        self.reuse_feature = [None] * self.depth

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    

    def load_ranking(self, path, num_steps, timestep_map, thres):
        self.rank = [None] * num_steps
        from models.router_models import Router

        act_layer, total_layer = 0, 0
        ckpt = torch.load(path, map_location='cpu')['routers']
        self.routers = torch.nn.ModuleList([
            Router(2*self.depth) for _ in range(num_steps)
        ])
        self.routers.load_state_dict(ckpt)
        self.timestep_map =  {timestep: i for i, timestep in enumerate(timestep_map)}

        # act_att, act_mlp = 0, 0
        # for idx, router in enumerate(self.routers):
        #     if idx % 2 == 0:
        #         self.rank[idx] = STE.apply(router(), thres).nonzero().squeeze(0)
        #         #print(router(), STE.apply(router(), thres).nonzero())
        #         total_layer += 2 * self.depth
        #         act_layer += len(self.rank[idx])
        #         print(f"TImestep {idx}: Not Reuse: {self.rank[idx].squeeze()}")

        #         if len(self.rank[idx]) > 0:
        #             act_att += sum(1 - torch.remainder(self.rank[idx], 2)).item()
        #             act_mlp += sum(torch.remainder(self.rank[idx], 2)).item()
                    
        # print(f"Total Activate Layer: {act_layer}/{total_layer}")
        # print(f"Total Activate Attention: {act_att}/{total_layer//2}")
        # print(f"Total Activate MLP: {act_mlp}/{total_layer//2}")

            
    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, y, activate_mod_router=False, activate_router=False, label_for_dropout=None, fix_reuse_feature=False, thres=0.1):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        timestep = t[0].item()
        if activate_router:
            router_idx = self.timestep_map[timestep]

        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training, force_drop_ids=label_for_dropout)    # (N, D)
        c = t + y                                # (N, D)

        if not self.training:
            if self.cur_timestep % 2 == 1:
                self.reuse_feature = [None] * len(self.reuse_feature)

        if activate_router:
            router_idx = self.timestep_map[timestep]
            scores = self.routers[router_idx]()
            router_l1_loss = scores.sum()
            
        for i, block in enumerate(self.blocks):
            att, mlp = None, None
            
            if activate_router and self.reuse_feature[i] is not None and 2*i not in scores:
                att = self.reuse_feature[i][0]
                # print("Reuse Attention")

            if activate_router and self.reuse_feature[i] is not None and 2*i+1 not in scores:
                mlp = self.reuse_feature[i][1]
                # print("Reuse MLP")

            if (activate_router and not activate_mod_router) or (activate_mod_router and not self.diffrate):
                x, reuse_feature = block(x, c, reuse_attn=att, reuse_mlp=mlp, activate_mod_router=activate_mod_router) # (N, T, D)
            else:
                x, reuse_feature, diff_mod_ratio = block(x, c, reuse_attn=att, reuse_mlp=mlp, activate_mod_router=activate_mod_router)
                self.kept_mod_ratios[i] = diff_mod_ratio

            if not fix_reuse_feature:
                self.reuse_feature[i] = reuse_feature
        
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                    # (N, out_channels, H, W)

        self.cur_timestep -= 1
        if activate_mod_router and self.diffrate:
            mod_loss = self.calculate_mod_loss(self.kept_mod_ratios, self.target_ratio)
        else:
            mod_loss = torch.tensor(0, dtype=torch.float32).cuda()

        return x, mod_loss

    def calculate_mod_loss(self, ratios, target_ratio):
        avg_mod_ratio = torch.mean(torch.stack(ratios))
        target_ratio_tensor = torch.tensor([target_ratio], device=avg_mod_ratio.device)
        loss = F.mse_loss(avg_mod_ratio, target_ratio_tensor)

        return loss

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        # model_out, l2_loss = self.forward(combined, t, y, activate_mod_router=False, fix_reuse_feature=False, activate_router=True) # l2c
        # model_out, l2_loss = self.forward(combined, t, y, activate_mod_router=True, fix_reuse_feature=False, activate_router=True) # l2c + mod
        model_out, l2_loss = self.forward(combined, t, y, activate_mod_router=True, fix_reuse_feature=True, activate_router=False) # mod
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
}
