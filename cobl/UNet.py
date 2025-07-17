import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from functools import partial

from .LDM.openaimodel import *
from .LDM.attention import *

import torch as th
from einops import rearrange, repeat

import torch.nn as nn

import xformers
import xformers.ops


def convert_module_to_f16(module):
    """
    Converts module parameters and buffers to float16, excluding GroupNorm32, nn.GroupNorm, and nn.LayerNorm layers.

    Args:
        module (nn.Module): The module to convert.
    """
    # Convert parameters to float16
    for name, param in module.named_parameters(recurse=False):
        param.data = param.data.half()
        if param.grad is not None:
            param.grad.data = param.grad.data.half()

    # Convert buffers to float16
    for name, buffer in module.named_buffers(recurse=False):
        buffer.data = buffer.data.half()


class TempPositionalEncoding(nn.Module):
    def __init__(self, dim, max_pos=512):
        super().__init__()

        pos = torch.arange(max_pos)

        freq = torch.arange(dim // 2) / dim
        freq = (freq * torch.tensor(10000).log()).exp()

        x = rearrange(pos, "L -> L 1") / freq
        x = rearrange(x, "L d -> L d 1")

        pe = torch.cat((x.sin(), x.cos()), dim=-1)
        self.pe = rearrange(pe, "L d sc -> L (d sc)")

        self.dummy = nn.Parameter(torch.rand(1))

    def forward(self, length):
        enc = self.pe[:length]
        enc = enc.to(self.dummy.device)
        return enc


class TemporalAttentionLayer(nn.Module):
    # Rewritten to entirely be self attention (k = q = v)
    def __init__(self, dim, n_frames, n_heads=8, kv_dim=None):
        super().__init__()
        self.n_frames = n_frames
        self.n_heads = n_heads

        self.pos_enc = TempPositionalEncoding(dim)

        head_dim = dim // n_heads
        proj_dim = head_dim * n_heads
        self.q_proj = nn.Linear(dim, proj_dim, bias=False)

        kv_dim = dim
        self.k_proj = nn.Linear(kv_dim, proj_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, proj_dim, bias=False)
        self.o_proj = nn.Linear(proj_dim, dim, bias=False)

        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, q, mask=None):
        skip = q

        bt, c, h, w = q.shape
        b = int(bt / self.n_frames)
        q = rearrange(q, "(b t) c h w -> (b h w) t c", t=self.n_frames)

        q = q + self.pos_enc(self.n_frames).to(q.dtype)
        kv = q

        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # q = rearrange(q, "bhw t (heads d) -> bhw heads t d", heads=self.n_heads)
        # k = rearrange(k, "bhw s (heads d) -> bhw heads s d", heads=self.n_heads)
        # v = rearrange(v, "bhw s (heads d) -> bhw heads s d", heads=self.n_heads)
        # out = F.scaled_dot_product_attention(q, k, v, mask)
        # out = rearrange(out, "bhw heads t d -> bhw t (heads d)")

        # Apply xFormers memory-efficient attention
        q = rearrange(q, "bhw t (heads d) -> bhw t heads d", heads=self.n_heads)
        k = rearrange(k, "bhw s (heads d) -> bhw s heads d", heads=self.n_heads)
        v = rearrange(v, "bhw s (heads d) -> bhw s heads d", heads=self.n_heads)
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask)
        out = rearrange(out, "bhw t heads d -> bhw t (heads d)")

        out = self.o_proj(out)
        out = rearrange(out, "(b h w) t c -> (b t) c h w", b=b, h=h, w=w)

        with torch.no_grad():
            self.alpha.clamp_(0, 1)

        out = self.alpha * skip + (1 - self.alpha) * out
        return out


class SpatialTemporalTransformer(SpatialTransformer):
    def __init__(
        self,
        in_channels,
        n_frames,
        temporal_heads,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.use_checkpoint = kwargs["use_checkpoint"]
        self.temporalAttention = TemporalAttentionLayer(
            in_channels, n_frames, temporal_heads
        )

    def forward(self, x, context=None):
        return checkpoint(
            self._forward, (x, context), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, context=None):
        x = super().forward(x, context)
        return self.temporalAttention(x)


class Conv3DLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_frames):
        super().__init__()
        self.n_frames = n_frames
        k, p = (3, 1, 1), (1, 0, 0)
        # k, p = (7, 1, 1), (3, 0, 0)
        self.block1 = nn.Sequential(
            nn.GroupNorm(32, in_dim),
            nn.SiLU(),
            nn.Conv3d(in_dim, out_dim, kernel_size=k, stride=1, padding=p),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(32, out_dim),
            nn.SiLU(),
            nn.Conv3d(out_dim, out_dim, kernel_size=k, stride=1, padding=p),
        )
        self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, x):
        # Input is [(B N) D H W]
        h = rearrange(x, "(b t) c h w -> b c t h w", t=self.n_frames)
        h = self.block1(h)
        h = self.block2(h)
        h = rearrange(h, "b c t h w -> (b t) c h w")

        with torch.no_grad():
            self.alpha.clamp_(0, 1)

        return self.alpha * x + (1 - self.alpha) * h


class ResBlockWithT2I(ResBlock):
    def __init__(self, useT2I, use_conv3d, n_frames, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if useT2I:
            self.alpha = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        else:
            self.alpha = None

        if use_conv3d:
            out_ch = (
                kwargs["out_channels"] if "out_channels" in kwargs.keys() else args[0]
            )
            self.conv3d = Conv3DLayer(out_ch, out_ch, n_frames)
        else:
            self.conv3d = None

    def forward(self, x, emb, cond):
        x = super()._forward(x, emb)

        # T2I adaption
        if cond is not None and self.alpha is not None:
            with torch.no_grad():
                self.alpha.clamp_(0, 1)
            x = x * self.alpha + (1 - self.alpha) * cond

        # Conv3D block
        if self.conv3d is not None:
            x = self.conv3d(x)

        return x


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, context=None, cond=None):
        for layer in self:
            if isinstance(layer, ResBlockWithT2I):
                x = layer(x, emb, cond)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context)
            else:
                x = layer(x)
        return x


class UNetModel(nn.Module):
    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_spatial_transformer=False,  # custom transformer support
        transformer_depth=1,  # custom transformer support
        context_dim=None,  # custom transformer support
        num_attention_blocks=None,
        use_linear_in_transformer=False,
        n_frames=7,
        n_temp_heads=8,
        use_conv3d=False,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert (
                context_dim is not None
            ), "Fool!! You forgot to include the dimension of your cross-attention conditioning..."

        if context_dim is not None:
            assert (
                use_spatial_transformer
            ), "Fool!! You forgot to use the spatial transformer for your cross-attention conditioning..."
            from omegaconf.listconfig import ListConfig

            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert (
                num_head_channels != -1
            ), "Either num_heads or num_head_channels has to be set"

        if num_head_channels == -1:
            assert (
                num_heads != -1
            ), "Either num_heads or num_head_channels has to be set"

        self.n_frames = n_frames
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError(
                    "provide num_res_blocks either as an int (globally constant) or "
                    "as a list/tuple (per-level) with the same length as channel_mult"
                )
            self.num_res_blocks = num_res_blocks
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(
                map(
                    lambda i: self.num_res_blocks[i] >= num_attention_blocks[i],
                    range(len(num_attention_blocks)),
                )
            )
            print(
                f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                f"attention will still not be set."
            )

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        self.fcond_tracker = [0]
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlockWithT2I(
                        True,
                        use_conv3d,
                        n_frames,
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    ),
                ]

                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if (
                        not exists(num_attention_blocks)
                        or nr < num_attention_blocks[level]
                    ):
                        layers.append(
                            SpatialTemporalTransformer(
                                ch,
                                n_frames,
                                n_temp_heads,
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=False,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
                self.fcond_tracker.append(level)

            if level != len(channel_mult) - 1:
                self.fcond_tracker.append(level + 1)  # this value holds place
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlockWithT2I(
                False,
                use_conv3d,
                n_frames,
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            SpatialTemporalTransformer(
                ch,
                n_frames,
                n_temp_heads,
                ch,
                num_heads,
                dim_head,
                depth=transformer_depth,
                context_dim=context_dim,
                disable_self_attn=False,
                use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint,
            ),
            ResBlockWithT2I(
                False,
                use_conv3d,
                n_frames,
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlockWithT2I(
                        False,
                        use_conv3d,
                        n_frames,
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels

                    if (
                        not exists(num_attention_blocks)
                        or i < num_attention_blocks[level]
                    ):
                        layers.append(
                            SpatialTemporalTransformer(
                                ch,
                                n_frames,
                                n_temp_heads,
                                ch,
                                num_heads,
                                dim_head,
                                depth=transformer_depth,
                                context_dim=context_dim,
                                disable_self_attn=False,
                                use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint,
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )

        self.T2I_Weights = nn.Parameter(
            torch.tensor([1.0] * len(channel_mult)), requires_grad=True
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        # self.apply(convert_module_to_f16)
        # self.apply(convert_module_to_f16)
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)
        self.out.apply(convert_module_to_f16)
        self.time_embed.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, context=None, cond=None, **kwargs):
        # x passed in as [B, N*D, H, W]
        # Reshape x to [(B N) D H W]

        # Cond needs to be reshaped so all parallel diffusion models receive it
        if cond is not None:
            cond_resh = []
            # If using depth, then compute fusion of conditions
            if len(cond) == 2:
                for i, (fc1, fc2) in enumerate(zip(cond[0], cond[1])):
                    fweight = self.T2I_Weights[i].to(self.dtype)
                    cond_resh.append(
                        repeat(
                            fc1 + fweight * fc2,
                            "B C H W -> (B N) C H W",
                            N=self.n_frames,
                        )
                    )
            else:
                for i, fc in enumerate(cond[0]):
                    fweight = self.T2I_Weights[i].to(self.dtype)
                    cond_resh.append(
                        repeat(
                            fc * fweight,
                            "B C H W -> (B N) C H W",
                            N=self.n_frames,
                        )
                    )
        else:
            # None feature vector for each UNet feature depth
            cond_resh = [None] * 4

        if context is not None:
            context = repeat(context, "B S E -> (B N) S E", N=self.n_frames)

        x = rearrange(
            x, "B (N D) H W -> (B N) D H W", D=self.in_channels, N=self.n_frames
        )
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.to(self.dtype)
        emb = self.time_embed(t_emb)
        emb = emb.to(self.dtype)
        emb = repeat(emb, "B E -> (B N) E", N=self.n_frames)
        h = x.type(self.dtype)
        hs = []

        for module, fiter in zip(self.input_blocks, self.fcond_tracker):
            h = module(h, emb, context, cond_resh[fiter])
            hs.append(h)

        h = self.middle_block(h, emb, context, None)

        for module in self.output_blocks:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb, context, None)
        h = h.type(x.dtype)

        h = self.out(h)
        return rearrange(
            h, "(B N) D H W -> B (N D) H W", D=self.in_channels, N=self.n_frames
        )


if __name__ == "__main__":
    # ### Test loading checkpoint
    # ckpt_path = (
    #     "/home/deanhazineh/ssd4tb_mounted/LDM/scripts/SD2p1/v2-1_512-ema-pruned.ckpt"
    # )
    # sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"]
    # for key, value in sd.items():
    #     print(key)

    # prefix = "model.diffusion_model."
    # stripped_state_dict = {
    #     key[len(prefix) :]: value for key, value in sd.items() if key.startswith(prefix)
    # }

    # model = UNetModel(
    #     use_checkpoint=True,
    #     use_fp16=False,
    #     in_channels=4,
    #     out_channels=4,
    #     model_channels=320,
    #     attention_resolutions=[4, 2, 1],
    #     num_res_blocks=2,
    #     channel_mult=[1, 2, 4, 4],
    #     num_head_channels=64,  # need to fix for flash-attn
    #     use_spatial_transformer=True,
    #     use_linear_in_transformer=True,
    #     transformer_depth=1,
    #     context_dim=1024,
    #     use_conv3d=True,
    # ).to("cuda")

    # missing, unexpected = model.load_state_dict(stripped_state_dict, strict=False)
    # print(f"Restored with {len(missing)} missing and {len(unexpected)} unexpected keys")
    # print(missing)
    # print(unexpected)

    ### Test model forward call
    from diffusers import T2IAdapter

    model = UNetModel(
        use_checkpoint=True,
        use_fp16=True,
        in_channels=4,
        out_channels=4,
        model_channels=320,
        attention_resolutions=[4, 2, 1],
        num_res_blocks=2,
        channel_mult=[1, 2, 4, 4],
        num_head_channels=64,  # need to fix for flash-attn
        use_spatial_transformer=True,
        use_linear_in_transformer=True,
        transformer_depth=1,
        context_dim=1024,
        use_conv3d=True,
    ).to("cuda")

    adapter = T2IAdapter(
        in_channels=3,
    ).to("cuda")
    adapter_depth = T2IAdapter(
        in_channels=1,
    ).to("cuda")

    # # B C H W
    scene = torch.zeros((1, 3, 512, 512), dtype=torch.float32, device="cuda")
    depth = torch.zeros((1, 1, 512, 512), dtype=torch.float32, device="cuda")
    scene_latent = torch.zeros((1, 4 * 7, 64, 64), device="cuda", dtype=torch.float32)

    scene_cond = adapter(scene)
    depth_cond = adapter_depth(depth)
    ccond = [scene_cond, depth_cond]
    timesteps = torch.tensor([1000], dtype=torch.float16, device="cuda")
    text_emb = torch.zeros([1, 77, 1024], dtype=torch.float16, device="cuda")

    ## convert everything to float16 then call
    cc = [
        [
            tensor.clone().detach().to(dtype=torch.float16).requires_grad_(True)
            for tensor in sublist
        ]
        for sublist in ccond
    ]
    xin = scene_latent.to(torch.float16)
    tc = text_emb.to(torch.float16)
    model.convert_to_fp16()
    model.apply(convert_module_to_f16)
    out = model(xin, timesteps, tc, cc)

    print(out.shape)

    # mask_model = MaskModel(320, 7 * 4, 7, 1024, chdim=32).to("cuda")
    # pred_mask = mask_model(out, timesteps)
    # print(pred_mask.shape)
