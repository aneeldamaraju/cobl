import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm
from .ddpm_utils import extract
from .ddpm import LatentDiffusion


class DDIM_Sampler(nn.Module):
    def __init__(
        self, diffusion_model, n_steps, ddim_scheme="uniform", ddim_eta=0, cfg_scale=1.0
    ):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.prediction_type = diffusion_model.prediction_type
        timesteps = self.diffusion_model.timesteps
        self.cfg_scale = cfg_scale

        if ddim_scheme == "uniform":
            c = timesteps // n_steps
            ddim_times = np.asarray(list(range(0, timesteps, c)))
            # # The original DDIM paper does not enforce starting at terminal T
            # # but later works have shown that this is flawed so I modify it
            # c = timesteps // n_steps
            # ddim_times = list(range(0, timesteps, c))
            # if ddim_times[-1] != 999:
            #     ddim_times.append(999)
            # ddim_times = np.array(ddim_times)
        else:
            raise ValueError("Unknown ddim scheme input")

        self._initialize_schedule(
            self.diffusion_model.alphas_cumprod, ddim_times, ddim_eta
        )
        self.n_steps = len(ddim_times)
        self.ddim_times = ddim_times
        self.ldm = diffusion_model.is_ldm
        self.uc_text_embd = self.diffusion_model.text_stage_model([""])

    def _initialize_schedule(self, orig_alphas_cumprod, ddim_times, ddim_eta):
        alphas_cumprod = orig_alphas_cumprod[ddim_times].clone()
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        ddim_sigma = (
            ddim_eta
            * torch.sqrt((1 - alphas_cumprod_prev) / (1 - alphas_cumprod))
            * torch.sqrt(1 - alphas_cumprod / alphas_cumprod_prev)
        )

        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.ddim_sigma = ddim_sigma
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.ddim_coeff = torch.sqrt(1 - alphas_cumprod_prev - ddim_sigma**2)
        return

    def p_sample(self, x_t, t, ti, xcond_embd, ccond_embd, clip):
        ### Collect coefficients
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, ti, x_t.shape
        )
        sqrt_recip_alphas_cumprod_t = extract(
            self.sqrt_recip_alphas_cumprod, ti, x_t.shape
        )
        sqrt_alpha_cumprod_prev_t = torch.sqrt(
            extract(self.alphas_cumprod_prev, ti, x_t.shape)
        )
        ddim_coeff_t = extract(self.ddim_coeff, ti, x_t.shape)
        ddim_sigma_t = extract(self.ddim_sigma, ti, x_t.shape)

        ### Get noise/score while allowing for CFG
        uc_text_embd = repeat(
            self.uc_text_embd, "b s c -> (b g) s c", g=xcond_embd.shape[0]
        )
        eps = []
        with torch.no_grad():
            for use_xcond in [xcond_embd, uc_text_embd]:
                model_output = self.diffusion_model.model(x_t, t, use_xcond, ccond_embd)
                if self.prediction_type == "epsilon":
                    epsilon = model_output
                elif self.prediction_type == "start_x":
                    sqrt_alphas_cumprod_t = torch.sqrt(
                        extract(self.alphascumprod, ti, x_t.shape)
                    )
                    epsilon = (
                        x_t - model_output * sqrt_alphas_cumprod_t
                    ) / sqrt_one_minus_alphas_cumprod_t
                else:
                    raise ValueError("Unsupported model prediction type.")
                eps.append(epsilon)
        epsilon = eps[1] + self.cfg_scale * (eps[0] - eps[1])

        ### Get x0hat
        pred_xstart = (
            x_t - sqrt_one_minus_alphas_cumprod_t * epsilon
        ) * sqrt_recip_alphas_cumprod_t
        if clip:
            pred_xstart = torch.clamp(pred_xstart, -1.0, 1.0)

        ### Compute DDIM Step
        nonzero_mask = (1 - (t == 0).float()).reshape(
            x_t.shape[0], *((1,) * (len(x_t.shape) - 1))
        )
        noise = torch.randn_like(x_t)

        xtm1 = (
            sqrt_alpha_cumprod_prev_t * pred_xstart
            + ddim_coeff_t * epsilon
            + nonzero_mask * ddim_sigma_t * noise
        )

        return xtm1, pred_xstart

    def p_sample_loop(
        self,
        im,
        xcond_embd=None,
        ccond_embd=None,
        return_intermediate=False,
        clip=False,
    ):
        batch_size = im.shape[0]
        img_shape = im.shape[1:]  # Get the shape of each image tensor
        total_steps = self.n_steps if return_intermediate else 1
        imgs = torch.zeros((total_steps, batch_size, *img_shape), device="cpu")
        x0s = torch.zeros((total_steps, batch_size, *img_shape), device="cpu")

        progress_bar = tqdm(
            zip(reversed(range(self.n_steps)), reversed(self.ddim_times)),
            total=self.n_steps,
            desc="",
        )
        step_idx = 0
        for idx, i in progress_bar:
            im, x0 = self.p_sample(
                im,
                torch.full((im.shape[0],), i, device=im.device, dtype=torch.long),
                torch.full((im.shape[0],), idx, dtype=torch.long, device="cuda"),
                xcond_embd,
                ccond_embd,
                clip=clip,
            )

            if (i == 0) or return_intermediate:
                im_ = im.clone().detach()
                x0_ = x0.clone().detach()

                ## Apply LSQ patch scaling
                imgs[step_idx] = im_.cpu()
                x0s[step_idx] = x0_.cpu()
                step_idx += 1
            progress_bar.set_description(f"Index {idx}, Time: {i}")

        return imgs, x0s

    @torch.no_grad()
    def sample(
        self,
        image_size=[64, 64],
        batch_size=16,
        return_intermediate=False,
        x_start=None,
        text_cond=None,
        cond=None,
        clip=False,  # Generally we don't use this for LDM
        depth=None,
    ):
        dm = self.diffusion_model
        device = next(dm.parameters()).device
        dtype = dm.use_dtype

        if self.ldm and cond is not None and dm.embed_cond:
            cond = dm.embd_in_latent(cond.to(dtype=dtype, device=device))

        if x_start is None:
            x_start = torch.randn(
                (batch_size, dm._seed_channels, *image_size),
                dtype=dtype,
                device=device,
            )
        else:
            x_start = x_start.to(dtype=dtype, device=device)

        if dm._use_cond:
            assert cond is not None, "Model uses cond but none is passed in"
            cond = cond.to(dtype=dtype, device=device)
            cond = dm.cond_stage_model(cond)
        else:
            cond = None

        if dm.use_depth:
            depth = depth.to(dtype=dtype, device=device)
            depth = dm.depth_stage_model(depth)
            assert len(cond) == len(depth), "Unexpected exception."

            # Change this code to add them in the Unet forward with a learnable weight
            cond = [cond, depth]
            # for i in range(len(cond)):
            #     cond[i] = cond[i] + depth[i]
            print("ADDED DEPTH TO COND")
        else:
            cond = [cond]

        if dm._use_text:
            assert (
                text_cond is not None
            ), "Model uses text conditioning but non is given"
            text_cond = dm.text_stage_model(text_cond).to(dtype=dtype, device=device)
        else:
            text_cond = None

        out, _ = self.p_sample_loop(
            x_start,
            text_cond,
            cond,
            return_intermediate,
            clip,
        )

        if not self.ldm:
            return out

        if return_intermediate:
            b = out.shape[0]
            for t in tqdm(
                range(b), desc="Decoding time sequences to pixel space", total=b
            ):
                out[t] = dm.decode(out[t].to("cuda")).cpu()
        else:
            print("Decoding the final timestep")
            out = dm.decode(out[-1].to("cuda"))[None].cpu()

        return out
