from torch.utils.checkpoint import checkpoint
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm
from copy import deepcopy
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt

from cobl.ddpm_utils import extract
from cobl.midas import MidasDepthEstimator
from cobl.datasets import reverse_transform, CenterSquareCrop, permute_dimensions
from cobl.load_utils import to_cobl_path
from cobl.U2Net.u2net_utils import load_mask_net, u2net_masks


def split_layers(x0):
    return rearrange(x0, "b v h w (n c) -> b v n h w c", c=3)  # [B, V, 7, 3, 512, 512]


def composite_imgs(imgs, masks):
    composite_out = imgs[-1:]
    for img, mask in reversed(list(zip(imgs[:-1], masks))):
        composite_out = img * mask + composite_out * (1 - mask)
    return composite_out


def calc_composite_loss(
    pred_layers, mask_pred, cond, shadow_invariant=False, use_ordinal_scale=False
):
    composite = composite_imgs(pred_layers, mask_pred[:-1])
    composite_target = cond
    if not shadow_invariant:
        composite_loss = torch.mean((composite[0] - composite_target) ** 2)
    else:
        composite_inv = composite / (torch.sum(composite, dim=-3, keepdim=True) + 1e-4)
        target_inv = composite_target / (
            torch.sum(composite_target, dim=-3, keepdim=True) + 1e-4
        )
        composite_loss = torch.mean((composite_inv[0, :2] - target_inv[:2]) ** 2)
    return composite, composite_loss


def permute(x):
    return x.transpose(-3, -1).transpose(-3, -2)


class loss_logger:
    def __init__(self):
        self.composite_loss = []
        self.sds_loss = []
        self.total_loss = []


class DDIM_Sampler(nn.Module):
    def __init__(
        self, diffusion_model, n_steps, ddim_scheme="uniform", ddim_eta=0, cfg_scale=1.0
    ):
        super().__init__()
        self.diffusion_model = diffusion_model.to("cuda")
        timesteps = self.diffusion_model.timesteps
        self.ddim_scheme = ddim_scheme
        self.ddim_eta = ddim_eta

        if ddim_scheme == "uniform":
            c = timesteps // n_steps
            ddim_times = np.asarray(list(range(0, timesteps, c)))
            # # The original DDIM paper does not enforce starting at terminal T
            # # but later works have shown that this is flawed. Possibly better fix:
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
        self.use_guidance = False
        self.cfg_scale = torch.tensor(cfg_scale, device="cuda")
        self.sds_weighting = 0.0
        self.guidance_weight = 0.0
        self.scene = None
        self.transform = T.Compose(
            [
                T.ToTensor(),
                CenterSquareCrop(),
                T.Resize((512, 512)),
            ]
        )
        self.depth_model = MidasDepthEstimator().to("cuda")
        self.uc_ccond = None
        self.logger = None
        # Pull out the latent space decoder which may be used in latent-to-pixel space guidance
        self.decoder = self.diffusion_model.first_stage_model
        self.scale = self.diffusion_model.scale_factor

    def guidance_call(self, x_t, t, ti, xcond_embd, ccond_embd):
        return x_t, None

    def p_sample(self, x_t, t, ti, xcond_embd, ccond_embd):
        ### Get Noise while using CFG
        with torch.no_grad():
            eps = self.diffusion_model.model(x_t, t, xcond_embd, ccond_embd)
            eps_uc = self.diffusion_model.model(x_t, t, xcond_embd, self.uc_ccond)
        epsilon = eps_uc + self.cfg_scale * (eps - eps_uc)

        ### Run Guidance
        if self.use_guidance:
            x_t, permuted_idxs = self.guidance_call(x_t, t, ti, xcond_embd, ccond_embd)
            x_t = self._permute_idxs(x_t, permuted_idxs)
            epsilon = self._permute_idxs(epsilon, permuted_idxs)

        ### Get x0hat and Compute DDIM Step
        pred_xstart = self._get_x0(x_t, epsilon, t, ti)
        xtm1 = self._get_xtm1(pred_xstart, epsilon, t, ti)
        return xtm1, pred_xstart

    def p_sample_loop(
        self,
        im,
        xcond_embd=None,
        ccond_embd=None,
        return_intermediate=False,
    ):
        batch_size = im.shape[0]
        img_shape = im.shape[1:]  # Get the shape of each image tensor
        total_steps = self.n_steps if return_intermediate else 1
        imgs = torch.zeros(
            (total_steps, batch_size, *img_shape), device="cpu", dtype=im.dtype
        )
        x0s = torch.zeros(
            (total_steps, batch_size, *img_shape), device="cpu", dtype=im.dtype
        )

        progress_bar = tqdm(
            zip(reversed(range(self.n_steps)), reversed(self.ddim_times)),
            total=self.n_steps,
            desc="",
        )
        step_idx = 0
        for idx, i in progress_bar:
            im, x0 = self.p_sample(
                im,
                torch.full((im.shape[0],), i, device=im.device, dtype=torch.float16),
                torch.full((im.shape[0],), idx, device=im.device, dtype=torch.long),
                xcond_embd,
                ccond_embd,
            )

            progress_bar.set_description(f"Index {idx}, Time: {i}")
            if (i == 0) or return_intermediate:
                imgs[step_idx] = im.clone().detach().cpu()
                x0s[step_idx] = x0.clone().detach().cpu()
                step_idx += 1

        return imgs, x0s

    def sample(
        self,
        image_size=[64, 64],
        batch_size=1,
        return_intermediate=False,
        x_start=None,
        text_cond=[""],
        cond=None,
        use_guidance=False,
        guidance_weight=None,
        sds_weighting=None,
        cfg_scale=None,
    ):

        # Update the sampling parameters
        self.use_guidance = use_guidance
        if guidance_weight is not None:
            self.guidance_weight = guidance_weight
        if sds_weighting is not None:
            self.sds_weighting = sds_weighting
        if cfg_scale is not None:
            self.cfg_scale = cfg_scale

        # Switch the model to float16 and housekeeping
        dtype = torch.float16
        decode_dtype = (
            torch.float32
        )  # I had issues with gradients of decoder otherwise...
        self.dtype = dtype
        self.decode_dtype = decode_dtype
        self.diffusion_model.to(dtype)
        self.diffusion_model.model.dtype = dtype
        if dtype == torch.float16:
            self.diffusion_model.model.convert_to_fp16()
        dm = self.diffusion_model
        device = next(dm.parameters()).device
        self.decoder.to(self.decode_dtype)

        scene, cond, depth = self.condition_transform(cond)
        self.scene = scene.to(dtype=dtype, device=device)

        if x_start is None:
            x_start = torch.randn(
                (batch_size, dm._seed_channels, *image_size),
                dtype=dtype,
                device=device,
            )
        else:
            x_start = x_start.to(dtype=dtype, device=device)

        if dm._use_cond:
            print("Using Cond")
            cond = cond.to(dtype=dtype, device=device)
            cond = dm.cond_stage_model(cond)

        if dm.use_depth:
            depth = depth.to(dtype=dtype, device=device)
            depth = dm.depth_stage_model(depth)
            assert len(cond) == len(depth), "Unexpected exception."
            cond = [cond, depth]
            print("Depth features added to cond")

        if dm._use_text:
            print("Using Text")
            assert (
                text_cond is not None
            ), "Model uses text conditioning but none is given"
            with torch.no_grad():
                text_cond = dm.text_stage_model(text_cond).to(
                    dtype=dtype, device=device
                )

        uc_cond = self.diffusion_model.cond_stage_model(
            torch.zeros((1, 3, 512, 512), dtype=torch.float16, device="cuda")
        )
        uc_depth = self.diffusion_model.depth_stage_model(
            torch.zeros((1, 1, 512, 512), dtype=torch.float16, device="cuda")
        )
        self.uc_ccond = [uc_cond, uc_depth]
        self.uc_text_embd = self.diffusion_model.text_stage_model([""])

        out, _ = self.p_sample_loop(
            x_start,
            text_cond,
            cond,
            return_intermediate,
        )

        if not self.ldm:
            return reverse_transform(out)

        if return_intermediate:
            b = out.shape[0]
            for t in tqdm(
                range(b), desc="Decoding time sequences to pixel space", total=b
            ):
                out[t] = dm.decode(out[t].to("cuda")).cpu()
        else:
            print("Decoding the final timestep")
            self.decoder.to(dtype)
            out = dm.decode(out[-1].to("cuda"))[None].cpu()

        return reverse_transform(out), self.logger

    def condition_transform(self, scene):
        if not torch.is_tensor(scene):
            scene = self.transform(scene)

        depth = self.depth_model(self.depth_model.midas_prep(scene.to("cuda")))
        cond = 2 * scene[None] - 1
        depth = 2 * depth[None] - 1
        return scene, cond, depth

    def _get_x0(self, xt, epsilon, t, ti):
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, ti, xt.shape
        ).to(xt.dtype)
        sqrt_recip_alphas_cumprod_t = extract(
            self.sqrt_recip_alphas_cumprod, ti, xt.shape
        ).to(xt.dtype)

        pred_xstart = (
            xt - sqrt_one_minus_alphas_cumprod_t * epsilon
        ) * sqrt_recip_alphas_cumprod_t

        return pred_xstart

    def _get_xtm1(self, x0, epsilon, t, ti):
        sqrt_alpha_cumprod_prev_t = torch.sqrt(
            extract(self.alphas_cumprod_prev, ti, x0.shape)
        ).to(x0.dtype)
        ddim_coeff_t = extract(self.ddim_coeff, ti, x0.shape).to(x0.dtype)
        ddim_sigma_t = extract(self.ddim_sigma, ti, x0.shape).to(x0.dtype)
        nonzero_mask = (
            (1 - (t == 0).float())
            .reshape(x0.shape[0], *((1,) * (len(x0.shape) - 1)))
            .to(x0.dtype)
        )
        noise = torch.randn_like(x0)

        xtm1 = (
            sqrt_alpha_cumprod_prev_t * x0
            + ddim_coeff_t * epsilon
            + nonzero_mask * ddim_sigma_t * noise
        )

        return xtm1

    def _decode(self, x0):
        out = []
        for i in range(x0.shape[0]):
            out.append(
                checkpoint(
                    self.decoder.decode,
                    x0[i : i + 1].to(self.decode_dtype)
                    / self.scale.to(self.decode_dtype),
                ).to(self.dtype)
            )
        out = torch.cat(out, axis=0)
        out = rearrange(out, "(b g) c h w -> b (g c) h w", c=3, g=7)
        return out

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

    @torch.no_grad()
    def _permute_idxs(self, x, permuted_idxs):
        if permuted_idxs is None:
            return x
        order = list(permuted_idxs)
        x_rearr = rearrange(x, "b (n c) h w -> b n c h w", n=7)
        x_rearr = x_rearr[:, order]
        return rearrange(x_rearr, "b n c h w -> b (n c) h w", n=7)


class Guided_Layer_Sampler(DDIM_Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = loss_logger()

        u2net_ckpt = to_cobl_path("cobl/U2Net/u2net.pth")
        self.u2net = load_mask_net(u2net_ckpt)

        # Make a unet clone for SDS
        self.unet_clone = deepcopy(self.diffusion_model.model)
        with torch.no_grad():
            for name, param in self.unet_clone.named_parameters():
                if name.endswith("alpha"):
                    # change alphas to their non-modified state
                    param.fill_(1.0)
        self.unet_clone.to("cuda").to(torch.float16)
        self.unet_clone.dtype = torch.float16
        self.unet_clone.convert_to_fp16()

        self.shadow_norm = True

    def guidance_call(self, x_t, t, ti, xcond_embd, ccond_embd):
        xtp = x_t.clone()
        xtp.requires_grad_(True)
        xc = xcond_embd.clone()
        xc.requires_grad_(True)
        cc = [
            [tensor.clone().detach().requires_grad_(True) for tensor in sublist]
            for sublist in ccond_embd
        ]

        ### Main Guidance Body
        # repredict x0 because guidance may be looped for gradient descent
        epsilon = checkpoint(self.diffusion_model.model, xtp, t, xc, cc)
        x0_raw = self._get_x0(xtp, epsilon, t, ti)
        x0 = rearrange(x0_raw, "b (g z) h w -> (b g) z h w", b=1, z=4)
        x0 = self._decode(x0)  # pixel space [1, 21, 512, 512]
        x0 = rearrange(x0[0], "(n c) h w -> n c h w", c=3)  # [7, 3, 512, 512]
        x0 = (x0 + 1) / 2
        x0 = torch.clamp(x0, 0, 1)

        # Compute composition loss
        mask_pred = u2net_masks(x0, self.u2net).to(dtype=x0.dtype, device=x0.device)
        _, composite_loss = calc_composite_loss(
            x0, mask_pred, self.scene, self.shadow_norm
        )
        comp_lval = composite_loss.item()
        self.logger.composite_loss.append(comp_lval)

        # print(mask_pred.shape, x0.shape)
        # fig, ax = plt.subplots(2, 7, figsize=(3 * 7, 3 * 2))
        # for i in range(7):
        #     ax[0, i].imshow(permute(x0[i]).detach().cpu().numpy().astype(np.float32))
        #     ax[1, i].imshow(permute(mask_pred[i]).cpu().numpy().astype(np.float32))

        # for axi in ax.flatten():
        #     axi.axis("off")
        # plt.tight_layout()

        # composite_out = x0[-1]
        # for img, mask in reversed(list(zip(x0[:-1], mask_pred))):
        # #for img, mask in list(zip(x0[:-1], mask_pred)):
        #     composite_out = img * mask + composite_out * (1 - mask)
        #     fig, ax = plt.subplots(1,3)
        #     ax[0].imshow(permute(img).detach().cpu().numpy().astype(np.float32))
        #     ax[1].imshow(permute(mask).detach().cpu().numpy().astype(np.float32))
        #     ax[2].imshow(permute(composite_out).detach().cpu().numpy().astype(np.float32))

        #     plt.tight_layout()

        # # Get permutation order based on layer, mask_pred, and computed depth
        # with torch.no_grad():
        #     isolated_layers = x0[:-1] * mask_pred[:-1] + (1-mask_pred[:-1]) * x0[-1:]
        #     depth = torch.zeros((6,1,512,512), dtype=torch.float16, device='cuda')
        #     for i in range(6):
        #         _, _, depth_est = self.condition_transform(isolated_layers[i], already_tensor=True)
        #         depth[i] = depth_est

        #     masked_depth = depth * mask_pred[:-1]
        #     max_depth = torch.amax(masked_depth, dim=(-1,-2, -3)).flatten()
        #     max_depth = torch.round(max_depth * 100) / 100
        #     max_depth = torch.where(max_depth==0.00, torch.ones_like(max_depth)*1.1, max_depth)

        #     fig, ax = plt.subplots(2,6)
        #     for i in range(6):
        #         ax[0, i].imshow(permute(isolated_layers[i]).cpu().numpy().astype(np.float32))
        #         ax[1, i].imshow(permute(depth[i]).cpu().numpy().astype(np.float32))
        #         ax[1, i].set_title(np.round(max_depth[i].item(), 2))

        #     if t[0] <= 600:
        #         permutation_order = torch.argsort(max_depth, dim=0)  # ascending order by default
        #         permutation_order = permutation_order.squeeze().cpu().numpy()[::-1]
        #     else:
        #         permutation_order = None
        # permutation_order = None

        # with torch.no_grad():
        #     permutation_order = get_best_permutation(mask_pred, x0, self.scene, num_idxs=6)
        #     if permutation_order == list(range(7)):
        #         permutation_order = None
        #     print(permutation_order)
        permutation_order = None

        # Compute SDS Loss
        with torch.no_grad():
            renoised_xt = self.diffusion_model.q_sample(x0_raw, ti)
            renoised_xt = renoised_xt.to(x_t.dtype)
            uc_text_embd = repeat(
                self.uc_text_embd.to(torch.float16),
                "b s c -> (b g) s c",
                g=xcond_embd.shape[0],
            )
            eps_uc = self.unet_clone(renoised_xt, t, uc_text_embd, ccond_embd)
        sds_loss = torch.mean((epsilon - eps_uc) ** 2)
        sds_lval = self.sds_weighting * sds_loss.item()
        self.logger.sds_loss.append(sds_lval)

        ### Gradient Update
        total_loss = composite_loss + self.sds_weighting * sds_loss
        self.logger.total_loss.append(total_loss.item())
        total_loss.backward()

        with torch.no_grad():
            # xtp -= self.guidance_weight*xtp.grad #/ torch.norm(xtp.grad) # dont normalize because can divide by zero issues
            xtp -= self.guidance_weight * xtp.grad / (torch.norm(xtp.grad) + 1e-6)
            xtp.grad.zero_()

        xtp = xtp.detach()
        return xtp, permutation_order

    def get_composition(self, x0):
        istensor = torch.is_tensor(x0)
        if not istensor:
            x0 = torch.tensor(x0, dtype=torch.float16, device="cuda")

        x0 = rearrange(x0, "h w (n c) -> n c h w", c=3)  # [7, 3, 512, 512]
        mask_pred = u2net_masks(x0, self.u2net).to(dtype=x0.dtype, device=x0.device)
        composite = composite_imgs(x0, mask_pred)

        if not istensor:
            composite = composite.cpu().numpy().astype(np.float32)
            mask_pred = mask_pred.cpu().numpy().astype(np.float32)

        return composite, mask_pred
