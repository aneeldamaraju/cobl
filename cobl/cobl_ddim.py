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
from itertools import permutations

from cobl.ddpm_utils import extract
from cobl.midas import MidasDepthEstimator
from cobl.datasets import reverse_transform, CenterSquareCrop, permute_dimensions
from cobl.load_utils import to_cobl_path
from cobl.U2Net.u2net_utils import load_mask_net, u2net_masks




def composite_imgs(imgs, masks):
    composite_out = imgs[-1:]
    for img, mask in reversed(list(zip(imgs[:-1], masks))):
        composite_out = img * mask + composite_out * (1 - mask)
    return composite_out


def calc_composite_loss(
    pred_layers, mask_pred, cond
):
    composite = composite_imgs(pred_layers, mask_pred[:-1])
    composite_loss = torch.mean((composite[0] - cond) ** 2)
    return composite, composite_loss


def calc_psm_loss(dm,x0_raw,epsilon,x_t,t,ti,uc_text_embd,xcond_embd,unet_clone):
    with torch.no_grad():
        renoised_xt = dm.q_sample(x0_raw, ti)
        renoised_xt = renoised_xt.to(x_t.dtype)
        uc_text_embd = repeat(
            uc_text_embd.to(torch.float16),
            "b s c -> (b g) s c",
            g=xcond_embd.shape[0],
        )
        eps_uc = unet_clone(renoised_xt, t, uc_text_embd)
    sds_loss = torch.mean((epsilon - eps_uc) ** 2)
    return sds_loss

### Non differentiable cleaning steps (see supplemental)
def permute_and_clean(t, ti, mask_pred, x0, sample_cond,nlayers):
    t_flat = t.flatten()[0]
    if (ti+1)%5==0:
        permuted_idxs = get_best_permutation(mask_pred, x0, sample_cond, num_idxs=nlayers-1)
        idxs_to_remove = []
    elif (ti+2)%5==0:
        permuted_idxs, idxs_to_remove =  clean_and_sort_masks(mask_pred)
    else:
        permuted_idxs = False
        idxs_to_remove = []
        
    return permuted_idxs,idxs_to_remove
        

def get_best_permutation(masks,predictions,cond, num_idxs = 3):
    masks = masks[:masks.shape[0]-1,0,...].to(predictions.device)
    permuted_idxs = list(permutations(range(num_idxs)))
    permuted_idxs_with_bgs = [list(idx)+[num_idxs] for idx in permuted_idxs]
    composite_target = cond.to(predictions.device).squeeze().unsqueeze(0)
    best_permutation = permuted_idxs_with_bgs[0]
    lowest_loss = 1

    with torch.no_grad():
        for idxs in permuted_idxs_with_bgs:

            permuted_imgs = predictions[idxs,...]
            permuted_composite = composite_imgs(permuted_imgs,masks[idxs[:num_idxs],...])
            composite_loss = F.mse_loss(permuted_composite,composite_target)
            if (composite_loss.item() < lowest_loss):
                best_permutation = idxs
                lowest_loss = composite_loss.item()
                
    return best_permutation

    
def move_empty_masks_to_front(mask_pred,thresh=.8):
    empty_mask_idxs = []
    unempty_mask_idxs = []
    
    for itr,mask in enumerate(mask_pred[:-1]):
        empty_el = torch.sum(mask<thresh)
        if (empty_el/torch.numel(mask)) > 1-1e-3:
            empty_mask_idxs.append(itr)
        else:
            unempty_mask_idxs.append(itr)
    return empty_mask_idxs+unempty_mask_idxs+[itr+1]
    
    
def masks_to_remove(mask_imgs,composite_out,thresh=.8):
    occluded_mask_idxs = []
    for itr, (mask,composite) in enumerate(zip(mask_imgs[1:], composite_out[:-1])):
        intersection = torch.count_nonzero((mask>thresh)*(composite>thresh))
        occlusion_frac = intersection/(torch.sum(mask>thresh) + 1e-5)
        if occlusion_frac > 1-1e-1:
            occluded_mask_idxs.append(itr+1)
    return occluded_mask_idxs
    
def composite_masks(masks):
    #intialize background and outputs
    composite_list = []
    composite_out = masks[0]
    composite_list.append(composite_out)
    for mask in masks[1:]:
        composite_out = mask + composite_out * (1-mask)
        composite_list.append(composite_out)
    return composite_list

def clean_and_sort_masks(mask_pred):
    mask_pred = mask_pred.squeeze()
    permuted_idxs = move_empty_masks_to_front(mask_pred)
    composite_list = composite_masks(mask_pred[:-1])
    idxs_to_remove = masks_to_remove(mask_pred[:-1],composite_list)
    return permuted_idxs, idxs_to_remove

###

def permute(x):
    return x.transpose(-3, -1).transpose(-3, -2)


class loss_logger:
    def __init__(self):
        self.composite_loss = []
        self.psm_loss = []
        self.total_loss = []
        self.masks = []

class DDIM_Sampler(nn.Module):
    def __init__(
        self, diffusion_model, n_steps, ddim_scheme="uniform", ddim_eta=0, cfg_scale=1.75,
        psm_weighting = 1e-6, guidance_weight=1.0,
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
        self.psm_weighting = psm_weighting
        self.guidance_weight = guidance_weight
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
        return  x+T, permutation_order, idxs_to_remove

    def p_sample(self, x_t, t, ti, xcond_embd, ccond_embd):
        ### Get Noise while using CFG
        with torch.no_grad():
            eps = self.diffusion_model.model(x_t, t, xcond_embd, ccond_embd)
            eps_uc = self.diffusion_model.model(x_t, t, xcond_embd, self.uc_ccond)
        epsilon = eps_uc + self.cfg_scale * (eps - eps_uc)

        ### Run Guidance
        if self.use_guidance:
            xtp,permutation_order, idxs_to_remove = self.guidance_call(x_t, t, ti, xcond_embd, ccond_embd)
        
            # x_t = self._permute_idxs(x_t, permuted_idxs)
            # epsilon = self._permute_idxs(epsilon, permuted_idxs)

        ### Get x0hat and Compute DDIM Step
        pred_xstart = self._get_x0(x_t, epsilon, t, ti)
        xtm1 = self._get_xtm1(pred_xstart, epsilon, t, ti)
        
        
        if self.use_guidance:
            nlayers = int(pred_xstart.shape[1]/4)
            pred_xstart = self._permute_idxs(pred_xstart, permutation_order,nlayers)
            pred_xstart = self._remove_latents(pred_xstart, idxs_to_remove,ti,nlayers)

            xtm1 = self._permute_idxs(xtm1, permutation_order,nlayers)
            xtm1 = self._remove_latents(xtm1, idxs_to_remove,ti,nlayers)
        
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
        psm_weighting=None,
        cfg_scale=None,
    ):
        
        # Update the sampling parameters
        self.use_guidance = use_guidance
        if guidance_weight is not None:
            self.guidance_weight = guidance_weight
        if psm_weighting is not None:
            self.psm_weighting = psm_weighting
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
        self.sample_cond = cond
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
            # print("Using Cond")
            cond = cond.to(dtype=dtype, device=device)
            cond = dm.cond_stage_model(cond)

        if dm.use_depth:
            depth = depth.to(dtype=dtype, device=device)
            depth = dm.depth_stage_model(depth)
            assert len(cond) == len(depth), "Unexpected exception."
            cond = [cond, depth]
            # print("Depth features added to cond")

        if dm._use_text:
            # print("Using Text")
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
    def _permute_idxs(self,x,permuted_idxs,nlayers):
        '''
        Permute dimensions of x according to permuted_idxs 
        Given input x, rearranges and returns a new x
        '''
        if permuted_idxs:
            dm = self.diffusion_model
            x_rearr = rearrange(x,'b (n c) h w -> b n c h w', n = nlayers)
            x_rearr = x_rearr[:,permuted_idxs, ...]
            return rearrange(x_rearr, 'b n c h w -> b (n c) h w',n=nlayers)
        else:
            return x
    
    @torch.no_grad()
    def _remove_latents(self,x,idxs_to_remove, ti, nlayers, zero=True):
        '''
        Remove selected latents and replace them with either zeros or a designated empty image latent
        Given input x, removes, rearranges and returns a new x
        '''
        dm = self.diffusion_model
        x_rearr = rearrange(x,'b (n c) h w -> b n c h w', n = nlayers)
        if zero:
            replacement_tensor = torch.zeros_like(x_rearr[:,0,...])
        else:
            replacement_tensor = dm.q_sample(self.empty_image_latent, ti)
        x_rearr[:,idxs_to_remove, ...] = replacement_tensor
        return rearrange(x_rearr, 'b n c h w -> b (n c) h w',n=nlayers)

class Guided_Layer_Sampler(DDIM_Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = loss_logger()
        self.sd_sampler = DDIM_Sampler(self.diffusion_model, self.n_steps)
        dtype = torch.float16
        
          
        u2net_ckpt = to_cobl_path("cobl/U2Net/u2net.pth")
        self.u2net = load_mask_net(u2net_ckpt)
        
        self.unet_clone = deepcopy(self.diffusion_model.model)
        with torch.no_grad():
            for name, param in self.unet_clone.named_parameters():
                if name.endswith("alpha"):
                    # change alphas to their non-modified state
                    param.fill_(1.0)
        self.unet_clone.to("cuda").to(torch.float16)
        self.unet_clone.dtype = torch.float16
        self.unet_clone.convert_to_fp16()

    

    
    def guidance_call(self, x_t, t, ti, xcond_embd, ccond_embd):
        nlayers = int(x_t.shape[1]/4) # TODO: remove dynamically allocate n_layers
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
            x0, mask_pred, self.scene
        )
        self.logger.masks.append(mask_pred.cpu().detach().squeeze().numpy())
        comp_lval = composite_loss.item()
        self.logger.composite_loss.append(comp_lval)
        
        # Compute PSM Loss
        # psm_loss = calc_psm_loss(x0_raw,t,ti,self.sd_sampler,self.uc_text_embd)
        psm_loss = calc_psm_loss(self.diffusion_model,x0_raw,epsilon,xtp,t,ti,self.uc_text_embd,xcond_embd,self.unet_clone)
        psm_lval = self.psm_weighting * psm_loss.item()
        self.logger.psm_loss.append(psm_lval)

        ### Gradient Update
        total_loss = composite_loss + self.psm_weighting * psm_loss
        self.logger.total_loss.append(total_loss.item())
        total_loss.backward()

        
        #Run permutation and empty layer check (see paper supplement)
        with torch.no_grad():
            permutation_order, idxs_to_remove =  permute_and_clean(t,ti, mask_pred, x0, self.sample_cond,nlayers)
        
        
        with torch.no_grad():
            xtp -= self.guidance_weight * xtp.grad / (torch.norm(xtp.grad) + 1e-8)
            xtp.grad.zero_()

        xtp = xtp.detach()
        return xtp,permutation_order, idxs_to_remove

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
