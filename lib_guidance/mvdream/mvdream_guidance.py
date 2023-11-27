import os, os.path as osp, sys

sys.path.append(osp.dirname(__file__))

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    PNDMScheduler,
    DDIMScheduler,
    StableDiffusionPipeline,
)
from diffusers.utils.import_utils import is_xformers_available

# suppress partial model loading warning
logging.set_verbosity_error()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


def normalize_camera(camera_matrix):
    """normalize the camera location onto a unit-sphere"""
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1, 4, 4)
        translation = camera_matrix[:, :3, 3]
        translation = translation / (np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8)
        camera_matrix[:, :3, 3] = translation
    else:
        camera_matrix = camera_matrix.reshape(-1, 4, 4)
        translation = camera_matrix[:, :3, 3]
        translation = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
        camera_matrix[:, :3, 3] = translation
    # ! JH: from this function, the cam_matrix is T_wc;
    return camera_matrix.reshape(-1, 16)


class MVDream(nn.Module):
    def __init__(
        self,
        device,
        fp16=True,
        t_range=[0.02, 0.98],
        n_view=4,
    ):
        super().__init__()

        self.N_view = n_view

        self.device = device

        from mvdream.model_zoo import build_model

        print("Loading MVDream")
        self.model = build_model("sd-v2.1-base-4view")

        model_key = "stabilityai/stable-diffusion-2-1-base"
        self.dtype = torch.float16 if fp16 else torch.float32
        # Create model
        sd_pipe = StableDiffusionPipeline.from_pretrained(model_key, torch_dtype=self.dtype)
        sd_pipe.to(device)
        self.tokenizer = sd_pipe.tokenizer
        self.text_encoder = sd_pipe.text_encoder
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype
        )
        del sd_pipe
        self.model.to(self.device)

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.pos_embeddings, self.neg_embeddings = None, None
        return

    @torch.no_grad()
    def set_text_embeds(self, prompts, negative_prompts):
        pos_embeds = self._encode_text(prompts)  # [1, 77, 768]
        neg_embeds = self._encode_text(negative_prompts)
        self.pos_embeddings = pos_embeds  # [1, 77, 768]
        self.neg_embeddings = neg_embeds  # [1, 77, 768]
        # ! for on the fly modication
        self.prompts = prompts
        # self.embeddings = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]

    def _encode_text(self, prompt):
        # prompt: [str]
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings

    def get_camera_cond(self, camera):
        # camera: B,4,43
        # Note: the input of threestudio is already blender coordinate system
        # camera = convert_opengl_to_blender(camera)
        # ! Camera is T_wc, in blender format
        camera = normalize_camera(camera)
        camera = camera.flatten(start_dim=1)
        return camera

    def train_step(
        self,
        pred_rgb,
        camera,
        step_ratio=None,
        guidance_scale=100,
        as_latent=False,
        return_noise=False,
        noise=None,
        t=None, # can specify t explicitly
    ):
        B = pred_rgb.shape[0]
        assert B == self.N_view, "TODO: check whether MVDream can work on more views?"
        pred_rgb = pred_rgb.to(self.dtype)
        assert len(camera) == B

        if as_latent:
            latents = (
                F.interpolate(pred_rgb, (64, 64), mode="bilinear", align_corners=False) * 2 - 1
            )
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode="bilinear", align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_images(pred_rgb_256)

        if t is None:
            if step_ratio is not None:
                # dreamtime-like
                # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
                t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(
                    self.min_step, self.max_step
                )
                t = torch.Tensor([t]).to(pred_rgb.device).long()
                t_expand = torch.full((B * 2,), int(t), dtype=torch.long, device=self.device)
            else:
                # ! use the same t, not B, but expand to B
                t = torch.randint(
                    self.min_step,
                    self.max_step + 1,
                    (1,),
                    dtype=torch.long,
                    device=self.device,
                )
                t_expand = t.expand(B * 2)
        else:
            t = torch.Tensor([t]).to(pred_rgb.device).long()
            t_expand = torch.full((B * 2,), int(t), dtype=torch.long, device=self.device) 
        # check t is in batchsize

        #######################
        text_embeddings = self.pos_embeddings.expand(B, -1, -1)  # type: ignore
        uncond_text_embeddings = self.neg_embeddings.expand(B, -1, -1)
        # ! FROM MV DIFFUSION THREESTUDIO: IMPORTANT: we return (cond, uncond), which is in different order than other implementations!
        # text_embeddings = torch.cat([text_embeddings, uncond_text_embeddings], dim=0)
        # ! debug here
        text_embeddings = torch.cat([uncond_text_embeddings, text_embeddings], dim=0)
        ########################

        # predict the noise residual with unet, NO grad!
        # ! text_embeddings has two copies
        with torch.no_grad():
            # add noise
            if noise is None:
                noise = torch.randn_like(latents)
            else:
                assert noise.shape == latents.shape
            # latents_noisy = self.scheduler.add_noise(latents, noise, t)
            latents_noisy = self.model.q_sample(latents, t, noise)

            latent_model_input = torch.cat([latents_noisy] * 2)
            # save input tensors for UNet
            if camera is not None:
                camera = self.get_camera_cond(camera)
                camera = camera.repeat(2, 1).to(text_embeddings)
                context = {
                    "context": text_embeddings,
                    "camera": camera,
                    "num_frames": self.N_view,
                }
            else:
                context = {"context": text_embeddings}
            noise_pred = self.model.apply_model(latent_model_input, t_expand, context)

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t, None, None, None]).expand(B, -1, -1, -1)

        # check whether this batch
        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        # seems important to avoid NaN...
        # grad = grad.clamp(-1, 1)

        target = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), target, reduction="sum") / latents.shape[0]

        if return_noise:
            return loss, noise
        return loss

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents

        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs

    # def encode_imgs(self, imgs):
    #     # imgs: [B, 3, H, W]
    #     imgs = 2 * imgs - 1
    #     posterior = self.vae.encode(imgs).latent_dist
    #     latents = posterior.sample() * self.vae.config.scaling_factor
    #     return latents

    def encode_images(self, imgs):
        # B,3,256,256; return B,4,32,32
        imgs = imgs * 2.0 - 1.0
        latents = self.model.get_first_stage_encoding(self.model.encode_first_stage(imgs))
        return latents  # [B, 4, 32, 32] Latent space image


if __name__ == "__main__":
    guidance = MVDream(device="cuda", fp16=False)
    guidance.set_text_embeds(["a photo of a cat"], ["a photo of a dog"])
    pred = torch.randn(4, 3, 256, 256).cuda()
    camera = torch.randn(4, 4, 4).cuda()
    loss = guidance.train_step(pred, camera, step_ratio=0.5, guidance_scale=100, as_latent=False)
    print(loss)
