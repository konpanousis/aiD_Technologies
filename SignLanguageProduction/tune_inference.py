from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from diffusers import DDIMScheduler
from tuneavideo.util import save_videos_grid, ddim_inversion
import torch

from einops import rearrange

import os

MODEL_NAME = "CompVis/stable-diffusion-v1-4"
MODEL_NAME = f"./checkpoints/{MODEL_NAME}"

OUTPUT_DIR = "outputs/final" #@param {type:"string"}
print(f"[*] Weights will be saved at {OUTPUT_DIR}")
os.makedirs(OUTPUT_DIR, exist_ok = True)

pretrained_model_path = MODEL_NAME
my_model_path = OUTPUT_DIR
subfolder='best'
video_length = 25
num_inv_steps=100
num_inference_steps = 50
guidance_scale = 20

input_latents = torch.load(os.path.join(my_model_path,'random_latent'))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
weight_dtype = torch.float32 if device == 'cpu' else torch.float16


unet = UNet3DConditionModel.from_pretrained(my_model_path, subfolder=subfolder, torch_dtype=weight_dtype).to(device)
validation_pipeline = TuneAVideoPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=weight_dtype).to(device)
validation_pipeline.enable_xformers_memory_efficient_attention()
validation_pipeline.enable_vae_slicing()

prompts = ['the weather today is very nice.']

global_step = subfolder.split("-")[-1]
print(global_step)
samples = []

use_inv_latent = False
ddim_inv_latent = None

for idx, prompt in enumerate(prompts):
    print(prompt)
    ##### Inversion DDIM ######
    if use_inv_latent:
      inv_latents_path = os.path.join(my_model_path, f"inv_latents/ddim_latent-{global_step}-{prompt}.pt")
      ddim_inv_latent = ddim_inversion(
          validation_pipeline, ddim_inv_scheduler, video_latent=input_latents,
          num_inv_steps=num_inv_steps, prompt=prompt)[-1].to(weight_dtype)
      torch.save(ddim_inv_latent, inv_latents_path)
    ##########################

    video = validation_pipeline(prompt, latents=ddim_inv_latent, video_length=video_length, height=512, width=512, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).videos

    save_videos_grid(video, f"{my_model_path}/samples/sample-{global_step}/{prompt}.mp4")
    samples.append(video)
samples = torch.concat(samples)
save_path = f"{my_model_path}/samples/sample-{global_step}.mp4"
save_videos_grid(samples, save_path)
