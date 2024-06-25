import sys
import cv2
import torch
import numpy as np
import gradio as gr
import PIL 
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from imwatermark import WatermarkEncoder
from pathlib import Path

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

from fastapi import FastAPI
from pydantic import BaseModel, Field

import random

torch.set_grad_enabled(False)


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
        "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
        "txt": num_samples * [txt],
        "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
        "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
    }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h // 8, w // 8)
    start_code = torch.from_numpy(start_code).to(
        device=device, dtype=torch.float32)

    with torch.no_grad(), \
            torch.autocast("cuda"):
        batch = make_batch_sd(image, mask, txt=prompt,
                              device=device, num_samples=num_samples)

        c = model.cond_stage_model.encode(batch["txt"])

        c_cat = list()
        for ck in model.concat_keys:
            cc = batch[ck].float()
            if ck != model.masked_image_key:
                bchw = [num_samples, 4, h // 8, w // 8]
                cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
            else:
                cc = model.get_first_stage_encoding(
                    model.encode_first_stage(cc))
            c_cat.append(cc)
        c_cat = torch.cat(c_cat, dim=1)

        # cond
        cond = {"c_concat": [c_cat], "c_crossattn": [c]}

        # uncond cond
        uc_cross = model.get_unconditional_conditioning(num_samples, "")
        uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

        shape = [model.channels, h // 8, w // 8]
        samples_cfg, intermediates = sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=False,
            eta=1.0,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=uc_full,
            x_T=start_code,
        )
        x_samples_ddim = model.decode_first_stage(samples_cfg)

        result = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                             min=0.0, max=1.0)

        result = result.cpu().numpy().transpose(0, 2, 3, 1) * 255
    return [put_watermark(Image.fromarray(img.astype(np.uint8)), wm_encoder) for img in result]

def pad_image(input_image):
    pad_w, pad_h = np.max(((2, 2), np.ceil(
        np.array(input_image.size) / 64).astype(int)), axis=0) * 64 - input_image.size
    im_padded = Image.fromarray(
        np.pad(np.array(input_image), ((0, pad_h), (0, pad_w), (0, 0)), mode='edge'))
    return im_padded

# def predict(input_image, prompt, ddim_steps, num_samples, scale, seed):
def predict(init_image_src, init_mask_src, prompt_tx, pre_id, seed, scale, ddim_steps):
    # init_image = Image.open("C:\\Users\\tobal\\OneDrive\\Github\\stablediffusion\\test.jpeg")
    # init_mask = Image.open("C:\\Users\\tobal\\OneDrive\\Github\\stablediffusion\\mask.jpeg")

    print(f"{init_image_src}, {init_mask_src}, {prompt_tx}")
    

    init_image = Image.open(init_image_src)
    init_mask = Image.open(init_mask_src)
    init_image = init_image.convert("RGB")
    init_mask = init_mask.convert("RGB")

    image = pad_image(init_image) # resize to integer multiple of 32
    mask = pad_image(init_mask) # resize to integer multiple of 32
    width, height = image.size
    print("Inpainting...", width, height)

    result = inpaint(
        sampler=sampler,
        image=image,
        mask=mask,
        prompt=prompt_tx,
        seed=seed,
        scale=scale,
        ddim_steps=ddim_steps,
        num_samples=1,
        h=height, w=width
    )
    
    result[0].save(f"C:\\Users\\tobal\\OneDrive\\Github\\SAM-SD_Integration\\www\\demo\\{pre_id}_result.jpg")


sampler = initialize_model(config = "C:\\Users\\tobal\\OneDrive\\Github\\stablediffusion\\configs\\stable-diffusion\\v2-inpainting-inference.yaml", 
                           ckpt = "C:\\Users\\tobal\\OneDrive\\Github\\stablediffusion\\checkpoints\\512-inpainting-ema.ckpt")

def get_random_seed():
    return random.randint(0, 10000000)


class Request(BaseModel):
    init_image_src: str
    init_mask_src: str
    prompt_tx: str
    pre_id: str
    seed: int = Field(default_factory = get_random_seed)
    scale: int
    ddim_steps: int


app = FastAPI()

@app.post("/")
async def run_wrapper(request: Request):
    predict(init_image_src = request.init_image_src, init_mask_src = request.init_mask_src, prompt_tx = request.prompt_tx, pre_id = request.pre_id, seed = request.seed, scale = request.scale, ddim_steps = request.ddim_steps)
    return "Success"

np.log(random.random())

