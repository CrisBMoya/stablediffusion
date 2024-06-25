import sys
import cv2
import torch
import numpy as np
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from imwatermark import WatermarkEncoder
from pathlib import Path

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config


torch.set_grad_enabled(False)

def predict(input_image):
    init_image = input_image["image"].convert("RGB")
    init_mask = input_image["mask"].convert("RGB")

    print(init_image)
    init_image.save("test.jpeg", "JPEG")
    init_mask.save("mask.jpeg", "JPEG")

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## Stable Diffusion Inpainting")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source='upload', tool='sketch', type="pil")
            prompt = gr.Textbox(label="Prompt")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Advanced options", open=False):
                num_samples = gr.Slider(
                    label="Images", minimum=1, maximum=4, value=4, step=1)
                ddim_steps = gr.Slider(label="Steps", minimum=1,
                                       maximum=50, value=45, step=1)
                scale = gr.Slider(
                    label="Guidance Scale", minimum=0.1, maximum=30.0, value=10, step=0.1
                )
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=2147483647,
                    step=1,
                    randomize=True,
                )
        with gr.Column():
            gallery = gr.Gallery(label="Generated images", show_label=False).style(
                grid=[2], height="auto")

    run_button.click(fn=predict, inputs=[
                     input_image], outputs=[gallery])


block.launch()
