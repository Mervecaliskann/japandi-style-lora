"""Japandi Room Restyle: my own Japandi style LoRA + ControlNet.

Tab 1 (Restyle my room): Canny ControlNet + img2img keeps the walls and furniture in place,
the LoRA (trigger word "jpdi") brings the Japandi style. Shows the result without and with the LoRA.
Tab 2 (Generate a room): text-to-image with an adjustable LoRA strength.
LoRA: https://huggingface.co/Mervecaliskan/japandi-interior-lora
"""
import os
import time

ZERO_GPU = bool(os.environ.get("SPACES_ZERO_GPU"))   # set by Hugging Face on ZeroGPU Spaces
if ZERO_GPU:
    import spaces                                     # must be imported before torch

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionPipeline
from PIL import Image

DEVICE = "cuda" if (ZERO_GPU or torch.cuda.is_available()) else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
WIDTH = 768 if DEVICE == "cuda" else 512
STEPS = 25 if DEVICE == "cuda" else 15
LORA_REPO = "Mervecaliskan/japandi-interior-lora"
TRIGGER = "jpdi"
NEGATIVE = "blurry, distorted, low quality, deformed, text, watermark"

# ---------- models (loaded once when the Space starts) ----------
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=DTYPE)
pipe_cn = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet, torch_dtype=DTYPE,
    safety_checker=None, requires_safety_checker=False,
).to(DEVICE)
pipe_cn.load_lora_weights(LORA_REPO)

# text-to-image pipeline that shares the same weights (and the same LoRA)
pipe_t2i = StableDiffusionPipeline(
    vae=pipe_cn.vae, text_encoder=pipe_cn.text_encoder, tokenizer=pipe_cn.tokenizer,
    unet=pipe_cn.unet, scheduler=pipe_cn.scheduler,
    safety_checker=None, feature_extractor=None, requires_safety_checker=False,
)

ROOMS = ["living room", "bedroom", "kitchen", "dining room", "bathroom", "home office"]


# ---------- helpers ----------
def resize_keep_ratio(img: Image.Image) -> Image.Image:
    w, h = img.size
    new_h = int(round(h * WIDTH / w / 8)) * 8          # Stable Diffusion needs multiples of 8
    return img.convert("RGB").resize((WIDTH, max(new_h, 64)))


def canny_map(room: Image.Image) -> Image.Image:
    edges = cv2.Canny(np.array(room), 100, 200)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def gpu(fn):
    return spaces.GPU(duration=90)(fn) if ZERO_GPU else fn


# ---------- tab 1: restyle a photo ----------
@gpu
def restyle(image, room_type, lora_scale, change, keep_layout):
    if image is None:
        raise gr.Error("Please upload an interior photo first.")
    t0 = time.perf_counter()
    room = resize_keep_ratio(image)
    edges = canny_map(room)
    prompt = f"a photo of a {TRIGGER} style {room_type}, photorealistic interior, natural light"

    def run(scale):
        return pipe_cn(prompt, negative_prompt=NEGATIVE, image=room, control_image=edges,
                       strength=change, num_inference_steps=STEPS,
                       controlnet_conditioning_scale=keep_layout,
                       cross_attention_kwargs={"scale": scale},
                       generator=torch.manual_seed(42)).images[0]

    without_lora, with_lora = run(0.0), run(lora_scale)
    seconds = time.perf_counter() - t0
    info = (f"Same prompt, seed and settings. Only the LoRA strength changes "
            f"(0 vs {lora_scale:.1f}). Change {change:.2f}, keep layout {keep_layout:.1f}. {seconds:.0f} s on {'GPU' if DEVICE == 'cuda' else 'CPU'}.")
    return without_lora, with_lora, info


# ---------- tab 2: generate from text ----------
@gpu
def generate(room_type, lora_scale, seed):
    prompt = f"a photo of a {TRIGGER} style {room_type}"
    return pipe_t2i(prompt, negative_prompt=NEGATIVE, num_inference_steps=STEPS,
                    cross_attention_kwargs={"scale": lora_scale},
                    generator=torch.manual_seed(int(seed))).images[0]


# ---------- UI ----------
with gr.Blocks(title="Japandi Room Restyle") as demo:
    gr.Markdown(
        "# Japandi Room Restyle\n"
        "A style LoRA I trained on 17 Japandi interiors (4.9 minutes on a free T4 GPU, 6.4 MB). "
        "Combined with ControlNet, it changes the **style** of your room while the walls and furniture stay in place.\n\n"
        f"[LoRA on Hugging Face](https://huggingface.co/{LORA_REPO}) · "
        "[GitHub](https://github.com/Mervecaliskann)"
    )
    with gr.Tab("Restyle my room"):
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Interior photo")
                room_type = gr.Dropdown(ROOMS, value="living room", label="Room type")
                scale = gr.Slider(0.0, 1.0, value=1.0, step=0.1, label="LoRA strength")
                change = gr.Slider(0.3, 0.95, value=0.9, step=0.05,
                                   label="How much to change (img2img strength)",
                                   info="Lower keeps more of the original, higher changes more.")
                keep = gr.Slider(0.3, 1.0, value=0.8, step=0.1,
                                 label="How strictly to keep the layout (ControlNet scale)",
                                 info="1.0 locks every edge (style changes little). Lower lets the style change colors and details.")
                btn = gr.Button("Restyle", variant="primary")
            with gr.Column():
                out_without = gr.Image(label="Without LoRA (base model)")
                out_with = gr.Image(label="With Japandi LoRA")
                info = gr.Markdown()
        btn.click(restyle, inputs=[inp, room_type, scale, change, keep], outputs=[out_without, out_with, info])

    with gr.Tab("Generate a room"):
        with gr.Row():
            with gr.Column():
                room2 = gr.Dropdown(ROOMS, value="bedroom", label="Room type")
                scale2 = gr.Slider(0.0, 1.0, value=1.0, step=0.1, label="LoRA strength (0 = off)")
                seed = gr.Number(value=42, precision=0, label="Seed")
                btn2 = gr.Button("Generate", variant="primary")
                gr.Markdown("Bathroom and home office were **not** in the training photos: "
                            "they show whether the style transfers to new rooms.")
            out2 = gr.Image(label="Generated room")
        btn2.click(generate, inputs=[room2, scale2, seed], outputs=out2)

demo.queue(max_size=5).launch()
