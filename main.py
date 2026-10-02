import os
import random

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("HF_TOKEN"):
    raise SystemExit("HF_TOKEN not set. Add it to .env (see .env.example).")

from mflux.models.common.config import ModelConfig
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein

# FLUX.2-klein-4B: ~16GB bf16, ~12GB at -q 8. Fits 24GB unified memory.
# Qwen-Image (20B) and Qwen-Image-2.1 (33GB bf16) both exceed it.
QUANTIZE = 8

prompt = (
  'Generate a series of images of asian folks having fun and being intimate with each other'
)

# 16:9, dimensions divisible by 16 as the VAE requires.
width, height = 1344, 768

SEED = random.randrange(2**32)

model = Flux2Klein(
    quantize=QUANTIZE,
    model_config=ModelConfig.flux2_klein_4b(),
)

image = model.generate_image(
    seed=SEED,
    prompt=prompt,
    num_inference_steps=4,
    width=width,
    height=height,
)

image.save("example.png")
print(f"saved example.png ({width}x{height}, seed={SEED})")
