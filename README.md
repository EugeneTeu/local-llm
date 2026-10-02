# Local image generation on 2026 macbook pro

- 24 GB of ram
- wanted to use some free image generation

# qwen

Local text-to-image generation on Apple silicon.

Two independent paths:

| script | model | runtime | memory | notes |
|---|---|---|---|---|
| `qwen21.py` | Qwen-Image 2.1 (4-bit) | mlx-serve over HTTP | ~14.4 GB resident | Best-in-class text rendering, incl. Chinese |
| `main.py` | FLUX.2-klein-4B | mflux, in-process | ~12 GB | Self-contained, 4 steps, no server |

Developed on an M5 Pro / 24 GB. Both fit; neither has much headroom.

## Setup

```sh
uv sync
cp .env.example .env     # then add your HF_TOKEN
```

`HF_TOKEN` is read by both scripts via `python-dotenv`. `huggingface_hub` picks
it up from the environment, so nothing passes it explicitly.

`.env` is gitignored. `.env.example` is not — keep it in sync when adding keys.

## main.py — FLUX.2-klein (no server)

```sh
uv run main.py          # -> example.png
```

First run downloads ~16 GB. Loads the model on every invocation; nothing stays
warm. Good enough for one-off generations.

## qwen21.py — Qwen-Image 2.1 (via mlx-serve)

Needs a local server holding the model in memory. The model loads once and
stays warm, so repeat generations skip the load entirely.

### 1. Install mlx-serve

```sh
brew tap ddalcu/mlx-serve https://github.com/ddalcu/mlx-serve
brew install mlx-serve          # CLI only; `--cask mlx-core` for the GUI app
```

Note: the `mlx-serve` on PyPI is an unrelated project. Don't install that one.

### 2. Download the weights

`mlx-serve pull` does NOT work for these image repos. It fetches only the two
root-level JSON files, reports `2 files, 0 MB total` and `success: ... ready`,
and never recurses into `transformer/`, `text_encoder/` or `vae/`. The model
then registers in `/v1/models` but fails to load with
`QwenImageConfigMissing` — which looks like a download you forgot to run.

Fetch the weights directly into the directory mlx-serve reads from:

```sh
HF_HUB_ENABLE_HF_TRANSFER=1 uv run python - <<'PY'
from dotenv import load_dotenv; load_dotenv()
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ddalcu/Qwen-Image-2.1-MLX-Serve-4bit",
    local_dir="~/.mlx-serve/models/ddalcu/Qwen-Image-2.1-MLX-Serve-4bit",
    max_workers=8,
)
PY
```

(Expand `~` to your home directory — `snapshot_download` won't.)

`HF_HUB_ENABLE_HF_TRANSFER=1` matters: it cut the download from ~30 min to ~4.
`hf_transfer` is already a dependency but is opt-in via that variable.

Verify — `SIZE` must not be `0 KB`:

```sh
mlx-serve list
# ddalcu/Qwen-Image-2.1-MLX-Serve-4bit    image    10.4 GB
```

### 3. Run

```sh
mlx-serve serve --skip-mem-preflight --max-resident-mem 20GB     # leave running
uv run qwen21.py                                                 # interactive CLI
```

`qwen21.py` is an interactive prompt: type a prompt to generate, then tweak with
`/retry`, `/add`, `/seed`, `/steps`, `/size`. Images land in `outputs/`. `/help`
lists everything.

Reference photos (instruction editing):

```sh
uv run qwen21.py -r photo.jpg "make the car blue"
```

or inside the prompt: `/ref photo.jpg other.png`, `/refs`, `/unref [n|all]`.
With references loaded, the prompt is an edit instruction ("put the person from
image 2 in image 1"), up to 10 photos. Photos with a side over 1536 px are
downscaled (macOS `sips`) before upload. Edits are slower than plain generation
(~45 s at `/steps 4`, minutes at the default 40).

Both server flags are required:

- `--skip-mem-preflight`: the preflight computes a budget of ~11.7 GB and
  refuses the model, which needs ~14.4 GB resident (10.4 GB on disk — MLX
  expands it during staged load). The machine genuinely has the room; the check
  is conservative. Without the flag you get HTTP 503 `out_of_memory`.
- `--max-resident-mem 20GB`: a separate cap on total loaded-model size. It
  defaults to `auto` (80% of the MLX wired limit, ~12 GB on 24 GB), which is
  below the model's resident size. On mlx-serve 26.10.1 this also gives HTTP 503
  `out_of_memory` ("exceed the resident-model budget"). 20 GB leaves ~4 GB for
  macOS; `0` disables the cap but risks swapping. Close heavy apps first.

Use mlx-serve >= 26.10.1 (`brew upgrade mlx-serve`) if you want Qwen-Image 2.1
instruction editing; 26.9.x only supports text-to-image and img2img and rejects
`mode: "edit"` with "requires a FLUX.2 or Mage-Flow-Edit model".

mlx-serve raises `iogpu.wired_limit_mb` to 20480 itself. No `sudo sysctl` needed.

Stop the server when done — it holds 14.4 GB indefinitely:

```sh
pkill -f "mlx-serve serve"
```

## Choosing a model

Measured against 24 GB unified memory:

| model | bf16 | verdict |
|---|---|---|
| Qwen-Image (20B) | 57.7 GB | no |
| Qwen-Image 2.1 | 33.1 GB | only quantized |
| ↳ 4-bit (mlx-serve) | 14.4 GB resident | yes |
| ↳ 8-bit (mlx-serve) | ~18.4 GB | too tight |
| FLUX.2-klein-4B | ~16 GB | yes |
| FLUX.1-schnell | ~15 GB | yes |
| SDXL | ~7 GB | yes, comfortably |

Qwen-Image 2.1 is 7.1B + 8B text encoder — a different, smaller model than the
20B original, not a quantization of it.

mflux will not load the mlx-serve 4-bit build: it is MLX affine-quantized
(`"mlx_format": true`), which neither mflux nor diffusers reads. The diffusers
class names in `model_index.json` are a layout convention, not compatibility.

Also note mflux never quantizes the Qwen text encoder — it stays bf16 at
17.5 GB regardless of `-q`, so `mflux -q 4` still won't fit 24 GB. That is the
specific reason this project uses mlx-serve for Qwen and mflux only for klein.

## Research: labeling photos (not built)

Findings from reading mlx-serve's docs and release notes (v26.10.1). Nothing
here was run on this machine, so treat it as "documented", not "tested".

Qwen-Image 2.1 cannot label photos. The server reports it as
`capabilities: ["image"]`, `input_modalities: ["text"]`; its vision tower only
conditions edits. Labeling needs a vision-language model (VLM) behind the same
mlx-serve.

Candidate VLMs listed in the docs (exact HF repo ids not given):

| model | vision encoder | fit on 24 GB |
|---|---|---|
| Gemma 4 E4B / E2B (4-bit) | SigLIP | small, try first |
| Qwen3.5-4B | Qwen3-VL | small |
| LFM2.5-VL 3B / 1.6B | SigLIP2 (tiles big images) | smallest, likely weakest |
| Gemma 4 26B-A4B, Qwen3.6/3.8-27B | — | too big (Qwen3.8 27B is 18.2 GB) |

Open questions, in the order to probe them:

1. `mlx-serve pull` may hit the same bug as for image repos (fetches only the
   root JSON files); check that `mlx-serve list` shows a non-zero size.
2. Memory: Qwen-Image holds ~15 GB, so run one model at a time. The docs list
   `POST /v1/unload-model` and `POST /v1/load-model` (`"default": true` makes it
   the serving model), so a labeler could swap without a server restart.
3. Image input: `/v1/chat/completions` accepts `image_url` content blocks
   (base64 or URL). Whether base64 needs a `data:` URI, and any size limits, are
   undocumented.
4. JSON output: `response_format` is accepted, and the README mentions
   JSON-schema-constrained decoding, but there are no examples and no word on
   whether it works with image input. Fallback: ask for JSON in the prompt,
   validate, retry once.

Likely design if built: a separate `label.py`, stdlib-only like `qwen21.py`,
writing one JSON line per photo to `labels.jsonl` (resumable). Small VLMs are
fine for captions and object lists, weaker at counts, small text and boxes.

## Gotchas

- `uv pip install` does not add to `pyproject.toml`, and `uv run` syncs the
  venv to the lockfile — so pip-installed packages get dropped. Use `uv add`.
- Homebrew ships no bare `pip`/`python`, only `pip3`/`python3`.
- A root-owned `~/.cache/uv` (from a past `sudo uv`) causes
  `Permission denied ... git-v0/locks`. Fix with
  `sudo chown -R "$USER":staff ~/.cache/uv`, and don't run `uv` under sudo.
