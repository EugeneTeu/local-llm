"""Qwen-Image 2.1 interactive CLI, via a local mlx-serve server.

Needs the server running separately:
    brew tap ddalcu/mlx-serve https://github.com/ddalcu/mlx-serve
    brew install mlx-serve
    mlx-serve            # holds the model warm on localhost:11234

Run:
    uv run qwen21.py                 # start a session
    uv run qwen21.py "a red fox"     # start with a first prompt
    uv run qwen21.py -r photo.jpg "make it black and white"   # edit a photo

With reference photos (/ref) the prompt is an edit instruction, e.g. "make the
car blue" or "put the person from image 2 in image 1". Needs mlx-serve >= 26.10.1.

Type a prompt to generate. Tweak with /commands (see /help), then /retry.
"""

import argparse
import base64
import json
import os
import random
import shlex
import readline  # noqa: F401  (side effect: line editing + history in input())
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if not os.environ.get("HF_TOKEN"):
    raise SystemExit("HF_TOKEN not set. Add it to .env (see .env.example).")

SERVER = os.environ.get("MLX_SERVE_URL", "http://localhost:11234")
MODEL = "ddalcu/Qwen-Image-2.1-MLX-Serve-4bit"  # 11.2GB, fits the ~12.3GB MLX budget
OUT_DIR = Path("outputs")

MAX_REFS = 10  # 1 image + up to 9 ref_images
MAX_REF_SIDE = 1536  # longer sides are downscaled before upload (edit mode is memory-heavy)

SIZES = {
    "16:9": "1664x928",  # Qwen's native ratio
    "1:1": "1328x1328",
    "4:3": "1472x1104",
    "3:4": "1104x1472",
    "9:16": "928x1664",
}

HELP = """\
  <text>            generate from a new prompt
  /retry            regenerate with the same prompt (new seed unless /seed is set)
  /add <text>       append text to the current prompt and regenerate
  /prompt           show the current prompt
  /ref <path...>    add reference photo(s); prompts then act as edit instructions
  /refs             list references
  /unref [n|all]    remove one reference (by number) or all of them
  /seed <n|random>  pin a seed (reproducible) or go back to a fresh one per run
  /steps <n>        sampling steps (default 40)
  /size <WxH|ratio> e.g. 1664x928, or one of: {sizes}
  /open             open the last image
  /settings         show current settings
  /help             this message
  /quit             exit (Ctrl-D also works)"""


class State:
    def __init__(self):
        self.prompt = ""
        self.steps = 40
        self.size = SIZES["16:9"]
        self.seed = None  # None = fresh each run
        self.refs: list[Path] = []
        self.last_file = None


def generate(s: State) -> None:
    seed = s.seed if s.seed is not None else random.randrange(2**32)
    payload = {
        "model": MODEL,
        "prompt": s.prompt,
        "size": s.size,
        "steps": s.steps,
        "seed": seed,
        "n": 1,
        "response_format": "b64_json",
    }
    if s.refs:
        encoded = [encode_ref(p) for p in s.refs]
        payload["mode"] = "edit"
        payload["image"] = encoded[0]
        if len(encoded) > 1:
            payload["ref_images"] = encoded[1:]
    req = urllib.request.Request(
        f"{SERVER}/v1/images/generations",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    mode = f", editing with {len(s.refs)} ref(s)" if s.refs else ""
    print(f"generating ({s.size}, {s.steps} steps, seed={seed}{mode}) ...")
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=1800) as resp:
            body = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace").strip()
        print(f"mlx-serve returned HTTP {e.code}:\n{detail}")
        return
    except urllib.error.URLError as e:
        print(f"Could not reach mlx-serve at {SERVER} ({e.reason}). Start it with: mlx-serve")
        return

    item = body["data"][0]
    if "b64_json" in item:
        data = base64.b64decode(item["b64_json"])
    elif "url" in item:
        with urllib.request.urlopen(item["url"], timeout=300) as r:
            data = r.read()
    else:
        print(f"Unexpected response shape: {list(item)}")
        return

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"qwen21_{seed}.png"
    path.write_bytes(data)
    s.last_file = path
    print(f"saved {path} in {time.time() - start:.0f}s")
    open_image(path)


def encode_ref(path: Path) -> str:
    """Base64 of the photo, downscaled with macOS `sips` if it is large."""
    data_path = path
    if sys.platform == "darwin":
        out = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
            capture_output=True, text=True,
        ).stdout
        dims = [int(l.split(":")[1]) for l in out.splitlines() if "pixel" in l]
        if dims and max(dims) > MAX_REF_SIDE:
            tmp = Path(tempfile.mkdtemp()) / (path.stem + ".png")
            subprocess.run(
                ["sips", "-Z", str(MAX_REF_SIDE), "-s", "format", "png", str(path), "--out", str(tmp)],
                capture_output=True, check=True,
            )
            data_path = tmp
    return base64.b64encode(data_path.read_bytes()).decode()


def add_refs(s: State, args: list[str]) -> None:
    for a in args:
        path = Path(a).expanduser()
        if not path.is_file():
            print(f"not a file: {a}")
        elif len(s.refs) >= MAX_REFS:
            print(f"max {MAX_REFS} references")
            break
        else:
            s.refs.append(path)
    print(f"refs: {len(s.refs)}")


def open_image(path: Path) -> None:
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    try:
        subprocess.run([opener, str(path)], check=False)
    except FileNotFoundError:
        pass  # no opener available; the path is already printed


def parse_size(arg: str) -> str | None:
    if arg in SIZES:
        return SIZES[arg]
    w, sep, h = arg.lower().partition("x")
    if sep and w.isdigit() and h.isdigit():
        return f"{w}x{h}"
    return None


def command(s: State, line: str) -> bool:
    """Handle a /command. Returns False to quit."""
    cmd, _, arg = line[1:].partition(" ")
    arg = arg.strip()

    if cmd in ("quit", "exit", "q"):
        return False
    elif cmd == "help":
        print(HELP.format(sizes=", ".join(SIZES)))
    elif cmd == "retry":
        if s.prompt:
            generate(s)
        else:
            print("No prompt yet.")
    elif cmd == "add":
        if not arg:
            print("Usage: /add <text>")
        elif not s.prompt:
            print("No prompt yet.")
        else:
            s.prompt = f"{s.prompt.rstrip()} {arg}"
            print(f"prompt: {s.prompt}")
            generate(s)
    elif cmd == "ref":
        # shlex handles quoted paths and drag-and-dropped paths with escaped spaces
        add_refs(s, shlex.split(arg))
    elif cmd == "refs":
        for i, p in enumerate(s.refs, 1):
            print(f"  {i}. {p}")
        if not s.refs:
            print("(none)")
    elif cmd == "unref":
        if arg in ("", "all"):
            s.refs.clear()
            print("refs cleared")
        elif arg.isdigit() and 1 <= int(arg) <= len(s.refs):
            print(f"removed {s.refs.pop(int(arg) - 1)}")
        else:
            print("Usage: /unref [n|all]")
    elif cmd == "prompt":
        print(s.prompt or "(none)")
    elif cmd == "seed":
        if arg in ("", "random"):
            s.seed = None
            print("seed: fresh each run")
        elif arg.isdigit():
            s.seed = int(arg)
            print(f"seed: {s.seed}")
        else:
            print("Usage: /seed <number|random>")
    elif cmd == "steps":
        if arg.isdigit() and int(arg) > 0:
            s.steps = int(arg)
            print(f"steps: {s.steps}")
        else:
            print("Usage: /steps <n>")
    elif cmd == "size":
        size = parse_size(arg)
        if size:
            s.size = size
            print(f"size: {s.size}")
        else:
            print(f"Usage: /size <WxH|{'|'.join(SIZES)}>")
    elif cmd == "open":
        if s.last_file:
            open_image(s.last_file)
        else:
            print("Nothing generated yet.")
    elif cmd == "settings":
        print(f"size={s.size} steps={s.steps} seed={s.seed if s.seed is not None else 'random'}")
        print(f"refs={[str(p) for p in s.refs]}")
        print(f"last={s.last_file}")
    else:
        print(f"Unknown command /{cmd}. Try /help.")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Qwen-Image 2.1 interactive CLI")
    ap.add_argument("-r", "--ref", action="append", default=[], metavar="PHOTO",
                    help="reference photo (repeatable)")
    ap.add_argument("prompt", nargs="*", help="first prompt")
    args = ap.parse_args()

    s = State()
    print("Qwen-Image 2.1 CLI. Type a prompt, or /help.")
    if args.ref:
        add_refs(s, args.ref)

    pending = " ".join(args.prompt).strip()
    if pending:
        s.prompt = pending
        generate(s)

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            if not command(s, line):
                break
        else:
            s.prompt = line
            generate(s)


if __name__ == "__main__":
    main()
