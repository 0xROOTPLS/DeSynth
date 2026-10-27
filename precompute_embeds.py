# One-time: encode desynth's fixed prompts with Qwen-Image 2.1's Qwen3-VL text
# encoder and cache them, so desynth.py never loads the ~17.5 GB encoder.

from __future__ import annotations

from pathlib import Path

import torch
from diffusers import QwenImage21Pipeline

ROOT = Path(__file__).parent
EMBEDS_CACHE = ROOT / "embeds_cache.pt"
QWEN_IMAGE_REPO = "Qwen/Qwen-Image-2.1"
PROMPT_POS = ""
PROMPT_NEG = "blurry, ugly, bad quality,"


def main() -> None:
    # Text encoder only, on CPU: bf16 needs ~18 GB of RAM, more than 8 GB VRAM
    # holds, and a single short forward pass is quick anyway. Going through the
    # pipeline's encode_prompt keeps the template, system-token drop and
    # pre-norm hidden state exactly as 2.1 expects them.
    pipe = QwenImage21Pipeline.from_pretrained(
        QWEN_IMAGE_REPO, transformer=None, vae=None, dtype=torch.bfloat16
    )
    device = torch.device("cpu")

    cache = {}
    for key, prompt in (("prompt", PROMPT_POS), ("negative_prompt", PROMPT_NEG)):
        embeds, _, _ = pipe.encode_prompt(prompt, device=device)
        cache[f"{key}_embeds"] = embeds[0]
        cache[f"{key}_embeds_mask"] = torch.ones(embeds.shape[1], dtype=torch.int64)
    cache["meta"] = {
        "template": pipe.prompt_template_t2i,
        "drop_idx": pipe._drop_idx,
        "prompt_pos": PROMPT_POS,
        "prompt_neg": PROMPT_NEG,
        "source": f"{QWEN_IMAGE_REPO}/text_encoder",
    }

    torch.save(cache, EMBEDS_CACHE)
    print(
        f"wrote {EMBEDS_CACHE.name}: pos={tuple(cache['prompt_embeds'].shape)} "
        f"neg={tuple(cache['negative_prompt_embeds'].shape)}"
    )


if __name__ == "__main__":
    main()
