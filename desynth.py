# Qwen-Image GGUF img2img + frequency-domain detail restore. Removes SynthID
# from images while keeping them visually close to the source.

from __future__ import annotations

import argparse
import logging
import os
import warnings
from pathlib import Path

# Silence known-harmless noise from torch/diffusers/HF.
warnings.filterwarnings("ignore", message=".*not writable.*")
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*pooled_projection_dim.*")
warnings.filterwarnings("ignore", message=".*classifier-free guidance is not enabled.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
try:
    from diffusers.utils import logging as _df_logging
    _df_logging.set_verbosity_error()
except Exception:
    pass

import torch


# diffusers' load_gguf_checkpoint .copy()s every tensor onto the heap (~10GB
# of independent allocations). Windows' allocator eventually fails. Read
# straight from the mmap instead.
def _patch_gguf_loader() -> None:
    from diffusers.models import model_loading_utils as mlu
    import gguf as _gguf
    from gguf import GGUFReader
    from diffusers.quantizers.gguf.utils import SUPPORTED_GGUF_QUANT_TYPES, GGUFParameter

    def load_gguf_checkpoint(path, return_tensors=False):
        reader = GGUFReader(path)
        parsed = {}
        for t in reader.tensors:
            qt = t.tensor_type
            is_quant = qt not in (_gguf.GGMLQuantizationType.F32, _gguf.GGMLQuantizationType.F16)
            if is_quant and qt not in SUPPORTED_GGUF_QUANT_TYPES:
                raise ValueError(f"{t.name} quant {qt} unsupported")
            w = torch.from_numpy(t.data)
            parsed[t.name] = GGUFParameter(w, quant_type=qt) if is_quant else w
        return parsed

    mlu.load_gguf_checkpoint = load_gguf_checkpoint


# Sequential CPU offload moves params through meta device, dropping quant_type
# and crashing GGUFParameter.__new__. Recover it from the source if possible.
def _patch_gguf_param() -> None:
    from diffusers.quantizers.gguf.utils import GGUFParameter
    _orig = GGUFParameter.__new__

    def _new(cls, data, requires_grad=False, quant_type=None):
        if quant_type is None and isinstance(data, GGUFParameter):
            quant_type = data.quant_type
        if quant_type is None:
            d = data if data is not None else torch.empty(0)
            self = torch.Tensor._make_subclass(cls, d, requires_grad)
            self.quant_type = None
            self.quant_shape = None
            return self
        return _orig(cls, data, requires_grad, quant_type)

    GGUFParameter.__new__ = _new


_patch_gguf_loader()
_patch_gguf_param()

from diffusers import (
    AutoencoderKLQwenImage,
    GGUFQuantizationConfig,
    QwenImageImg2ImgPipeline,
    QwenImageTransformer2DModel,
)
from diffusers.utils import load_image

ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / "Original.png"
OUT_DIR = ROOT / "out"

# Ordered list of GGUF quants to try, best quality first.
# Each entry is (label, [candidate paths]) — first existing path wins per entry.
# On OOM the pipeline tears down and retries with the next entry automatically.
GGUF_FALLBACK_CHAIN: list[tuple[str, list[Path]]] = [
    ("Q4_K_M (~13 GB)", [
        ROOT / "Qwen-Image-2512-Q4_K_M.gguf",
        ROOT / "qwen-image-2512-Q4_K_M.gguf",   # README-suggested lowercase
    ]),
    ("Q2_K_M (~7 GB)", [
        ROOT / "Qwen-Image-2512-Q2_K_M.gguf",
        ROOT / "qwen-image-2512-Q2_K_M.gguf",
    ]),
]

def _resolve_gguf(candidates: list[Path]) -> Path | None:
    """Return the first path in candidates that exists on disk, or None."""
    return next((p for p in candidates if p.exists()), None)

# Default transformer = best available quant right now (used by --transformer default).
GGUF_TRANSFORMER = next(
    (p for _, cands in GGUF_FALLBACK_CHAIN for p in cands if p.exists()),
    GGUF_FALLBACK_CHAIN[0][1][0],  # fallback to first path even if missing (gives clear error)
)
LIGHTNING_LORA = ROOT / "Qwen-Image-2512-Lightning-4steps-V1.0-fp32.safetensors"
EMBEDS_CACHE = ROOT / "embeds_cache.pt"


def _is_oom(exc: BaseException) -> bool:
    """Return True for any out-of-memory error across CUDA, MPS, and CPU."""
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(k in msg for k in ("out of memory", "oom", "memory allocation"))
    # torch.cuda.OutOfMemoryError is a subclass of RuntimeError, already covered.
    return False


def _free_pipeline(pipe) -> None:
    """Aggressively release a pipeline and clear device caches."""
    import gc
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# HF repo used only for tiny configs and the ~250MB VAE.
QWEN_IMAGE_REPO = "Qwen/Qwen-Image"

# Lightning is a 4-step LoRA. We run at 8 steps so denoise quantizes to 1/8;
# 2/8 = 0.25 is the minimum that defeats SynthID. Lower = no-op (input passes
# through). Higher = more drift, no extra benefit.
DEFAULT_STEPS = 8
CFG = 1.0
DEFAULT_DENOISE = 0.25
# SynthID's spatial cutoff is at ~2px. 1.95 is the safe ceiling that takes
# back maximum detail from the original without dragging the watermark along.
DEFAULT_RESTORE_SIGMA = 1.95

# ---------------------------------------------------------------------------
# Device / dtype selection — CUDA > MPS (Apple Silicon) > CPU
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_DTYPE = torch.bfloat16   # native on Ampere+
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    # PyTorch 2.14 has solid bfloat16 MPS coverage. More importantly, the
    # diffusers pipeline's image preprocessor naturally produces bfloat16
    # tensors on MPS — using float16 here causes a dtype mismatch in the
    # VAE encoder (Input bfloat16 vs bias float16). Keep bfloat16 throughout.
    COMPUTE_DTYPE = torch.bfloat16
    # Disable PyTorch's conservative MPS OOM guard (~70% threshold) so macOS's
    # own memory compressor and SSD swap can manage pressure instead.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
else:
    DEVICE = "cpu"
    COMPUTE_DTYPE = torch.float32    # bf16/fp16 are slow on CPU without HW support

print(f"[device] {DEVICE}  dtype={COMPUTE_DTYPE}")


def build_pipeline(transformer_path: Path = GGUF_TRANSFORMER) -> QwenImageImg2ImgPipeline:
    for p in (transformer_path, LIGHTNING_LORA, EMBEDS_CACHE):
        if not p.exists():
            raise FileNotFoundError(
                f"{p}\n(Run `python precompute_embeds.py` first if embeds_cache.pt is missing.)"
            )

    dtype = COMPUTE_DTYPE

    # Offline if cached; download once if not.
    def _load(loader, **kwargs):
        try:
            return loader(**kwargs, local_files_only=True)
        except Exception:
            print("(cache miss — fetching configs/VAE from HF, one-time)")
            return loader(**kwargs, local_files_only=False)

    transformer = _load(
        QwenImageTransformer2DModel.from_single_file,
        pretrained_model_link_or_path_or_dict=str(transformer_path),
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        dtype=dtype,
        config=QWEN_IMAGE_REPO,
        subfolder="transformer",
    )

    vae = _load(
        AutoencoderKLQwenImage.from_pretrained,
        pretrained_model_name_or_path=QWEN_IMAGE_REPO,
        subfolder="vae",
        dtype=dtype,
    )

    # No text encoder loaded — embeds are precomputed.
    pipe = _load(
        QwenImageImg2ImgPipeline.from_pretrained,
        pretrained_model_name_or_path=QWEN_IMAGE_REPO,
        transformer=transformer,
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        dtype=dtype,
    )

    # Load LoRA via safetensors mmap so its 1.6 GB stays SSD-backed (same
    # strategy as the GGUF weights). safetensors.torch.load_file() memory-maps
    # the file by default; tensors are only paged into RAM when accessed.
    try:
        from safetensors.torch import load_file as _st_load
        _lora_sd = _st_load(str(LIGHTNING_LORA), device="cpu")
        pipe.load_lora_weights(_lora_sd, adapter_name="lightning")
    except Exception:
        # Fallback: let diffusers handle it normally (heap allocation).
        pipe.load_lora_weights(str(LIGHTNING_LORA), adapter_name="lightning")
    pipe.set_adapters(["lightning"], adapter_weights=[0.8])

    # Offload strategy depends on available hardware.
    #   CUDA:     sequential offload — moves one layer at a time, fits 13GB GGUF
    #             in 8 GB VRAM (tested config from the original author).
    #   MPS/CPU:  model-level offload — accelerate moves whole sub-modules;
    #             sequential offload's internals are CUDA-only in diffusers.
    #             On CPU this still runs entirely in system RAM.
    if DEVICE == "cuda":
        pipe.enable_sequential_cpu_offload()
    else:
        # enable_model_cpu_offload falls back to CPU→device per sub-module,
        # which accelerate supports on MPS as of accelerate>=0.26.
        try:
            pipe.enable_model_cpu_offload(device=DEVICE)
        except TypeError:
            # Older accelerate doesn't accept 'device' kwarg — move manually.
            pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    return pipe


def _split(arr, sigma):
    import cv2
    f = arr.astype("float32")
    low = cv2.GaussianBlur(f, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return low, f - low


# Gaussian: clean low band + original high band at one fixed sigma.
def _restore_gaussian(clean, original, sigma):
    low, _ = _split(clean, sigma)
    _, high = _split(original, sigma)
    return (low + high).clip(0, 255)


# Edge-aware: two recipes blended by edge mask. At edges, sigma=1.0 pulls
# more original detail back; in flats, sigma=1.95 keeps the watermark dead.
# Watermark survives only where edge mask -> 0, so flat skin/sky still clean.
def _restore_edge(clean, original, sigma_safe, sigma_edge=1.0):
    import cv2
    import numpy as np
    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype("float32")
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), sigmaX=2.0, sigmaY=2.0)
    m = mag / (np.percentile(mag, 99) + 1e-6)
    m = np.clip(m, 0, 1)[..., None]
    safe = _restore_gaussian(clean, original, sigma_safe)
    sharp = _restore_gaussian(clean, original, sigma_edge)
    return (m * sharp + (1 - m) * safe).clip(0, 255)


def _restore(clean_pil, original_pil, *, sigma: float, mode: str = "gaussian", unsharp_strength: float = 0.0):
    import cv2
    import numpy as np
    from PIL import Image

    clean = np.asarray(clean_pil.convert("RGB"))
    original = np.asarray(original_pil.convert("RGB"))
    if clean.shape != original.shape:
        clean = cv2.resize(clean, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    if mode == "gaussian":
        combined = _restore_gaussian(clean, original, sigma)
    elif mode == "edge":
        combined = _restore_edge(clean, original, sigma)
    else:
        raise ValueError(f"unknown restore mode: {mode}")
    combined = combined.astype("float32")
    if unsharp_strength > 0:
        blurred = cv2.GaussianBlur(combined, (0, 0), sigmaX=1.0, sigmaY=1.0)
        combined = (combined + unsharp_strength * (combined - blurred)).clip(0, 255)
    return Image.fromarray(combined.astype("uint8"))


def _sample(pipe, image, embeds, *, denoise: float, steps: int, seed: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return pipe(
        image=image,
        prompt_embeds=embeds["prompt_embeds"].unsqueeze(0),
        prompt_embeds_mask=embeds["prompt_embeds_mask"].unsqueeze(0),
        negative_prompt_embeds=embeds["negative_prompt_embeds"].unsqueeze(0),
        negative_prompt_embeds_mask=embeds["negative_prompt_embeds_mask"].unsqueeze(0),
        num_inference_steps=steps,
        true_cfg_scale=CFG,
        strength=denoise,
        generator=generator,
    ).images[0]


def run_once(pipe, image, embeds, *, denoise: float, steps: int, seed: int, passes: int):
    current = image
    for i in range(passes):
        # Reseed per pass so iterations aren't identical perturbations.
        current = _sample(pipe, current, embeds, denoise=denoise, steps=steps, seed=seed + i)
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="input image path")
    parser.add_argument("--denoise", type=float, nargs="+", default=[DEFAULT_DENOISE], help="denoise strength(s)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="sampler steps")
    parser.add_argument("--passes", type=int, default=1, help="sequential img2img passes")
    parser.add_argument("--seed", type=int, default=None, help="random per run if omitted")
    parser.add_argument("--restore-sigma", type=float, default=DEFAULT_RESTORE_SIGMA, help="freq-restore sigma; cliff at 2.0")
    parser.add_argument("--restore-mode", choices=["gaussian", "edge"], default="gaussian", help="restore recipe")
    parser.add_argument("--unsharp", type=float, default=0.0, help="post-restore sharpening; try 0.2 for perceptual lift")
    parser.add_argument("--no-restore", dest="restore", action="store_false", help="skip frequency restore")
    parser.add_argument("--keep-intermediate", action="store_true", help="also save the pre-restore output")
    parser.add_argument("--transformer", type=Path, default=GGUF_TRANSFORMER, help="alt GGUF transformer path")
    parser.set_defaults(restore=True)
    args = parser.parse_args()

    if args.seed is None:
        import secrets
        args.seed = secrets.randbits(63)
        print(f"seed: {args.seed} (random)")
    else:
        print(f"seed: {args.seed} (fixed)")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"missing {input_path}")

    image = load_image(str(input_path))
    print(f"input: {input_path.name}  size={image.size}")

    embeds = torch.load(EMBEDS_CACHE, map_location="cpu", weights_only=False)
    print(f"embeds: pos={tuple(embeds['prompt_embeds'].shape)} neg={tuple(embeds['negative_prompt_embeds'].shape)}")

    # Build the fallback sequence to attempt.
    # If --transformer was given explicitly, respect it and skip auto-fallback.
    explicit_transformer = args.transformer != GGUF_TRANSFORMER
    if explicit_transformer:
        attempt_chain = [(args.transformer.stem, args.transformer)]
    else:
        # Walk GGUF_FALLBACK_CHAIN; skip entries whose file isn't on disk yet.
        attempt_chain = []
        for label, cands in GGUF_FALLBACK_CHAIN:
            p = _resolve_gguf(cands)
            if p is not None:
                attempt_chain.append((label, p))
        if not attempt_chain:
            raise SystemExit(
                "No GGUF transformer found. Download at least one quant:\n"
                "  Qwen-Image-2512-Q4_K_M.gguf  (~13 GB, best quality)\n"
                "  Qwen-Image-2512-Q2_K_M.gguf  (~7 GB,  fallback)\n"
                "from https://huggingface.co/Frederic75/Qwen-Image-2512-GGUF"
            )

    OUT_DIR.mkdir(exist_ok=True)
    stem = input_path.stem

    pipe = None
    used_label = None
    used_path = None

    for label, transformer_path in attempt_chain:
        if pipe is not None:
            _free_pipeline(pipe)
            pipe = None

        print(f"[quant] trying {label}  ({transformer_path.name})")
        try:
            pipe = build_pipeline(transformer_path=transformer_path)
            # Run a single denoise pass to confirm the quant fits in memory
            # before committing to the full sweep.
            _ = run_once(
                pipe, image, embeds,
                denoise=args.denoise[0], steps=args.steps,
                seed=args.seed, passes=args.passes,
            )
            used_label = label
            used_path = transformer_path
            break   # success — keep this pipe for remaining denoise values
        except BaseException as exc:
            if _is_oom(exc) and not explicit_transformer:
                next_entries = attempt_chain[attempt_chain.index((label, transformer_path)) + 1:]
                if next_entries:
                    print(f"[OOM] {label} ran out of memory — retrying with {next_entries[0][0]}")
                    continue
            raise   # non-OOM error or no fallback left — propagate

    if pipe is None or used_path is None:
        raise SystemExit("All quants exhausted without success.")

    # Include the transformer quant tag when it differs from the best available
    # (i.e. we fell back, or --transformer was explicit).
    default_path = attempt_chain[0][1] if attempt_chain else used_path
    model_tag = "" if used_path == default_path else f"_{used_path.stem}"

    # The first denoise value was already run above to probe memory; reuse it.
    denoise_queue = list(args.denoise)
    first_denoise = denoise_queue.pop(0)

    def _save(clean_pil, denoise):
        tag = f"_s{args.steps}_d{denoise:.3f}_p{args.passes}{model_tag}"
        if args.restore:
            final_pil = _restore(
                clean_pil, image,
                sigma=args.restore_sigma,
                mode=args.restore_mode,
                unsharp_strength=args.unsharp,
            )
            mode_tag = "" if args.restore_mode == "gaussian" else f"_{args.restore_mode}"
            final_path = OUT_DIR / f"{stem}_desynth{tag}_r{args.restore_sigma:g}{mode_tag}.png"
            final_pil.save(final_path)
            print(f"  -> {final_path.relative_to(ROOT)}")
            if args.keep_intermediate:
                mid_path = OUT_DIR / f"{stem}_desynth{tag}.png"
                clean_pil.save(mid_path)
                print(f"  -> {mid_path.relative_to(ROOT)}  (intermediate)")
        else:
            final_path = OUT_DIR / f"{stem}_desynth{tag}.png"
            clean_pil.save(final_path)
            print(f"  -> {final_path.relative_to(ROOT)}")

    # The probe run result was discarded (we only used it to check for OOM).
    # Re-run the first denoise value properly and save, then handle the rest.
    for denoise in [first_denoise] + denoise_queue:
        clean_pil = run_once(
            pipe, image, embeds,
            denoise=denoise, steps=args.steps,
            seed=args.seed, passes=args.passes,
        )
        _save(clean_pil, denoise)


if __name__ == "__main__":
    main()
