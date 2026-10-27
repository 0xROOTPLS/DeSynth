# Qwen-Image 2.1 GGUF img2img + frequency-domain detail restore. Removes SynthID
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
            # BF16 is native to torch. Left as a GGUFParameter it stays raw bytes
            # in non-Linear modules (2.1's txt_in.text_norm), which never dequantize.
            if qt == _gguf.GGMLQuantizationType.BF16:
                parsed[t.name] = torch.from_numpy(t.data).view(torch.bfloat16)
                continue
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

import numpy as np
from diffusers import (
    AutoencoderKLQwenImage21,
    GGUFQuantizationConfig,
    QwenImage21Pipeline,
    QwenImage21Transformer2DModel,
)
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions, calculate_shift
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor

ROOT = Path(__file__).parent
DEFAULT_INPUT = ROOT / "Original.png"
OUT_DIR = ROOT / "out"

# Ordered list of GGUF quants to try, best quality first.
# Each entry is (label, [candidate paths]); first existing path wins per entry.
# On OOM the pipeline tears down and retries with the next entry automatically.
GGUF_FALLBACK_CHAIN: list[tuple[str, list[Path]]] = [
    ("Q8_0 (~7.6 GB)", [
        ROOT / "qwen-image-2.1-Q8_0.gguf",
        ROOT / "Qwen-Image-2.1-Q8_0.gguf",
    ]),
    ("Q4_K_M (~4.2 GB)", [
        ROOT / "qwen-image-2.1-Q4_K_M.gguf",
        ROOT / "Qwen-Image-2.1-Q4_K_M.gguf",
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
# lightx2v/Qwen-Image-2.1-Lightning
LIGHTNING_LORA = ROOT / "pytorch_lora_weights.safetensors"
EMBEDS_CACHE = ROOT / "embeds_cache.pt"
# Width of Qwen-Image 2.1's Qwen3-VL text embeddings (the old Qwen2.5-VL cache is 3584).
TEXT_EMBED_DIM = 4096


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

# HF repo used only for tiny configs, the ~16MB processor and the ~1.35GB VAE.
QWEN_IMAGE_REPO = "Qwen/Qwen-Image-2.1"
# Default sampling area (RESOLUTION² pixels). Aspect ratio follows the input.
# Above 2.1's native ~1MP keeps more mid-band detail in the clean low band.
RESOLUTION = 1280
# Lightning LoRA strength; off by default. With the same 5 steps, the base
# model's sample restores ~0.75 dB closer to the original. 0.8 if you use it.
LIGHTNING_WEIGHT = 0.0

# 32 steps so denoise quantizes to 1/32. 5/32 (start sigma ~0.26 at 1280 after
# the scheduler's shift) is the lowest tested that defeats SynthID; 3/32 and
# 4/32 left it detectable. Higher = more drift, no extra benefit.
DEFAULT_STEPS = 32
CFG = 1.0
DEFAULT_DENOISE = 0.15625
# Restore cutoff per mode, as a Gaussian sigma (butterworth: the Gaussian with
# the same half-gain point). At the defaults above, SynthID returns between
# butterworth-12 2.6 (pass) and 2.8 (fail), and gaussian 0.9 / 1.0. Each keeps
# a step of margin: the old 1.95 sat at the edge and broke when SynthID changed.
DEFAULT_RESTORE_MODE = "butterworth"
DEFAULT_RESTORE_ORDER = 12
DEFAULT_RESTORE_SIGMA = {"butterworth": 2.4, "gaussian": 0.8, "edge": 0.8}

# ---------------------------------------------------------------------------
# Device / dtype selection: CUDA > MPS (Apple Silicon) > CPU
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = "cuda"
    COMPUTE_DTYPE = torch.bfloat16   # native on Ampere+
elif torch.backends.mps.is_available():
    DEVICE = "mps"
    # PyTorch 2.14 has solid bfloat16 MPS coverage. More importantly, the
    # diffusers pipeline's image preprocessor naturally produces bfloat16
    # tensors on MPS; using float16 here causes a dtype mismatch in the
    # VAE encoder (Input bfloat16 vs bias float16). Keep bfloat16 throughout.
    COMPUTE_DTYPE = torch.bfloat16
    # Disable PyTorch's conservative MPS OOM guard (~70% threshold) so macOS's
    # own memory compressor and SSD swap can manage pressure instead.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
else:
    DEVICE = "cpu"
    COMPUTE_DTYPE = torch.float32    # bf16/fp16 are slow on CPU without HW support

print(f"[device] {DEVICE}  dtype={COMPUTE_DTYPE}")


def build_pipeline(
    transformer_path: Path = GGUF_TRANSFORMER, lightning_weight: float = LIGHTNING_WEIGHT
) -> QwenImage21Pipeline:
    required = (transformer_path, EMBEDS_CACHE) + ((LIGHTNING_LORA,) if lightning_weight else ())
    for p in required:
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
            print("(cache miss, fetching configs/VAE from HF, one-time)")
            return loader(**kwargs, local_files_only=False)

    transformer = _load(
        QwenImage21Transformer2DModel.from_single_file,
        pretrained_model_link_or_path_or_dict=str(transformer_path),
        quantization_config=GGUFQuantizationConfig(compute_dtype=dtype),
        dtype=dtype,
        config=QWEN_IMAGE_REPO,
        subfolder="transformer",
    )

    vae = _load(
        AutoencoderKLQwenImage21.from_pretrained,
        pretrained_model_name_or_path=QWEN_IMAGE_REPO,
        subfolder="vae",
        dtype=dtype,
    )

    # No text encoder loaded; embeds are precomputed. The small processor
    # still loads: the pipeline's __init__ reads its chat template.
    pipe = _load(
        QwenImage21Pipeline.from_pretrained,
        pretrained_model_name_or_path=QWEN_IMAGE_REPO,
        transformer=transformer,
        vae=vae,
        text_encoder=None,
        dtype=dtype,
    )

    # Load LoRA from a state dict rather than the path: the file's embedded
    # PEFT metadata names LoraConfig fields that peft<0.20 rejects. Rank and
    # alpha (16/16) are inferred from the tensors either way. Weight 0 skips it
    # and samples with the base model (give it more --steps).
    if lightning_weight:
        try:
            from safetensors.torch import load_file as _st_load
            _lora_sd = _st_load(str(LIGHTNING_LORA), device="cpu")
            pipe.load_lora_weights(_lora_sd, adapter_name="lightning")
        except Exception:
            # Fallback: let diffusers handle it normally (heap allocation).
            pipe.load_lora_weights(str(LIGHTNING_LORA), adapter_name="lightning")
        pipe.set_adapters(["lightning"], adapter_weights=[lightning_weight])

    # Offload strategy depends on available hardware.
    #   CUDA:     sequential offload, moves one layer at a time, fits the
    #             7.6GB Q8_0 GGUF plus activations in 8 GB VRAM.
    #   MPS/CPU:  model-level offload; accelerate moves whole sub-modules;
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
            # Older accelerate doesn't accept 'device' kwarg; move manually.
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
# more original detail back; in flats, sigma_safe keeps the watermark dead.
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


# Butterworth: the same recombination as gaussian, with a sharper crossover
# H(f) = 1 / (1 + (f/fc)^(2n)). The Gaussian's soft skirt still passes a share
# of the original below its cutoff (16% at half the cutoff frequency), which
# carries the watermark back; n=4 passes 0.4% and n=12 next to nothing, so the
# cutoff can sit higher. Past n=12 the SynthID limit stopped moving.
# sigma is the Gaussian with the same half-gain point (fc = 0.1874/sigma px^-1).
def _restore_butterworth(clean, original, sigma, order=DEFAULT_RESTORE_ORDER, pad=64):
    import numpy as np
    original = original.astype("float32")
    diff = clean.astype("float32") - original
    # Reflect-pad so the FFT's wraparound doesn't seam opposite borders together.
    diff = np.pad(diff, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
    h, w = diff.shape[:2]
    f = np.hypot(np.fft.fftfreq(h)[:, None], np.fft.rfftfreq(w)[None, :])
    fc = np.sqrt(np.log(2) / 2) / (np.pi * sigma)
    gain = 1.0 / (1.0 + (f / fc) ** (2 * order))
    low = np.fft.irfft2(np.fft.rfft2(diff, axes=(0, 1)) * gain[..., None], s=(h, w), axes=(0, 1))
    return (original + low[pad:-pad, pad:-pad]).clip(0, 255)


def _restore(clean_pil, original_pil, *, sigma: float, mode: str = "gaussian", order: int = DEFAULT_RESTORE_ORDER, unsharp_strength: float = 0.0):
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
    elif mode == "butterworth":
        combined = _restore_butterworth(clean, original, sigma, order)
    else:
        raise ValueError(f"unknown restore mode: {mode}")
    combined = combined.astype("float32")
    if unsharp_strength > 0:
        blurred = cv2.GaussianBlur(combined, (0, 0), sigmaX=1.0, sigmaY=1.0)
        combined = (combined + unsharp_strength * (combined - blurred)).clip(0, 255)
    return Image.fromarray(combined.astype("uint8"))


# QwenImage21Pipeline has no img2img mode (its `image=` is edit-style context),
# so do it here: VAE-encode the input, noise it to the first sigma of the
# schedule's tail, and let the pipeline denoise from there. The tail is cut the
# way diffusers' img2img get_timesteps() cuts it, so `denoise` keeps its old
# meaning. The scheduler shifts each sigma independently, so the tail comes out
# identical to the end of the full schedule. no_grad because the VAE encode
# runs outside the pipeline's own: with autograd on, the encoder's activations
# (7+ GB) are kept alive through the whole denoise and spill out of VRAM.
@torch.no_grad()
def _sample(pipe, image, embeds, *, denoise: float, steps: int, seed: int, resolution: int):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    device = pipe._execution_device
    # Unlike the old pipeline, 2.1 leaves caller-supplied embeds where they are.
    embeds = {k: v.unsqueeze(0).to(device) for k, v in embeds.items() if torch.is_tensor(v)}
    dtype = embeds["prompt_embeds"].dtype

    t_start = int(max(steps - min(steps * denoise, steps), 0))
    sigmas = np.linspace(1.0, 1 / steps, steps)[t_start:]
    if len(sigmas) == 0:
        raise ValueError(f"denoise {denoise} at {steps} steps leaves no sampler step")

    width, height, _ = calculate_dimensions(resolution * resolution, image.width / image.height)
    # The 2.1 VAE takes RGBA, as in the pipeline's own condition-image path.
    pixels = pipe.image_processor.preprocess(image.convert("RGBA"), height=height, width=width)
    image_latents = pipe._encode_vae_image(pixels.unsqueeze(2).to(device, pipe.vae.dtype), generator).to(dtype)
    _, channels, _, lat_h, lat_w = image_latents.shape
    image_latents = pipe._pack_latents(image_latents, 1, channels, lat_h, lat_w)

    sched = pipe.scheduler.config
    mu = calculate_shift(
        image_latents.shape[1],
        sched.get("base_image_seq_len", 256),
        sched.get("max_image_seq_len", 4096),
        sched.get("base_shift", 0.5),
        sched.get("max_shift", 1.15),
    )
    pipe.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
    sigma = pipe.scheduler.sigmas[0].item()
    # The real noise level: the scheduler's shift makes it differ from `denoise`.
    print(f"  [sample] {width}x{height}  start sigma {sigma:.3f}  {len(sigmas)} step(s)")
    noise = randn_tensor(image_latents.shape, generator=generator, device=device, dtype=dtype)
    latents = sigma * noise + (1.0 - sigma) * image_latents

    return pipe(
        prompt_embeds=embeds["prompt_embeds"],
        prompt_embeds_mask=embeds["prompt_embeds_mask"],
        negative_prompt_embeds=embeds["negative_prompt_embeds"],
        negative_prompt_embeds_mask=embeds["negative_prompt_embeds_mask"],
        latents=latents,
        sigmas=sigmas.tolist(),
        height=height,
        width=width,
        true_cfg_scale=CFG,
        use_kv_cache=False,
        generator=generator,
    ).images[0].convert("RGB")


def run_once(pipe, image, embeds, *, denoise: float, steps: int, seed: int, passes: int, resolution: int):
    current = image
    for i in range(passes):
        # Reseed per pass so iterations aren't identical perturbations.
        current = _sample(
            pipe, current, embeds, denoise=denoise, steps=steps, seed=seed + i, resolution=resolution
        )
    return current


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT), help="input image path")
    parser.add_argument("--denoise", type=float, nargs="+", default=[DEFAULT_DENOISE], help="denoise strength(s)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="sampler steps")
    parser.add_argument("--passes", type=int, default=1, help="sequential img2img passes")
    parser.add_argument("--seed", type=int, default=None, help="random per run if omitted")
    parser.add_argument("--restore-sigma", type=float, nargs="+", default=None, help="freq-restore sigma(s); default depends on --restore-mode")
    parser.add_argument("--restore-mode", choices=["gaussian", "edge", "butterworth"], default=DEFAULT_RESTORE_MODE, help="restore recipe")
    parser.add_argument("--restore-order", type=int, default=DEFAULT_RESTORE_ORDER, help="butterworth crossover steepness; higher = sharper")
    parser.add_argument("--unsharp", type=float, default=0.0, help="post-restore sharpening; try 0.2 for perceptual lift")
    parser.add_argument("--no-restore", dest="restore", action="store_false", help="skip frequency restore")
    parser.add_argument("--keep-intermediate", action="store_true", help="also save the pre-restore output")
    parser.add_argument("--transformer", type=Path, default=GGUF_TRANSFORMER, help="alt GGUF transformer path")
    parser.add_argument("--resolution", type=int, default=RESOLUTION, help="sampling area as N*N px; aspect follows input")
    parser.add_argument("--lightning-weight", type=float, default=LIGHTNING_WEIGHT, help="Lightning LoRA strength; 0 = off (use more --steps)")
    parser.set_defaults(restore=True)
    args = parser.parse_args()
    if args.restore_sigma is None:
        args.restore_sigma = [DEFAULT_RESTORE_SIGMA[args.restore_mode]]

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
    if embeds["prompt_embeds"].shape[-1] != TEXT_EMBED_DIM:
        raise SystemExit(
            f"{EMBEDS_CACHE.name} is from the old Qwen-Image text encoder "
            f"({embeds['prompt_embeds'].shape[-1]}-dim, need {TEXT_EMBED_DIM}). "
            "Run `python precompute_embeds.py` to rebuild it for Qwen-Image 2.1."
        )

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
                "  qwen-image-2.1-Q8_0.gguf    (~7.6 GB, best quality)\n"
                "  qwen-image-2.1-Q4_K_M.gguf  (~4.2 GB, fallback)\n"
                "from https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF"
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
            pipe = build_pipeline(transformer_path=transformer_path, lightning_weight=args.lightning_weight)
            # Run a single denoise pass to confirm the quant fits in memory
            # before committing to the full sweep.
            first_clean = run_once(
                pipe, image, embeds,
                denoise=args.denoise[0], steps=args.steps,
                seed=args.seed, passes=args.passes, resolution=args.resolution,
            )
            used_label = label
            used_path = transformer_path
            break   # success, keep this pipe for remaining denoise values
        except BaseException as exc:
            if _is_oom(exc) and not explicit_transformer:
                next_entries = attempt_chain[attempt_chain.index((label, transformer_path)) + 1:]
                if next_entries:
                    print(f"[OOM] {label} ran out of memory, retrying with {next_entries[0][0]}")
                    continue
            raise   # non-OOM error or no fallback left, propagate

    if pipe is None or used_path is None:
        raise SystemExit("All quants exhausted without success.")

    # Include the transformer quant tag when it differs from the best available
    # (i.e. we fell back, or --transformer was explicit).
    default_path = attempt_chain[0][1] if attempt_chain else used_path
    model_tag = "" if used_path == default_path else f"_{used_path.stem}"
    # Non-default sampling settings go in the name so sweeps don't overwrite each other.
    if args.resolution != RESOLUTION:
        model_tag += f"_res{args.resolution}"
    if args.lightning_weight != LIGHTNING_WEIGHT:
        model_tag += f"_lw{args.lightning_weight:g}"

    # The first denoise value was already run above to probe memory; reuse it.
    denoise_queue = list(args.denoise)
    first_denoise = denoise_queue.pop(0)

    def _save(clean_pil, denoise):
        tag = f"_s{args.steps}_d{denoise:.3f}_p{args.passes}{model_tag}"
        if args.restore:
            mode_tag = {"butterworth": f"_bw{args.restore_order}"}.get(args.restore_mode, f"_{args.restore_mode}")
            # Restore is cheap post-processing, so every sigma reuses this one sample.
            for sigma in args.restore_sigma:
                final_pil = _restore(
                    clean_pil, image,
                    sigma=sigma,
                    mode=args.restore_mode,
                    order=args.restore_order,
                    unsharp_strength=args.unsharp,
                )
                final_path = OUT_DIR / f"{stem}_desynth{tag}_r{sigma:g}{mode_tag}.png"
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

    # Sampling is seeded per call, so the probe is the first denoise's result.
    _save(first_clean, first_denoise)
    for denoise in denoise_queue:
        clean_pil = run_once(
            pipe, image, embeds,
            denoise=denoise, steps=args.steps,
            seed=args.seed, passes=args.passes, resolution=args.resolution,
        )
        _save(clean_pil, denoise)


if __name__ == "__main__":
    main()
