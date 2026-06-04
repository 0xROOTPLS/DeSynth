# Quantitative similarity metrics between two images. No model downloads —
# pure pixel-level + structural metrics from numpy/skimage/opencv.

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

ROOT = Path(__file__).parent


def load(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"can't read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def metrics(a: np.ndarray, b: np.ndarray) -> dict:
    af = a.astype(np.float64)
    bf = b.astype(np.float64)
    diff = af - bf

    mse = float((diff ** 2).mean())
    mae = float(np.abs(diff).mean())
    psnr = float(10 * np.log10((255.0 ** 2) / mse)) if mse > 0 else float("inf")

    # Structural similarity, computed per channel + averaged.
    ssim_score, _ = ssim(a, b, channel_axis=2, full=True, data_range=255)

    # Histogram correlation per channel (Pearson).
    hist_corr = []
    for c in range(3):
        ha = cv2.calcHist([a], [c], None, [256], [0, 256]).flatten()
        hb = cv2.calcHist([b], [c], None, [256], [0, 256]).flatten()
        hist_corr.append(float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)))

    # Frequency-band agreement: split at sigma=1.95 (the pipeline cutoff) and
    # measure SSIM in each band separately. Tells us where similarity lives.
    sigma = 1.95
    low_a = cv2.GaussianBlur(af.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    low_b = cv2.GaussianBlur(bf.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)
    high_a = af.astype(np.float32) - low_a
    high_b = bf.astype(np.float32) - low_b
    ssim_low = float(ssim(low_a.clip(0, 255).astype(np.uint8),
                          low_b.clip(0, 255).astype(np.uint8),
                          channel_axis=2, data_range=255))
    # High-pass is centered around 0 with small range — shift to [0,255] for SSIM.
    ha = (high_a + 128).clip(0, 255).astype(np.uint8)
    hb = (high_b + 128).clip(0, 255).astype(np.uint8)
    ssim_high = float(ssim(ha, hb, channel_axis=2, data_range=255))

    return {
        "shape": a.shape,
        "mse": mse,
        "mae": mae,
        "psnr_db": psnr,
        "ssim": float(ssim_score),
        "ssim_low_pass": ssim_low,
        "ssim_high_pass": ssim_high,
        "hist_corr_rgb": hist_corr,
        "mean_hist_corr": float(np.mean(hist_corr)),
    }


def _draw_visual(a: np.ndarray, b: np.ndarray, m: dict, out_path: Path) -> None:
    # Side-by-side, normalized height, with a header strip carrying the stats.
    target_h = 720
    def fit(img):
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * target_h / h), target_h), interpolation=cv2.INTER_AREA)
    A, B = fit(a), fit(b)
    gap = 16
    label_h = 36
    stats_h = 84
    pad = 24
    canvas_w = A.shape[1] + B.shape[1] + gap + pad * 2
    canvas_h = label_h + target_h + stats_h + pad * 2
    canvas = np.full((canvas_h, canvas_w, 3), 24, dtype=np.uint8)

    # cv2 wants BGR; we've been working in RGB.
    A_bgr = cv2.cvtColor(A, cv2.COLOR_RGB2BGR)
    B_bgr = cv2.cvtColor(B, cv2.COLOR_RGB2BGR)
    y0 = pad + label_h
    canvas[y0:y0 + target_h, pad:pad + A.shape[1]] = A_bgr
    canvas[y0:y0 + target_h, pad + A.shape[1] + gap:pad + A.shape[1] + gap + B.shape[1]] = B_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "reference", (pad, pad + 24), font, 0.7, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "desynth output", (pad + A.shape[1] + gap, pad + 24), font, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

    stats_y = y0 + target_h + 28
    line1 = f"PSNR {m['psnr_db']:.2f} dB   SSIM {m['ssim']:.4f}   MAE {m['mae']:.2f}"
    line2 = f"SSIM low {m['ssim_low_pass']:.4f}   SSIM high {m['ssim_high_pass']:.4f}   hist corr {m['mean_hist_corr']:.4f}"
    cv2.putText(canvas, line1, (pad, stats_y), font, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, line2, (pad, stats_y + 28), font, 0.6, (180, 200, 220), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)
    print(f"  -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("a", help="reference image")
    parser.add_argument("b", help="comparison image")
    parser.add_argument("--visual", action="store_true", help="save side-by-side PNG")
    parser.add_argument("--visual-out", type=Path, default=None, help="output path for --visual")
    args = parser.parse_args()

    a = load(Path(args.a))
    b = load(Path(args.b))
    if a.shape != b.shape:
        print(f"resizing b from {b.shape} to {a.shape}")
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LANCZOS4)

    m = metrics(a, b)
    print(f"  shape:           {m['shape']}")
    print(f"  PSNR:            {m['psnr_db']:.2f} dB        (higher = closer; >40 typically imperceptible)")
    print(f"  SSIM:            {m['ssim']:.4f}            (1.0 = identical)")
    print(f"   - low-pass:     {m['ssim_low_pass']:.4f}            (broad composition agreement)")
    print(f"   - high-pass:    {m['ssim_high_pass']:.4f}            (fine-detail agreement)")
    print(f"  MAE:             {m['mae']:.3f}             (mean abs pixel diff, 0-255 scale)")
    print(f"  MSE:             {m['mse']:.3f}")
    print(f"  hist correl R/G/B: {m['hist_corr_rgb'][0]:.4f} / {m['hist_corr_rgb'][1]:.4f} / {m['hist_corr_rgb'][2]:.4f}")

    if args.visual:
        out = args.visual_out or (ROOT / "out" / f"comparison_{Path(args.b).stem}.png")
        out.parent.mkdir(exist_ok=True)
        _draw_visual(a, b, m, out)


if __name__ == "__main__":
    main()
