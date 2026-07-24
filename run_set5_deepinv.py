"""Standalone: run deepinv's DRUNet (KAIR weights) end-to-end on Set5.

Everything (model, data, metrics) goes through deepinv:
  * model   : deepinv.models.DRUNet + KAIR's drunet_color.pth
  * data    : deepinv.datasets.ImageFolder -- built inside eval_workflow() with
              per-sigma noise baked in via _make_pair_transform(); evaluate()
              itself never constructs a dataset.
  * metrics : deepinv.loss.metric PSNR/SSIM -- evaluate() takes the metric list
              as an argument, so swapping in others (LPIPS, FSIM, ...) is just
              appending to scoring_metrics().

The run is organized around the GOAL, not the check:
  * eval_workflow()  -- THE PURPOSE: load model, build per-sigma datasets,
                       evaluate(), print_scores(). Holds NO reference values and
                       is complete on its own.
  * COMPARISON      -- run_comparison() + build_benchmark(): a verification step,
                       NOT the goal. The only place that knows reference values /
                       tolerances; it reuses eval_workflow's scores to confirm we
                       are not worse than KAIR's benchmark.

Noise uses KAIR's benchmark convention (clean/255 + seeded AWGN, sigma/255,
not clipped).

Usage:
    run.bat            (sets the env paths, then runs the KAIR venv python)
"""

import os
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from utils.utils_image import uint2single, single2tensor4, imread_uint   # vendored from KAIR/utils

# --- paths come from the environment (set by run.bat) -----------------------
# Run via run.bat. Running this file directly requires SHIM_ROOT,
# DRUNET_WEIGHTS and TESTSET_DIR to be set in the environment.
SHIM_ROOT = os.environ["SHIM_ROOT"]
DRUNET_WEIGHTS = os.environ["DRUNET_WEIGHTS"]
TESTSET_DIR = os.environ["TESTSET_DIR"]

SIGMAS = [15, 25, 50]
BORDER = 1                       # KAIR denoise convention (shave 1px border)
TOL_DB, TOL_SSIM = 0.05, 0.001   # "no lower" tolerance (comparison layer only)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Put the shim in front of sys.path so `deepinv` resolves to the minimal shim.
sys.path.insert(0, SHIM_ROOT)


# ---------------------------------------------------------------------------
# data: deepinv ImageFolder (loader=path) -> _make_pair_transform reads via
#       vendored imread_uint + uint2single -> (clean, noisy) HWC f32 [0,1]
# ---------------------------------------------------------------------------


def _make_pair_transform(sigma):
    """Transform (path -> (clean, noisy)) numpy HWC float32 [0,1].

    Reads with vendored imread_uint (cv2, BGR->RGB) and scales via uint2single.
    """
    def tx(path):
        clean = uint2single(imread_uint(path))              # HWC float [0,1]
        np.random.seed(0)                                       # KAIR benchmark noise
        noise = np.random.normal(0, sigma / 255.0, clean.shape).astype(np.float32)
        noisy = (clean + noise).astype(np.float32)
        return clean, noisy
    return tx


# ---------------------------------------------------------------------------
# metrics (evaluation config; no references here)
# ---------------------------------------------------------------------------
@dataclass
class Metric:
    """A scoring metric: just a name + fn(out,gt)->float. No reference data."""
    name: str
    fn: Callable[[torch.Tensor, torch.Tensor], float]


def build_metric_fns(border):
    """Return dict name -> deepinv.loss.metric fn (border shaved via center_crop)."""
    from deepinv.loss.metric.distortion import PSNR, SSIM
    crop = -border if border else None
    return {
        "PSNR": PSNR(max_pixel=1.0, center_crop=crop, reduction="none"),
        "SSIM": SSIM(max_pixel=1.0, center_crop=crop, reduction="none"),
    }


def scoring_metrics():
    """The set of metrics to compute -- pure evaluation config, no reference
    values. Swapping/adding metrics (LPIPS, FSIM, ...) edits only this."""
    fns = build_metric_fns(BORDER)
    return [Metric("PSNR", fns["PSNR"]), Metric("SSIM", fns["SSIM"])]


# ---------------------------------------------------------------------------
# benchmark references (COMPARISON layer -- the ONLY place that knows refs)
# ---------------------------------------------------------------------------
@dataclass
class Benchmark:
    """Reference values + tolerance for one metric, used only by comparison."""
    ref: dict[int, float]       # sigma -> KAIR reference value
    tol: float = 0.0            # PASS when score - ref >= -tol


def build_benchmark():
    """ALL reference values / tolerances, isolated from the evaluation layer.

    This is the single source of "what we compare against". The evaluation
    path (eval_workflow / scoring_metrics / evaluate / print_scores) never
    imports or calls it, so the model can be scored without any benchmark at
    all. Swapping the target (e.g. BSD68) or adding a metric's reference means
    editing only this.
    """
    return {
        "PSNR": Benchmark(
            {15: 34.89824874853085, 25: 32.72472725850353, 50: 29.866565943629915},
            tol=TOL_DB),
        "SSIM": Benchmark(
            {15: 0.9263187717428659, 25: 0.8959376838157247, 50: 0.8450952369999241},
            tol=TOL_SSIM),
    }


# ---------------------------------------------------------------------------
# evaluate: dataset + metrics injected; pure scoring only (no refs at all)
# ---------------------------------------------------------------------------
def evaluate(model, dataset, metrics, sigma, device):
    """Score `model` on `dataset` (yields (clean,noisy) HWC f32 [0,1]) at level `sigma`.

    Returns scores = {metric.name: [per-image float]}.
    `sigma` is forwarded to the deepinv DRUNet; the dataset is responsible for
    producing the matching noisy input (e.g. via _make_pair_transform).
    Has zero knowledge of reference values.
    """
    scores = {m.name: [] for m in metrics}
    n = len(dataset)
    if n == 0:
        raise FileNotFoundError("Dataset is empty")
    for i in range(n):
        clean, noisy = dataset[i]                         # numpy (H,W,3) HWC [0,1]
        t = single2tensor4(noisy).to(device)              # (1,3,H,W)
        gt = single2tensor4(clean).to(device)             # (1,3,H,W)
        with torch.no_grad():
            out = model(t, sigma / 255.0).clamp(0, 1)     # deepinv pads + noise map
        for m in metrics:
            with torch.no_grad():
                scores[m.name].append(float(m.fn(out, gt)))
    return scores


def print_scores(scores, metrics, sigmas):
    """Print the raw per-sigma scores. No references, no comparison -- the
    output of eval_workflow alone."""
    head = ["sigma"] + [m.name for m in metrics]
    widths = [max(len(h), 8) for h in head]
    print(" | ".join(h.rjust(widths[i]) for i, h in enumerate(head)))
    print(" | ".join("-" * widths[i] for i in range(len(widths))))
    for sigma in sigmas:
        cells = [str(sigma).rjust(widths[0])]
        for j, m in enumerate(metrics):
            val = float(np.mean(scores[sigma][m.name]))
            cells.append(f"{val:.4f}".rjust(widths[1 + j]))
        print(" | ".join(cells))


# ---------------------------------------------------------------------------
# comparison: consumes eval_workflow()'s scores + build_benchmark()'s refs
# ---------------------------------------------------------------------------
def run_comparison(scores, metrics, benchmark, sigmas):
    """Compare the scores (already computed by evaluate) against references.

    This is the ONLY layer that uses references/tolerances. Returns True iff
    every (sigma, metric) check passes; does not re-run the model.
    """
    rows = []
    for sigma in sigmas:
        per_metric = {}
        sigma_ok = True
        for m in metrics:
            val = float(np.mean(scores[sigma][m.name]))
            b = benchmark[m.name]
            d = val - b.ref[sigma]
            ok = d >= -b.tol
            sigma_ok = sigma_ok and ok
            per_metric[m.name] = (val, d, ok)
        rows.append((sigma, per_metric, sigma_ok))

    head = (["sigma"] + [m.name for m in metrics]
            + [f"\u0394{m.name}" for m in metrics] + ["status"])
    widths = [max(len(h), 8) for h in head]
    print("\n" + " | ".join(h.rjust(widths[i]) for i, h in enumerate(head)))
    print(" | ".join("-" * widths[i] for i in range(len(widths))))

    all_pass = True
    for sigma, per_metric, sigma_ok in rows:
        all_pass = all_pass and sigma_ok
        cells = [str(sigma).rjust(widths[0])]
        for j, m in enumerate(metrics):
            val = per_metric[m.name][0]
            cells.append(f"{val:.4f}".rjust(widths[1 + j]))
        for j, m in enumerate(metrics):
            d = per_metric[m.name][1]
            cells.append(f"{d:+.4f}".rjust(widths[1 + len(metrics) + j]))
        cells.append(("PASS" if sigma_ok else "FAIL").rjust(widths[-1]))
        print(" | ".join(cells))

    print(f"\nResult: {'ALL PASS' if all_pass else 'SOME FAIL'} "
          f"(tol {TOL_DB} dB / {TOL_SSIM} SSIM)")
    return all_pass


# ---------------------------------------------------------------------------
# eval_workflow: THE GOAL -- end-to-end scoring, no reference values at all
# ---------------------------------------------------------------------------
def eval_workflow():
    """Run deepinv DRUNet end-to-end on Set5 and score it -- this IS the point.

    Loads the model, builds one ImageFolder per sigma (noise baked in via the
    transform), evaluates, and prints the raw scores. Holds NO reference values
    whatsoever: comparison against KAIR's benchmark is a separate verification
    step (run_comparison), not the purpose of this workflow.
    Returns (scores, metrics).
    """
    print(f"device        : {DEVICE}")
    print(f"weights       : {DRUNET_WEIGHTS}")
    print(f"testset       : {TESTSET_DIR}")
    print(f"sigmas        : {SIGMAS}")
    print(f"shim (deepinv): {SHIM_ROOT}\n")

    if not os.path.isfile(DRUNET_WEIGHTS):
        raise FileNotFoundError(f"Weights not found: {DRUNET_WEIGHTS}")
    if not os.path.isdir(TESTSET_DIR):
        raise FileNotFoundError(f"Testset dir not found: {TESTSET_DIR}")

    from deepinv.models import DRUNet
    from deepinv.datasets import ImageFolder

    model = DRUNet(
        in_channels=3, out_channels=3, nc=(64, 128, 256, 512), nb=4,
        act_mode="R", downsample_mode="strideconv",
        upsample_mode="convtranspose", pretrained=DRUNET_WEIGHTS,
    )
    model.to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    metrics = scoring_metrics()

    # datasets are built here (one per sigma, noise baked into the transform);
    # evaluate() itself stays dataset-agnostic.
    # loader returns the path so the transform reads it via imread_uint.
    datasets = {s: ImageFolder(TESTSET_DIR, transform=_make_pair_transform(s),
                            loader=lambda p: p)
                for s in SIGMAS}
    n = len(datasets[SIGMAS[0]])
    print(f"Found {n} images in {TESTSET_DIR} (via deepinv.datasets.ImageFolder)\n")

    scores = {s: evaluate(model, datasets[s], metrics, s, DEVICE)
              for s in SIGMAS}
    print_scores(scores, metrics, SIGMAS)
    return scores, metrics


def main():
    # --- the goal: end-to-end evaluation (complete on its own) -------------
    scores, metrics = eval_workflow()

    # --- verification only: confirm we're not worse than KAIR's benchmark --
    # (this is a sanity check, not the purpose of the run)
    benchmark = build_benchmark()
    all_pass = run_comparison(scores, metrics, benchmark, SIGMAS)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
