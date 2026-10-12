"""Run ONE model over the test set and profile it. Run it inside that model's environment:

    python model_test/run_model.py --model aeroblade          # or trufor / forma / pscc
    python model_test/run_model.py --model pscc --device cpu --limit 5

Phases (each is timed, and RAM / GPU are sampled separately for each):
  load       build the model and load the weights
  warmup     the first --warmup images, run and discarded (CUDA context, cuDNN autotuning, caches)
  measure    every image once, batch size 1, CUDA-synchronized stage timings; peak PyTorch GPU memory
             per image; detection and localization metrics against the ground truth
  flops      FLOPs of one full prediction per image (torch.utils.flop_counter, PyTorch >= 2.1)
  profile    torch.profiler over --profile-images images (default 3, ForMa 1): per-operator CPU/GPU
             time, memory, FLOPs. Runs last, after the results above are saved.
Writes model_test/outputs/<model>/: per_image.csv, profile.json, gpu_samples.csv, torch_ops.csv/.txt,
trace.json.gz (chrome://tracing or ui.perfetto.dev), maps/ (small previews of the score maps).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bench import metrics as M  # noqa: E402
from bench import profiling as P  # noqa: E402

MODELS = {"aeroblade": "adapters.aeroblade", "trufor": "adapters.trufor", "forma": "adapters.forma",
          "pscc": "adapters.pscc"}


def pct(values, q):
    v = sorted(values)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def latency_stats(values):
    if not values:
        return None
    return {"mean": statistics.fmean(values) if hasattr(statistics, "fmean") else statistics.mean(values),
            "median": statistics.median(values), "p95": pct(values, 0.95), "min": min(values),
            "max": max(values), "std": statistics.pstdev(values)}


def save_preview(arr: np.ndarray, path: Path, max_side: int = 384) -> None:
    img = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    s = max_side / max(img.size)
    if s < 1:
        img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.BILINEAR)
    img.save(path)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--device", default="auto", help="auto (cuda if available), cuda, cuda:N or cpu")
    p.add_argument("--manifest", type=Path, default=HERE / "testset" / "manifest.csv")
    p.add_argument("--out", type=Path, help="default: model_test/outputs/<model>")
    p.add_argument("--limit", type=int, default=0, help="first N images only (0 = all)")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--profile-images", type=int, help="images for torch.profiler (default 3, ForMa 1)")
    p.add_argument("--flops-images", type=int, default=0, help="images for the FLOP pass (0 = all)")
    p.add_argument("--no-trace", action="store_true", help="skip the chrome trace file")
    p.add_argument("--threads", type=int, help="torch CPU threads")
    p.add_argument("--pscc-root", help="PSCC-Net folder (default: auto-detect)")
    p.add_argument("--pscc-threshold", type=float, default=0.5)
    p.add_argument("--forma-weights", help="ForMa checkpoint (default: ForMa/ForMa_weights/ForMa_weights.pth)")
    args = p.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit(f"--device {device}: this PyTorch ({torch.__version__}) sees no CUDA GPU")
    if args.threads:
        torch.set_num_threads(args.threads)
    on_gpu = device.startswith("cuda")
    gpu_index = int(device.split(":")[1]) if ":" in device else 0
    out = args.out or HERE / "outputs" / args.model
    (out / "maps").mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(args.manifest, newline="")))
    if args.limit:
        rows = rows[:args.limit]
    root = args.manifest.parent
    print(f"{args.model}: {len(rows)} images on {device} ({torch.__version__})", flush=True)

    t_origin = time.perf_counter()
    rss = P.RSSSampler()
    rss.start()
    gpu = P.GPUSampler(gpu_index) if on_gpu else None
    if gpu:
        gpu.start()
        time.sleep(0.5)
    phases = {}
    rss_before, _ = P.process_memory()

    # ---- load
    t0 = time.perf_counter()
    opts = {"pscc_root": args.pscc_root, "pscc_threshold": args.pscc_threshold, "forma_weights": args.forma_weights}
    model = importlib.import_module(MODELS[args.model]).Model(device, opts)
    P.sync(device)
    phases["load"] = (t0, time.perf_counter())
    params = {name: sum(int(t.numel()) for t in m.parameters()) for name, m in model.modules().items()}
    gpu_after_load = {"allocated_mib": torch.cuda.memory_allocated() / 2**20,
                      "reserved_mib": torch.cuda.memory_reserved() / 2**20} if on_gpu else None
    print(f"  loaded in {phases['load'][1] - t0:.1f} s, {sum(params.values()) / 1e6:.1f} M parameters", flush=True)

    # ---- warmup
    t0 = time.perf_counter()
    for r in rows[:args.warmup]:
        model.predict(root / r["file"], P.Stopwatch(device))
    P.sync(device)
    phases["warmup"] = (t0, time.perf_counter())

    # ---- measure
    if on_gpu:
        torch.cuda.reset_peak_memory_stats()
    per_image, t0 = [], time.perf_counter()
    for i, r in enumerate(rows):
        sw = P.Stopwatch(device)
        if on_gpu:
            torch.cuda.reset_peak_memory_stats()
        t_img = time.perf_counter()
        try:
            res = model.predict(root / r["file"], sw)
            P.sync(device)
            err = ""
        except Exception as e:  # noqa: BLE001 - one bad image must not end the run
            res, err = None, f"{type(e).__name__}: {e}"
            traceback.print_exc()
        total = time.perf_counter() - t_img
        row = {k: r[k] for k in ("image_id", "category", "generator", "mask_type", "label", "width", "height")}
        row.update(status="ok" if res else "error", error=err, t_total_s=total)
        row.update({f"t_{k}_s": v for k, v in sw.times.items()})
        if on_gpu:
            row["gpu_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            row["gpu_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        if res:
            row["score"] = res["score"]
            row["pred_label"] = int(res["score"] > model.det_threshold)
            stem = Path(r["file"]).stem
            save_preview(res["map"], out / "maps" / f"{r['image_id']}.png")
            if r["mask_file"]:
                gt_img = Image.open(root / r["mask_file"]).convert("L")
                h, w = res["map"].shape
                if gt_img.size != (w, h):
                    gt_img = gt_img.resize((w, h), Image.NEAREST)
                gt = np.asarray(gt_img) > 127
                row.update({f"loc_{k}": v for k, v in M.localization(res["map"], res["pred"], gt).items()})
            for k, v in (res.get("extra") or {}).items():
                if isinstance(v, (int, float, str, bool)):
                    row[k] = v
            row["map_h"], row["map_w"] = res["map"].shape
        per_image.append(row)
        print(f"  [{i + 1}/{len(rows)}] {r['image_id']:22s} {total * 1e3:8.1f} ms  "
              f"score {row.get('score', float('nan')):.4f}" + (f"  ERROR {err}" if err else ""), flush=True)
    phases["measure"] = (t0, time.perf_counter())
    ok = [r for r in per_image if r["status"] == "ok"]

    # ---- flops
    t0 = time.perf_counter()
    flop_rows = rows if not args.flops_images else rows[:args.flops_images]
    flops = []
    for r in flop_rows:
        res = P.count_flops(lambda rec: model.predict(root / r["file"], rec))
        flops.append(res)
        if res["flops"] is None:
            break
    phases["flops"] = (t0, time.perf_counter())
    flop_vals = [f["flops"] for f in flops if f["flops"] is not None]
    for row, f in zip(per_image, flops):
        row["flops"] = f["flops"]

    rss.stop()
    if gpu:
        time.sleep(0.3)
        gpu.stop()
        gpu.save(out / "gpu_samples.csv", t_origin)
    _, rss_peak_process = P.process_memory()

    # ---- summary
    stages = sorted({k for r in ok for k in r if k.startswith("t_") and k != "t_total_s"})
    labels = [int(r["label"]) for r in ok]
    scores = [r["score"] for r in ok]
    det = {"n": len(ok), "roc_auc": M.roc_auc(labels, scores), "ap": M.average_precision(labels, scores),
           "accuracy_at_threshold": float(np.mean([int(r["pred_label"]) == int(r["label"]) for r in ok])) if ok else None,
           "threshold": model.det_threshold, "score_kind": model.score_kind}
    loc_rows = [r for r in ok if "loc_f1" in r]
    loc = {"n": len(loc_rows)}
    for k in ("f1", "iou", "precision", "recall", "pixel_auc", "pixel_ap", "pred_fraction", "gt_fraction"):
        vals = [r[f"loc_{k}"] for r in loc_rows if not (isinstance(r[f"loc_{k}"], float) and math.isnan(r[f"loc_{k}"]))]
        loc[k] = float(np.mean(vals)) if vals else None
    for g in ("SD", "PS"):
        sub = [r for r in loc_rows if r["generator"] == g]
        loc[f"f1_{g}"] = float(np.mean([r["loc_f1"] for r in sub])) if sub else None
        loc[f"pixel_auc_{g}"] = float(np.nanmean([r["loc_pixel_auc"] for r in sub])) if sub else None
    totals = [r["t_total_s"] for r in ok]
    summary = {
        "model": model.name, "protocol": model.protocol, "device": device, "n_images": len(rows),
        "n_ok": len(ok), "n_errors": len(per_image) - len(ok), "environment": P.environment(),
        "parameters": params, "parameters_total": sum(params.values()),
        "timing_s": {
            "load": phases["load"][1] - phases["load"][0],
            "warmup_total": phases["warmup"][1] - phases["warmup"][0], "warmup_images": min(args.warmup, len(rows)),
            "per_image_total": latency_stats(totals),
            "per_image_stage": {s[2:-2]: latency_stats([r[s] for r in ok if s in r]) for s in stages},
            "throughput_img_per_s": len(ok) / sum(totals) if totals else None,
            "measure_wall": phases["measure"][1] - phases["measure"][0]},
        "flops": {"method": flops[0]["method"] if flops else None, "n_images": len(flop_vals),
                  "per_image_mean_gflops": float(np.mean(flop_vals)) / 1e9 if flop_vals else None,
                  "per_image_min_gflops": min(flop_vals) / 1e9 if flop_vals else None,
                  "per_image_max_gflops": max(flop_vals) / 1e9 if flop_vals else None,
                  "by_stage_first_image": flops[0].get("by_stage") if flops else None,
                  "by_op_first_image": flops[0].get("by_op") if flops else None},
        "memory": {
            "ram_rss_before_load_mib": rss_before / 2**20 if rss_before else None,
            "ram_peak_load_mib": (rss.peak_between(*phases["load"]) or 0) / 2**20,
            "ram_peak_measure_mib": (rss.peak_between(*phases["measure"]) or 0) / 2**20,
            "ram_peak_process_mib": rss_peak_process / 2**20 if rss_peak_process else None,
            "gpu_after_load": gpu_after_load,
            "gpu_peak_allocated_mib_max": max((r["gpu_peak_allocated_mib"] for r in ok), default=None) if on_gpu else None,
            "gpu_peak_reserved_mib_max": max((r["gpu_peak_reserved_mib"] for r in ok), default=None) if on_gpu else None,
            "gpu_peak_allocated_mib_median": statistics.median([r["gpu_peak_allocated_mib"] for r in ok]) if on_gpu and ok else None},
        "gpu_sampling": {ph: gpu.summary(*phases[ph]) for ph in ("load", "warmup", "measure")} if gpu else None,
        "energy_j_per_image": None,
        "torch_profiler": {"available": False, "reason": "not run (it runs after the results are saved)"},
        "detection": det, "localization": loc,
        "phases_s": {k: [a - t_origin, b - t_origin] for k, (a, b) in phases.items()},
    }
    if gpu and summary["gpu_sampling"]["measure"].get("energy_j") and ok:
        summary["energy_j_per_image"] = summary["gpu_sampling"]["measure"]["energy_j"] / len(ok)
    summary["model_info"] = {k: getattr(model, k) for k in ("backends", "cudnn", "root", "threshold")
                             if hasattr(model, k)}

    cols = list(dict.fromkeys(k for r in per_image for k in r))
    with open(out / "per_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(per_image)
    P.write_json(out / "profile.json", summary)
    lat = summary["timing_s"]["per_image_total"]
    print(f"\n{model.name}: {len(ok)}/{len(rows)} ok | latency median {lat['median'] * 1e3:.1f} ms, "
          f"p95 {lat['p95'] * 1e3:.1f} ms | {summary['timing_s']['throughput_img_per_s']:.2f} img/s | "
          f"GFLOPs/img {summary['flops']['per_image_mean_gflops'] or float('nan'):.1f} | "
          f"RAM peak {summary['memory']['ram_peak_measure_mib']:.0f} MiB"
          + (f" | GPU peak {summary['memory']['gpu_peak_allocated_mib_max']:.0f} MiB" if on_gpu else ""))
    print(f"detection AUC {det['roc_auc']:.3f} AP {det['ap']:.3f} | localization F1 {loc.get('f1') or float('nan'):.3f} "
          f"IoU {loc.get('iou') or float('nan'):.3f} pixel-AUC {loc.get('pixel_auc') or float('nan'):.3f}")
    print(f"wrote {out}", flush=True)

    # ---- torch.profiler, last: everything above is already saved if it is slow or interrupted.
    # ForMa's PyTorch selective-scan fallback launches a huge number of small operators, so it
    # gets 1 image and no trace unless asked otherwise.
    n_prof = min(len(rows), args.profile_images if args.profile_images is not None else (1 if args.model == "forma" else 3))
    trace = not args.no_trace and not (args.model == "forma" and args.profile_images is None)
    print(f"torch.profiler on {n_prof} image(s){' + trace' if trace else ''}...", flush=True)
    t0 = time.perf_counter()
    try:
        prof = P.torch_profile(lambda i: model.predict(root / rows[i % len(rows)]["file"], P.Stopwatch(device)),
                               min(n_prof, len(rows)), device, out, trace=trace)
    except Exception as e:  # noqa: BLE001
        prof = {"available": False, "reason": f"{type(e).__name__}: {e}"}
    phases["profile"] = (t0, time.perf_counter())
    summary["torch_profiler"] = prof
    summary["phases_s"]["profile"] = [phases["profile"][0] - t_origin, phases["profile"][1] - t_origin]
    P.write_json(out / "profile.json", summary)
    print(f"profiler done in {phases['profile'][1] - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
