"""Turn the cached AEROBLADE outputs into the detection and localization dataframes and tables.

Runs: `full` (every processed image) and `resample_1..n` (resamples/resample_k.csv). For each run
it writes, into results/<run>/:

  detection_per_image.{csv,parquet}      image_id, category, generator, mask_type, variant, label,
                                         score, threshold, prediction, outcome, group-level metrics,
                                         LPIPS-map statistics, timings
  detection_table{,_by_mask_type}.{csv,md}      the ID / mask ratio / Number / generator / category /
                                         F1 / recall / precision table (+ accuracy, ROC-AUC, ...)
  localization_per_image.{csv,parquet}   IoU, dice, pixel precision / recall / F1 / AP / ROC-AUC
                                         (both polarities), error inside vs outside the GT, timings
  localization_table{,_by_mask_type}.{csv,md}
  run_summary.json                       sizes, thresholds, overall metrics, timing, environment

Detection: score = -AEROBLADE distance (min over the AEs); the threshold is the Youden point
(max TPR - FPR) of a calibration set that never contains the images it is applied to:
  full run    5-fold cross-fitting over source images (each fold uses a threshold fitted on the others)
  resample k  fitted on the source images that are NOT in resample k
Localization is scored only on inpainted images (authentic images have no mask), at the original
image resolution. Per-image pixel metrics are computed once and reused by every resample.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aeroblade_bench import metrics as M  # noqa: E402
from aeroblade_bench.config import (  # noqa: E402
    DEFAULT_CONFIG, ROOT, cache_dir, environment_info, load_config, rpath, sha256_file, sha256_json,
)

KEY_COLS = ["image_id", "source_id", "category", "generator", "mask_type", "variant", "label",
            "width", "height", "file"]
TIME_COLS = ["t_read_s", "t_ae_s", "t_recon_io_s", "t_lpips_s", "t_total_s"]


# ----------------------------------------------------------------------------- inputs

def load_records(cache: Path, aes: list[str]) -> pd.DataFrame:
    """One row per image: AEROBLADE distance (min over `aes`), the best AE, statistics of its map,
    and the summed timings. Images missing a record for any AE are dropped."""
    rows = []
    for path in sorted((cache / "records").glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # truncated last line of an interrupted run
    if not rows:
        raise SystemExit(f"no records under {cache / 'records'}; run scripts/run_aeroblade.py first")
    rec = pd.DataFrame(rows)
    rec = rec[rec["ae"].isin(aes)].drop_duplicates(["image_id", "ae"], keep="last")
    complete = rec.groupby("image_id")["ae"].nunique()
    rec = rec[rec["image_id"].isin(complete[complete == len(aes)].index)]
    best = rec.loc[rec.groupby("image_id")["aeroblade_distance"].idxmin()].set_index("image_id")
    out = best[["ae", "aeroblade_distance", "dist_std_pad0", "orig_w", "orig_h", "map_h", "map_w"]
               + [c for c in best.columns if c.startswith("map_") and c not in ("map_file", "map_h", "map_w")]]
    out = out.rename(columns={"ae": "best_ae"})
    out["map_files"] = rec.groupby("image_id")["map_file"].apply(list)
    out[TIME_COLS] = rec.groupby("image_id")[TIME_COLS].sum()
    if len(aes) > 1:
        dists = rec.pivot(index="image_id", columns="ae", values="aeroblade_distance")
        out = out.join(dists.add_prefix("distance_"))
    return out.reset_index()


# ----------------------------------------------------------------------------- localization

def compute_localization(rows: pd.DataFrame, cfg: dict, aes: list[str], cache: Path, workers: int) -> pd.DataFrame:
    """Per-image pixel metrics for the inpainted images in `rows`, cached under cache/eval/."""
    loc_cfg = cfg["localization"]
    key = sha256_json({"loc": loc_cfg, "aes": aes, "thr": cfg["data"]["mask_threshold"],
                       "size": cfg["model"]["image_size"], "pad": cfg["model"]["lpips_pad_map"]})[:10]
    cache_file = cache / "eval" / f"localization_{key}.jsonl"
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cached: dict[str, dict] = {}
    if cache_file.exists():
        for line in open(cache_file):
            try:
                r = json.loads(line)
                cached[r["image_id"]] = r
            except json.JSONDecodeError:
                continue
    data_root = rpath(cfg["data"]["root"])
    todo = rows[(rows["label"] == 1) & ~rows["image_id"].isin(cached)]
    jobs = [{
        "image_id": r.image_id, "map_files": [str(cache / f) for f in r.map_files],
        "mask_file": str(data_root / r.mask_file), "mask_threshold": cfg["data"]["mask_threshold"],
        "polarity": loc_cfg["polarity"], "normalize": loc_cfg["normalize"],
        "percentiles": loc_cfg["percentiles"],
    } for r in todo.itertuples()]
    if jobs:
        print(f"localization: scoring {len(jobs)} images ({len(cached)} cached) on {workers} workers", flush=True)
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_worker_init) as pool, open(cache_file, "a") as out:
            for res in tqdm(pool.map(M.localization_job, jobs, chunksize=4), total=len(jobs), unit="img"):
                cached[res["image_id"]] = res
                out.write(json.dumps(res) + "\n")
    return pd.DataFrame([cached[i] for i in rows.loc[rows["label"] == 1, "image_id"] if i in cached])


def _worker_init() -> None:
    import torch

    torch.set_num_threads(1)


# ----------------------------------------------------------------------------- detection

def fit_threshold(calib: pd.DataFrame) -> float | None:
    y = calib["label"].to_numpy()
    if y.size == 0 or y.min() == y.max():
        return None
    return M.youden_threshold(y, calib["det_score"].to_numpy())


def apply_thresholds(df: pd.DataFrame, mode: str, cfg: dict, calib: pd.DataFrame | None) -> pd.DataFrame:
    """Adds det_threshold / det_pred / calibration columns. See the module docstring."""
    df = df.copy()
    notes = []
    if mode == "crossfit":
        src = df.drop_duplicates("source_id")[["source_id", "category"]]
        folds = M.crossfit_folds(src, int(cfg["detection"]["crossfit_folds"]), int(cfg["detection"]["crossfit_seed"]))
        df["calib_fold"] = df["source_id"].map(folds)
        thr = np.full(len(df), np.nan)
        for f in sorted(df["calib_fold"].unique()):
            held = df["calib_fold"] == f
            t = fit_threshold(df[~held])
            if t is None:
                t, source = fit_threshold(df), "in-sample (calibration folds lacked a class)"
                notes.append(f"fold {f}: {source}")
            thr[held.to_numpy()] = t
        df["det_threshold"] = thr
        df["threshold_source"] = f"{cfg['detection']['crossfit_folds']}-fold cross-fit over source images"
    else:
        t = fit_threshold(calib) if calib is not None and len(calib) else None
        source = "source images outside the resample"
        if t is None:
            t, source = fit_threshold(df), "in-sample (no held-out calibration data)"
            notes.append(source)
        df["det_threshold"] = t
        df["calib_fold"] = -1
        df["threshold_source"] = source
        df["n_calibration_images"] = 0 if calib is None else len(calib)
    df["det_pred"] = (df["det_score"] >= df["det_threshold"]).astype(int)
    df["outcome"] = np.select(
        [(df["label"] == 1) & (df["det_pred"] == 1), (df["label"] == 0) & (df["det_pred"] == 1),
         (df["label"] == 1) & (df["det_pred"] == 0)], ["TP", "FP", "FN"], "TN")
    df["correct"] = (df["det_pred"] == df["label"]).astype(int)
    df.attrs["threshold_notes"] = notes
    return df


def add_group_metrics(det: pd.DataFrame) -> pd.DataFrame:
    """Repeat each inpainted image's (generator, category) group metrics on its row; the metrics
    of a group need >1 image, so they cannot be per-image quantities. NaN for authentic rows."""
    det = det.copy()
    for col in ("grp_accuracy", "grp_precision", "grp_recall", "grp_f1", "grp_roc_auc"):
        det[col] = np.nan
    for (gen, cat), pos in det[det["label"] == 1].groupby(["generator", "category"]):
        neg = det[(det["label"] == 0) & (det["category"] == cat)]
        grp = pd.concat([pos, neg])
        m = M.detection_metrics(grp["label"], grp["det_pred"], grp["det_score"])
        for name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            det.loc[pos.index, f"grp_{name}"] = m[name]
    return det


def paired_detection(det: pd.DataFrame) -> dict:
    """Each inpainted image against the authentic original of the same source image.

    Scene content dominates AEROBLADE's distance, so the pooled AUC mixes the effect of inpainting
    with differences between scenes; the paired comparison removes the scene. `frac_lower` is the
    share of pairs whose inpainted image has the lower distance (AEROBLADE's premise predicts > 0.5).
    """
    real = det[det["label"] == 0].set_index("source_id")["aeroblade_distance"]
    inp = det[det["label"] == 1].assign(d0=lambda d: d["source_id"].map(real)).dropna(subset=["d0"])
    inp = inp.assign(rel_diff=(inp["aeroblade_distance"] - inp["d0"]) / inp["d0"])
    groups = {"ALL": inp}
    groups |= {g: grp for g, grp in inp.groupby("generator")}
    groups |= {f"{g}_{mt}": grp for (g, mt), grp in inp.groupby(["generator", "mask_type"])}
    return {name: {"n_pairs": len(g), "frac_lower": float((g["rel_diff"] < 0).mean()),
                   "mean_rel_diff": float(g["rel_diff"].mean()),
                   "median_rel_diff": float(g["rel_diff"].median())} for name, g in groups.items()}


def overall(det: pd.DataFrame | None, loc: pd.DataFrame | None) -> dict:
    out: dict = {}
    if det is not None and len(det):
        out["detection_paired"] = paired_detection(det)
    if det is not None and len(det):
        for gen in ("ALL", "SD", "PS"):
            pos = det[(det["label"] == 1) & ((det["generator"] == gen) | (gen == "ALL"))]
            grp = pd.concat([pos, det[det["label"] == 0]])
            if len(pos):
                out.setdefault("detection", {})[gen] = M.detection_metrics(grp["label"], grp["det_pred"], grp["det_score"]) | {
                    "n_inpainted": len(pos), "n_authentic": int((det["label"] == 0).sum())}
    if loc is not None and len(loc):
        for gen in ("ALL", "SD", "PS"):
            g = loc[(loc["generator"] == gen) | (gen == "ALL")]
            if len(g):
                out.setdefault("localization", {})[gen] = {
                    "n_images": len(g),
                    "median_err_ratio_in_out": float(g["err_ratio_in_out"].median()),
                    "frac_images_error_lower_inside": float((g["err_ratio_in_out"] < 1).mean()),
                    "mean_gt_fraction": float(g["gt_fraction"].mean()),
                    **{f"macro_{c}": float(g[c].mean()) for c in (
                        "iou", "dice", "pixel_precision", "pixel_recall", "pixel_f1", "pixel_ap", "pixel_roc_auc",
                        "alt_iou", "alt_dice", "alt_pixel_ap", "alt_pixel_roc_auc")},
                    **{f"micro_{k}": v for k, v in M.pixel_counts_metrics(
                        *(int(g[c].sum()) for c in ("tp", "fp", "fn", "tn"))).items()},
                }
    return out


# ----------------------------------------------------------------------------- one run

def write_run(name: str, out_root: Path, base: pd.DataFrame, loc_all: pd.DataFrame, mode: str, calib: pd.DataFrame | None,
              cfg: dict, extra_meta: dict, aes: list[str]) -> dict:
    t_run = time.perf_counter()
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    stat_cols = [c for c in base.columns if c.startswith(("map_", "distance_", "dist_std")) and c not in ("map_files",)]

    # --- detection
    t = time.perf_counter()
    det = base[KEY_COLS + ["aeroblade_distance", "best_ae"] + stat_cols + TIME_COLS].copy()
    det["det_score"] = -det["aeroblade_distance"]
    det = apply_thresholds(det, mode, cfg, calib)
    det = add_group_metrics(det)
    lead = KEY_COLS + ["det_score", "aeroblade_distance", "best_ae", "det_threshold", "det_pred", "correct", "outcome",
                       "grp_accuracy", "grp_precision", "grp_recall", "grp_f1", "grp_roc_auc"]
    det = det[lead + [c for c in det.columns if c not in lead]]
    det_tab = M.detection_table(det)
    det_tab_mt = M.detection_table(det, by_mask_type=True)
    det_seconds = time.perf_counter() - t

    # --- localization
    t = time.perf_counter()
    loc_keys = base[base["label"] == 1][KEY_COLS + ["mask_file"] + TIME_COLS]
    loc = loc_keys.merge(loc_all, on="image_id", how="inner")
    lead_l = KEY_COLS + ["mask_file", "polarity", "iou", "dice", "pixel_precision", "pixel_recall", "pixel_f1",
                         "pixel_ap", "pixel_roc_auc", "tp", "fp", "fn", "tn", "gt_fraction", "pred_fraction"]
    loc = loc[lead_l + [c for c in loc.columns if c not in lead_l]]
    loc_tab = M.localization_table(loc)
    loc_tab_mt = M.localization_table(loc, by_mask_type=True)
    loc_seconds = time.perf_counter() - t

    for stem, df in (("detection_per_image", det), ("localization_per_image", loc)):
        df.to_csv(out / f"{stem}.csv", index=False)
        df.to_parquet(out / f"{stem}.parquet", index=False)
    M.save_table(det_tab, out / "detection_table")
    M.save_table(det_tab_mt, out / "detection_table_by_mask_type")
    M.save_table(loc_tab, out / "localization_table")
    M.save_table(loc_tab_mt, out / "localization_table_by_mask_type")

    summary = {
        "run": name, **extra_meta,
        "n_images_detection": len(det), "n_inpainted": int((det["label"] == 1).sum()),
        "n_authentic": int((det["label"] == 0).sum()), "n_images_localization": len(loc),
        "n_source_images": int(det["source_id"].nunique()),
        "autoencoders": aes,
        "detection_threshold": {
            "method": cfg["detection"]["threshold"], "score": cfg["detection"]["score"],
            "source": str(det["threshold_source"].iloc[0]) if len(det) else None,
            "values": sorted(set(np.round(det["det_threshold"].dropna(), 8).tolist())),
            "notes": det.attrs.get("threshold_notes", []),
        },
        "metrics": overall(det, loc),
        "timing": {
            "inference_seconds_sum": float(det["t_total_s"].sum()),
            "inference_seconds_mean_per_image": float(det["t_total_s"].mean()),
            "inference_seconds_median_per_image": float(det["t_total_s"].median()),
            "inference_seconds_by_stage_sum": {c: float(det[c].sum()) for c in TIME_COLS[:-1]},
            "localization_pixel_metrics_seconds_sum": float(loc["eval_seconds"].sum()),
            "detection_aggregation_seconds": det_seconds,
            "localization_aggregation_seconds": loc_seconds,
            "evaluation_seconds_this_run": time.perf_counter() - t_run,
            "note": "inference = reconstruction + LPIPS maps (one AEROBLADE pass serves both tasks); "
                    "resamples reuse the full run's per-image inference, so their inference time is the "
                    "sum over their own images.",
        },
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


# ----------------------------------------------------------------------------- overlays

def make_overlays(loc: pd.DataFrame, cfg: dict, cache: Path, out_dir: Path, n_per: int = 1) -> int:
    """Qualitative figures: the first `n_per` image of every (generator, category)."""
    from aeroblade_bench import viz
    from aeroblade_bench.lpips_map import upsample

    data_root = rpath(cfg["data"]["root"])
    n = 0
    for (_, _), grp in loc.groupby(["generator", "category"]):
        for r in grp.head(n_per).itertuples():
            maps = [np.load(cache / "maps" / a / r.category / f"{Path(r.file).stem}.npy")
                    for a in cfg["model"]["aes"]]
            gt = np.asarray(Image.open(data_root / r.mask_file).convert("L")) > cfg["data"]["mask_threshold"]
            err = upsample(np.minimum.reduce(maps), gt.shape)
            norm = M.normalize(err, cfg["localization"]["normalize"], tuple(cfg["localization"]["percentiles"]))
            score = 1 - norm if cfg["localization"]["polarity"] == "inverted" else norm
            t = M.otsu_threshold(score)
            viz.save_overlay(
                out_dir / f"{r.generator}_{r.category.replace(' ', '-')}_{r.image_id}.png",
                image=np.asarray(Image.open(data_root / r.file).convert("RGB")), score=score, pred=score > t,
                gt=gt, display_wh=(r.width, r.height),
                title=f"{r.image_id} | AE {'+'.join(cfg['model']['aes'])}, {cfg['model']['image_size']}x{cfg['model']['image_size']} "
                      f"| {cfg['localization']['polarity']} polarity",
                score_label="1 - norm. LPIPS error" if cfg["localization"]["polarity"] == "inverted" else "norm. LPIPS error",
                pred_label=f"Otsu mask (t={t:.2f})  IoU {r.iou:.2f}  F1 {r.dice:.2f}",
            )
            n += 1
    return n


# ----------------------------------------------------------------------------- comparison

def comparison(summaries: dict[str, dict], out_root: Path) -> None:
    rows_d, rows_l = [], []
    for name, s in summaries.items():
        for gen, m in s["metrics"].get("detection", {}).items():
            rows_d.append({"run": name, "generator": gen, **{k: m[k] for k in (
                "accuracy", "precision", "recall", "f1", "roc_auc", "ap", "balanced_accuracy")},
                "n_inpainted": m["n_inpainted"], "n_authentic": m["n_authentic"]})
        for gen, m in s["metrics"].get("localization", {}).items():
            rows_l.append({"run": name, "generator": gen, "n_images": m["n_images"], **{k: m[k] for k in (
                "macro_pixel_f1", "macro_pixel_recall", "macro_pixel_precision", "macro_iou", "macro_dice",
                "macro_pixel_ap", "macro_pixel_roc_auc", "micro_pixel_f1")}})
    for rows, stem in ((rows_d, "comparison_detection"), (rows_l, "comparison_localization")):
        df = pd.DataFrame(rows)
        if df.empty:
            continue
        rs = df[df["run"].str.startswith("resample_")]
        num = [c for c in df.columns if c not in ("run", "generator") and not c.startswith("n_")]
        extra = [df]
        if len(rs) > 1:  # mean / sample std (ddof=1) over the resample runs, per generator
            for label, fn in (("resamples mean", "mean"), ("resamples std", "std")):
                extra.append(rs.groupby("generator", sort=False)[num].agg(fn).reset_index().assign(run=label))
        M.save_table(pd.concat(extra, ignore_index=True)[list(df.columns)], out_root / stem)


# ----------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--out", type=Path, default=ROOT / "results")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--overlays", type=int, default=1, help="overlay figures per (generator, category); 0 = none")
    p.add_argument("--allow-partial", action="store_true", help="evaluate even if not all images were processed")
    args = p.parse_args()

    t_start = time.perf_counter()
    cfg = load_config(args.config)
    aes = list(cfg["model"]["aes"])
    cache = cache_dir(cfg)
    index = pd.read_csv(rpath(cfg["data"]["index_csv"]))
    rec = load_records(cache, aes)
    base = index.merge(rec, on="image_id", how="inner")
    missing = len(index) - len(base)
    if missing and not args.allow_partial:
        raise SystemExit(f"{missing} of {len(index)} images have no AEROBLADE record yet "
                         f"(--allow-partial evaluates what exists)")
    print(f"{len(base)}/{len(index)} images with records ({base['source_id'].nunique()} source images)", flush=True)

    # wall-clock of the compute runs (all invocations, incl. model loading)
    wall, load_total, runs = 0.0, 0.0, []
    for f in sorted((cache / "runs").glob("*.json")):
        r = json.loads(f.read_text())
        runs.append({"label": r["label"], "started_at": r["started_at"], "finished": r.get("finished"),
                     "images_done": r.get("progress", {}).get("images_done_this_run"),
                     "elapsed_s": r.get("progress", {}).get("elapsed_s"), "model_load_s": r.get("model_load_s")})
        wall += r.get("progress", {}).get("elapsed_s") or 0.0
        load_total += sum((r.get("model_load_s") or {}).values())
    env = environment_info()
    cfg_hash = sha256_json(cfg)
    shared = {
        "environment": env, "config": cfg, "config_sha256": cfg_hash,
        "index_csv_sha256": sha256_file(rpath(cfg["data"]["index_csv"])),
        "compute_runs": runs,
        "compute_wall_clock_seconds_all_invocations": wall,
        "compute_wall_clock_note": "includes model loading, image reading and writing the cache; "
                                   "sum over every invocation of run_aeroblade.py (resumed runs add up)",
    }

    loc_all = compute_localization(base, cfg, aes, cache, args.workers)

    summaries: dict[str, dict] = {}
    summaries["full"] = write_run("full", args.out, base, loc_all, "crossfit", None, cfg,
                                  shared | {"description": "all processed testing images"}, aes)
    resample_dir = ROOT / "resamples"
    for k in range(1, int(cfg["resamples"]["n"]) + 1):
        mp = resample_dir / f"resample_{k}.csv"
        if not mp.exists():
            print(f"skipping resample {k}: {mp} not found")
            continue
        manifest = pd.read_csv(mp)
        in_rs = base["image_id"].isin(set(manifest["image_id"]))
        sub = base[in_rs]
        calib = base[~base["source_id"].isin(set(manifest["source_id"]))].assign(
            det_score=lambda d: -d["aeroblade_distance"])
        if sub.empty:
            print(f"skipping resample {k}: none of its images has a record yet")
            continue
        meta = shared | {
            "description": f"resample {k}: {manifest['source_id'].nunique()} source images",
            "resample_csv": f"resamples/{mp.name}", "resample_csv_sha256": sha256_file(mp),
            "n_images_in_manifest": len(manifest), "n_images_missing_records": int(len(manifest) - len(sub)),
        }
        summaries[f"resample_{k}"] = write_run(f"resample_{k}", args.out, sub, loc_all, "heldout", calib, cfg, meta, aes)

    comparison(summaries, args.out)
    if args.overlays:
        loc = pd.read_parquet(args.out / "full" / "localization_per_image.parquet")
        n = make_overlays(loc, cfg, cache, args.out / "figures" / "overlays", args.overlays)
        print(f"{n} overlay figures -> {args.out / 'figures' / 'overlays'}")

    print(f"\nevaluation finished in {time.perf_counter() - t_start:.0f} s -> {args.out}")
    for name, s in summaries.items():
        d = s["metrics"].get("detection", {}).get("ALL", {})
        l = s["metrics"].get("localization", {}).get("ALL", {})
        print(f"  {name:11s} det: acc {d.get('accuracy', float('nan')):.3f} F1 {d.get('f1', float('nan')):.3f} "
              f"AUC {d.get('roc_auc', float('nan')):.3f} | loc: F1 {l.get('macro_pixel_f1', float('nan')):.3f} "
              f"IoU {l.get('macro_iou', float('nan')):.3f} AUC {l.get('macro_pixel_roc_auc', float('nan')):.3f}")


if __name__ == "__main__":
    main()
