"""Run AEROBLADE once over the TGIF testing images (authentic + SD2 + Photoshop inpaintings).

One reconstruction pass per image feeds both tasks:
  detection     AEROBLADE's distance = spatial mean of the LPIPS (VGG layer 2) map between the image
                and its autoencoder reconstruction, computed exactly as AEROBLADE does (no padding)
  localization  the same LPIPS map, kept per pixel (reflect-padded by 16 px first, which removes
                VGG's zero-padding border artifact), saved as float32 .npy

Everything goes to cache/<image_size>_seed<seed>/:
  recon/<AE>/<category>/<stem>.png     8-bit reconstruction (as AEROBLADE stores it)
  maps/<AE>/<category>/<stem>.npy      LPIPS layer-2 map, H/2 x W/2
  records/<label>.jsonl                one JSON line per (image, AE): distance, map statistics, timings
  runs/<label>_<time>.json             arguments, environment, model load times, wall-clock time

Resumable: images with a record and a map file are skipped, so an interrupted run (or a run on
another machine with --shard i/n, whose cache folder is copied back) just continues. Images are
processed family by family (see dataset.family_order), so any prefix is a usable benchmark.
Runs on a CUDA GPU when one is available (--device auto, the default) and otherwise on the CPU, where
one 512x512 image takes ~17 s with the SD2 autoencoder on 6 threads. Computation is fp32 on both.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aeroblade_bench import imaging  # noqa: E402
from aeroblade_bench.autoencoders import AUTOENCODERS, load_autoencoder, reconstruct  # noqa: E402
from aeroblade_bench.config import (  # noqa: E402
    DEFAULT_CONFIG, ROOT, cache_dir, environment_info, load_config, rpath, sha256_json,
)
from aeroblade_bench.dataset import build_index, check_families, family_order  # noqa: E402
from aeroblade_bench.lpips_map import LPIPSLayerMap  # noqa: E402

PERCENTILES = (1, 5, 25, 50, 75, 95, 99)


def load_done(records_dir: Path) -> set[tuple[str, str]]:
    """(image_id, ae) pairs with a complete record in any records/*.jsonl (truncated lines skipped)."""
    done = set()
    for path in sorted(records_dir.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done.add((r["image_id"], r["ae"]))
    return done


def map_stats(layer_map: np.ndarray) -> dict:
    pct = np.percentile(layer_map, PERCENTILES)
    return {
        "mean": float(layer_map.mean()), "std": float(layer_map.std()),
        "min": float(layer_map.min()), "max": float(layer_map.max()),
        **{f"p{p:02d}": float(v) for p, v in zip(PERCENTILES, pct)},
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--max-sources", type=int, default=0,
                   help="only the first N source images (families) of the processing order; 0 = all")
    p.add_argument("--shard", default="0/1", help="i/n: process source images i, i+n, i+2n, ... of the order")
    p.add_argument("--aes", nargs="+", choices=list(AUTOENCODERS), help="override config model.aes")
    p.add_argument("--threads", type=int, help="torch intra-op threads (config: run.threads)")
    p.add_argument("--device", help="auto (cuda if available, else cpu), cuda, cuda:N or cpu "
                                    "(config: run.device)")
    args = p.parse_args()

    cfg = load_config(args.config)
    m, run = cfg["model"], cfg["run"]
    aes = args.aes or m["aes"]
    device = str(args.device or run.get("device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit(f"--device {device} requested but this PyTorch build sees no CUDA GPU "
                 f"(torch {torch.__version__}); run scripts/check_env.py")
    threads = args.threads or run["threads"]
    size, seed, layer = int(m["image_size"]), int(m["seed"]), int(m["lpips_layer"])
    pad_detect, pad_map = int(m["lpips_pad_detect"]), int(m["lpips_pad_map"])
    shard_i, shard_n = (int(v) for v in args.shard.split("/"))
    if threads:
        torch.set_num_threads(int(threads))

    data_root = rpath(cfg["data"]["root"])
    index_csv = rpath(cfg["data"]["index_csv"])
    if index_csv.exists():
        index = pd.read_csv(index_csv)
    else:
        index = build_index(data_root)
        index_csv.parent.mkdir(parents=True, exist_ok=True)
        index.to_csv(index_csv, index=False)
    check_families(index)
    ordered = family_order(index, int(run["order_seed"]))
    sources = list(dict.fromkeys(ordered["source_id"]))
    if args.max_sources:
        sources = sources[: args.max_sources]
    sources = sources[shard_i::shard_n]
    todo = ordered[ordered["source_id"].isin(set(sources))]
    todo = todo.assign(_r=todo["source_id"].map({s: i for i, s in enumerate(sources)}))
    todo = todo.sort_values("_r", kind="stable").drop(columns="_r")

    cache = cache_dir(cfg)
    records_dir, runs_dir = cache / "records", cache / "runs"
    for d in (records_dir, runs_dir):
        d.mkdir(parents=True, exist_ok=True)
    label = f"{socket.gethostname()}_shard{shard_i}of{shard_n}"
    started_at = datetime.now(timezone.utc)
    run_file = runs_dir / f"{label}_{started_at.strftime('%Y%m%dT%H%M%S')}.json"
    done = load_done(records_dir)

    def pending(row) -> list[str]:
        return [ae for ae in aes if (row.image_id, ae) not in done
                or not (cache / "maps" / ae / row.category / f"{Path(row.file).stem}.npy").exists()]

    work = [(row, need) for row in todo.itertuples() if (need := pending(row))]
    where = (f"{device} ({torch.cuda.get_device_name(torch.device(device))})" if device.startswith("cuda")
             else f"{device}, {torch.get_num_threads()} threads")
    print(f"{len(todo)} images in {len(sources)} source images (shard {shard_i}/{shard_n}) | "
          f"AEs {'+'.join(aes)} | {size}x{size} | {where} | "
          f"{len(todo) - len(work)} already done, {len(work)} to do", flush=True)

    run_info = {
        "label": label, "started_at": started_at.isoformat(), "device": device,
        "args": vars(args) | {"config": str(args.config)},
        "config": cfg, "config_sha256": sha256_json(cfg), "aes": {a: AUTOENCODERS[a].repo_id for a in aes},
        "n_images_in_scope": len(todo), "n_images_todo": len(work), "environment": environment_info(),
        "finished": False,
    }

    def write_run(extra: dict) -> None:
        run_file.write_text(json.dumps(run_info | extra, indent=1, default=str))

    write_run({})
    if not work:
        print("nothing to do")
        return

    load_s: dict[str, float] = {}
    t0 = time.perf_counter()
    models = {}
    for ae in aes:
        t = time.perf_counter()
        models[ae] = load_autoencoder(ae, device)
        load_s[ae] = time.perf_counter() - t
    t = time.perf_counter()
    lpips_model = LPIPSLayerMap(layer, pad=pad_map).to(device)
    load_s["lpips"] = time.perf_counter() - t
    print("model load (s):", {k: round(v, 1) for k, v in load_s.items()}, flush=True)

    n_done, wall_start, interrupted = 0, time.perf_counter(), False
    sum_total = 0.0
    out_path = records_dir / f"{label}.jsonl"
    try:
        with open(out_path, "a") as out:
            bar = tqdm(work, unit="img", dynamic_ncols=True)
            for row, need in bar:
                stem = Path(row.file).stem
                t_img = time.perf_counter()
                img = Image.open(data_root / row.file).convert("RGB")
                orig_w, orig_h = img.size
                x = imaging.to_tensor(img.resize((size, size), Image.BICUBIC))
                t_read = time.perf_counter() - t_img
                for ae in need:
                    t_start = time.perf_counter()
                    rec_path = cache / "recon" / ae / row.category / f"{stem}.png"
                    map_path = cache / "maps" / ae / row.category / f"{stem}.npy"

                    t = time.perf_counter()
                    rec = reconstruct(models[ae], imaging.pad_to_multiple(x).to(device), seed)
                    t_ae = time.perf_counter() - t

                    t = time.perf_counter()
                    imaging.save_png(rec[..., :size, :size].cpu(), rec_path)
                    x_rec = imaging.to_tensor(Image.open(rec_path).convert("RGB"))
                    t_io = time.perf_counter() - t

                    t = time.perf_counter()
                    map0 = lpips_model(x.to(device), x_rec.to(device), pad=pad_detect)[0, 0].cpu().numpy()
                    map16 = lpips_model(x.to(device), x_rec.to(device), pad=pad_map)[0, 0].cpu().numpy()
                    t_lp = time.perf_counter() - t

                    imaging.save_npy(map16.astype(np.float32), map_path)
                    s0, s16 = map_stats(map0), map_stats(map16)
                    t_total = time.perf_counter() - t_start + (t_read if ae == need[0] else 0.0)
                    record = {
                        "image_id": row.image_id, "ae": ae, "file": row.file,
                        "orig_w": orig_w, "orig_h": orig_h, "proc_w": size, "proc_h": size,
                        "recon_file": rec_path.relative_to(cache).as_posix(),
                        "map_file": map_path.relative_to(cache).as_posix(),
                        "map_h": int(map16.shape[0]), "map_w": int(map16.shape[1]),
                        "aeroblade_distance": s0["mean"],  # exact AEROBLADE lpips_vgg_2 (positive; low = generated)
                        "dist_std_pad0": s0["std"],
                        **{f"map_{k}": v for k, v in s16.items()},  # statistics of the pad-16 map
                        "t_read_s": t_read if ae == need[0] else 0.0,
                        "t_ae_s": t_ae, "t_recon_io_s": t_io, "t_lpips_s": t_lp, "t_total_s": t_total,
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    done.add((row.image_id, ae))
                    sum_total += t_total
                n_done += 1
                bar.set_postfix(id=row.image_id, s_per_img=f"{sum_total / n_done:.1f}")
                if n_done % 25 == 0:
                    write_run({"progress": {"images_done_this_run": n_done, "images_todo": len(work),
                                            "elapsed_s": time.perf_counter() - wall_start}})
    except KeyboardInterrupt:
        interrupted = True
        print("\ninterrupted; progress is saved, re-run the same command to continue", flush=True)
    finally:
        elapsed = time.perf_counter() - wall_start
        write_run({
            "finished": n_done == len(work), "model_load_s": load_s,
            "progress": {"images_done_this_run": n_done, "images_todo": len(work), "elapsed_s": elapsed},
            "sum_per_image_total_s": sum_total,
            "mean_per_image_total_s": sum_total / n_done if n_done else None,
            "ended_at": datetime.now(timezone.utc).isoformat(),
        })
    print(f"\n{n_done}/{len(work)} images in {elapsed / 3600:.2f} h "
          f"({sum_total / max(n_done, 1):.1f} s/image) -> {out_path.relative_to(ROOT)}", flush=True)
    if interrupted:
        sys.exit(130)  # non-zero, so run_all.sh stops here instead of evaluating a partial run


if __name__ == "__main__":
    main()
