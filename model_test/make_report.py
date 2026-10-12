"""Combine the per-model outputs into model_test/outputs/report/REPORT.md (+ figures).

    python model_test/make_report.py

Needs numpy and matplotlib (any model env that has them, e.g. the AEROBLADE one). Reads
model_test/outputs/<model>/profile.json and per_image.csv for every model that has run.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"  # overridable with --outputs
REP = OUT / "report"
ORDER = ["aeroblade", "trufor", "forma", "pscc"]
COLORS = {"aeroblade": "#2a78d6", "trufor": "#eb6834", "forma": "#1baf7a", "pscc": "#d55181"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def fnum(v, d=1, unit=""):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    return f"{v:,.{d}f}{unit}"


def load():
    runs = {}
    for m in ORDER:
        pj = OUT / m / "profile.json"
        if pj.exists():
            prof = json.loads(pj.read_text(encoding="utf-8"))
            rows = list(csv.DictReader(open(OUT / m / "per_image.csv", newline="", encoding="utf-8")))
            runs[m] = (prof, rows)
    return runs


def ms(stat, key="median"):
    return stat[key] * 1e3 if stat and stat.get(key) is not None else None


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" if i == 0 else "---:" for i in range(len(headers))) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def figures(runs):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    made = []

    def style(ax):
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)

    models = [m for m in ORDER if m in runs]
    names = [runs[m][0]["model"] for m in models]

    # latency: median with p95 marker
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(models) + 1.2), facecolor=SURFACE)
    style(ax)
    med = [ms(runs[m][0]["timing_s"]["per_image_total"]) for m in models]
    p95 = [ms(runs[m][0]["timing_s"]["per_image_total"], "p95") for m in models]
    y = np.arange(len(models))
    ax.barh(y, med, color=[COLORS[m] for m in models], height=0.55)
    ax.scatter(p95, y, marker="|", s=250, color=INK, zorder=3, label="p95")
    for yi, v in zip(y, med):
        ax.text(v, yi, f"  {v:,.0f} ms", va="center", color=INK, fontsize=8)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("latency per image, batch 1 (bar = median, tick = p95)", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(REP / "latency.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    made.append("latency.png")

    # memory: GPU peak allocated and RAM peak
    on_gpu = all(runs[m][0]["memory"].get("gpu_peak_allocated_mib_max") for m in models)
    panels = [("ram_peak_measure_mib", "peak process RAM during the run (MiB)")]
    if on_gpu:
        panels.insert(0, ("gpu_peak_allocated_mib_max", "peak PyTorch GPU memory, largest image (MiB)"))
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 0.55 * len(models) + 1.2), facecolor=SURFACE)
    for ax, (key, label) in zip(np.atleast_1d(axes), panels):
        style(ax)
        vals = [runs[m][0]["memory"].get(key) or 0 for m in models]
        ax.barh(y, vals, color=[COLORS[m] for m in models], height=0.55)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f"  {v:,.0f}", va="center", color=INK, fontsize=8)
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xlabel(label, color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(REP / "memory.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    made.append("memory.png")

    # quality vs speed
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), facecolor=SURFACE)
    for ax, (getter, label) in zip(axes, [
            (lambda pr: pr["detection"]["roc_auc"], "detection ROC-AUC (image level)"),
            (lambda pr: pr["localization"]["pixel_auc"], "localization pixel ROC-AUC (mean per image)")]):
        style(ax)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for m, nm, x in zip(models, names, med):
            v = getter(runs[m][0])
            if v is None or (isinstance(v, float) and math.isnan(v)):
                continue
            ax.scatter(x, v, s=70, color=COLORS[m], edgecolors=SURFACE, linewidths=1.5, zorder=3)
            ax.annotate(nm, (x, v), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK)
        ax.axhline(0.5, color=MUTED, lw=1, ls="--")
        ax.set_xscale("log")
        ax.set_ylim(0.3, 1.0)
        ax.set_xlabel("median latency per image (ms, log scale)", color=INK, fontsize=9)
        ax.set_ylabel(label, color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(REP / "quality_vs_speed.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    made.append("quality_vs_speed.png")

    # qualitative examples
    try:
        from PIL import Image
        man = list(csv.DictReader(open(HERE / "testset" / "manifest.csv", newline="")))
        picks = [r for r in man if r["label"] == "1" and r["category"] in ("airplane", "zebra")][:4]
        cols = 2 + len(models)
        fig, axes = plt.subplots(len(picks), cols, figsize=(2.1 * cols, 1.75 * len(picks) + 0.4), facecolor=SURFACE)
        for i, r in enumerate(picks):
            img = Image.open(HERE / "testset" / r["file"]).convert("RGB")
            img.thumbnail((384, 384))
            gt = Image.open(HERE / "testset" / r["mask_file"]).convert("L").resize(img.size)
            panels = [(np.asarray(img), None), (np.asarray(gt), "gray")]
            for m in models:
                pth = OUT / m / "maps" / f"{r['image_id']}.png"
                panels.append((np.asarray(Image.open(pth).resize(img.size)) if pth.exists() else np.zeros(img.size[::-1]), "magma"))
            for j, (arr, cmap) in enumerate(panels):
                ax = axes[i][j]
                ax.imshow(arr, cmap=cmap, vmin=0, vmax=255)
                ax.set_xticks([]), ax.set_yticks([])
                for s in ax.spines.values():
                    s.set_visible(False)
            axes[i][0].set_ylabel(f"{r['category']}\n{r['generator']} {r['mask_type']}", fontsize=7, color=INK)
        for j, t in enumerate(["image", "ground truth"] + names):
            axes[0][j].set_title(t, fontsize=8, color=INK)
        fig.tight_layout()
        fig.savefig(REP / "examples.png", dpi=120, facecolor=SURFACE)
        plt.close(fig)
        made.append("examples.png")
    except Exception as e:  # noqa: BLE001
        print("examples figure skipped:", e)
    return made


def main() -> None:
    global OUT, REP
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs", type=Path, default=OUT, help="folder with one sub-folder per model")
    OUT = ap.parse_args().outputs
    REP = OUT / "report"
    REP.mkdir(parents=True, exist_ok=True)
    runs = load()
    if not runs:
        raise SystemExit(f"no model outputs under {OUT} yet")
    models = [m for m in ORDER if m in runs]
    any_prof = runs[models[0]][0]
    env0 = any_prof["environment"]
    L = ["# Four-model test: compute profile and quality on 65 TGIF images", ""]
    L.append(f"Machine: **{env0.get('gpu') or 'no GPU'}**"
             + (f" ({env0.get('gpu_memory_gb')} GB, compute capability {env0.get('gpu_compute_capability')})" if env0.get('gpu') else "")
             + f", CPU {env0.get('cpu') or '?'} ({env0.get('cpu_logical_cores')} logical cores), RAM {env0.get('ram_gb')} GB, "
             f"{env0.get('platform')}. NVIDIA driver / board: {env0.get('nvidia_smi', '–')}.")
    L.append("")
    L.append("Test set: `model_test/testset/`, one source photo per TGIF category (13), each with its authentic original, "
             "SD2-sp bbox and segm, PS-sp bbox and segm (65 images, 52 inpainted with masks). Every model sees every image "
             "once at batch size 1 after a warm-up, with CUDA-synchronized timings. Quality numbers on 65 images are "
             "indicative only; the full-split benchmarks are in each model's folder.")
    L.append("")

    L.append("## Environments")
    L.append(table(["model", "conda env", "python", "torch", "CUDA build", "device", "protocol"],
                   [[runs[m][0]["model"], runs[m][0]["environment"].get("conda_env") or "–",
                     runs[m][0]["environment"]["python"], runs[m][0]["environment"]["torch"],
                     runs[m][0]["environment"].get("torch_cuda") or "–", runs[m][0]["device"], runs[m][0]["protocol"]]
                    for m in models]))
    L.append("")

    L.append("## Compute")
    rows = []
    for m in models:
        pr = runs[m][0]
        t, mem, fl = pr["timing_s"], pr["memory"], pr["flops"]
        g = (pr.get("gpu_sampling") or {}).get("measure") or {}
        rows.append([pr["model"], fnum(pr["parameters_total"] / 1e6, 1), fnum(fl.get("per_image_mean_gflops"), 1),
                     fnum(ms(t["per_image_total"]), 1), fnum(ms(t["per_image_total"], "p95"), 1),
                     fnum(t.get("throughput_img_per_s"), 2), fnum(t["load"], 1),
                     fnum(mem.get("gpu_peak_allocated_mib_max"), 0), fnum(mem.get("gpu_peak_reserved_mib_max"), 0),
                     fnum(g.get("vram_used_max_mib"), 0), fnum(mem.get("ram_peak_measure_mib"), 0),
                     fnum(g.get("util_gpu_mean_pct"), 0), fnum(g.get("power_mean_w"), 1),
                     fnum(pr.get("energy_j_per_image"), 2)])
    L.append(table(["model", "params (M)", "GFLOPs / img", "latency median (ms)", "p95 (ms)", "img/s", "load (s)",
                    "GPU peak alloc (MiB)", "GPU peak reserved (MiB)", "VRAM used, nvidia-smi (MiB)",
                    "RAM peak (MiB)", "GPU util (%)", "power (W)", "energy (J/img)"], rows))
    L.append("")
    L.append("* Latency = whole prediction per image (reading, preprocessing, forward, post-processing), batch 1, after warm-up.\n"
             "* GFLOPs: `torch.utils.flop_counter`, 2 FLOPs per multiply-add of matmuls, convolutions and attention; "
             "element-wise operations and custom kernels are not counted.\n"
             "* GPU peak alloc/reserved: PyTorch's allocator in this process (largest image). VRAM used, GPU util, power: "
             "`nvidia-smi` every 100 ms during the measured pass (whole GPU, includes the CUDA context and other processes).\n"
             "* Energy: GPU board power integrated over the measured pass, divided by the number of images.")
    L.append("")

    L.append("## Where the time goes (median ms per stage)")
    stages = []
    for m in models:
        for s in runs[m][0]["timing_s"]["per_image_stage"]:
            if s not in stages:
                stages.append(s)
    L.append(table(["model"] + stages,
                   [[runs[m][0]["model"]] + [fnum(ms(runs[m][0]["timing_s"]["per_image_stage"].get(s)), 1) for s in stages]
                    for m in models]))
    L.append("")

    L.append("## Top operators (torch.profiler, self time over the profiled images)")
    for m in models:
        tp = runs[m][0].get("torch_profiler") or {}
        if not tp.get("available"):
            L.append(f"* {runs[m][0]['model']}: profiler unavailable ({tp.get('reason', '?')})")
            continue
        key = tp["sorted_by"]
        tot = tp["self_gpu_ms_total"] if key == "self_gpu_ms" else tp["self_cpu_ms_total"]
        L.append(f"**{runs[m][0]['model']}** ({tp['n_calls']} images, {'GPU' if key == 'self_gpu_ms' else 'CPU'} "
                 f"self time {tot:,.1f} ms; full table in `outputs/{m}/torch_ops.txt`)")
        L.append("")
        L.append(table(["operator", "calls", "self time (ms)", "share"],
                       [[r["op"][:60], r["calls"], fnum(r[key], 2), f"{100 * r[key] / tot:.1f}%" if tot else "–"]
                        for r in tp["top_ops"][:8]]))
        L.append("")

    L.append("## Quality on the 65 images (indicative)")
    qrows = []
    for m in models:
        d, l = runs[m][0]["detection"], runs[m][0]["localization"]
        qrows.append([runs[m][0]["model"], d["score_kind"], fnum(d["roc_auc"], 3), fnum(d["ap"], 3),
                      fnum(d["accuracy_at_threshold"], 3), fnum(l.get("f1"), 3), fnum(l.get("iou"), 3),
                      fnum(l.get("pixel_auc"), 3), fnum(l.get("f1_SD"), 3), fnum(l.get("f1_PS"), 3)])
    L.append(table(["model", "detection score", "det. ROC-AUC", "det. AP", "det. accuracy", "loc. F1", "loc. IoU",
                    "loc. pixel AUC", "F1 SD2-sp", "F1 PS-sp"], qrows))
    L.append("")
    L.append("13 authentic and 52 inpainted images: one image changes detection AUC by about 0.015. Localization F1 / IoU "
             "use each model's own binarization (TruFor, PSCC-Net, ForMa: > 0.5; AEROBLADE: Otsu); pixel AUC needs no threshold.")
    L.append("")

    notes = []
    for m in models:
        pr = runs[m][0]
        info = pr.get("model_info") or {}
        if "backends" in info:
            b = info["backends"]
            used = [k for k, v in b.items() if v]
            notes.append(f"* ForMa ran with selective-scan backends: {', '.join(used) if used else 'PyTorch fallback only (no CUDA/Triton kernels)'}. "
                         "Without the authors' CUDA kernels its GPU time is pessimistic.")
        if pr["n_errors"]:
            notes.append(f"* {pr['model']}: {pr['n_errors']} images failed (see per_image.csv).")
    if notes:
        L.append("## Notes")
        L.extend(notes)
        L.append("")

    figs = figures(runs)
    if figs:
        L.append("## Figures")
        for f in figs:
            L.append(f"![{f}]({f})")
        L.append("")
    (REP / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    summary = {m: {k: runs[m][0][k] for k in ("model", "device", "timing_s", "flops", "memory", "energy_j_per_image",
                                               "detection", "localization", "parameters_total")} for m in models}
    (REP / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print(f"wrote {REP / 'REPORT.md'} and {len(figs)} figures")


if __name__ == "__main__":
    main()
