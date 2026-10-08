"""Phase D steps 2-5: detection and localization metrics from the saved inference outputs (read-only), per resample
and mean ± std over resamples. Writes per-image data to outputs/phaseD/ and the tables to outputs/tables/.

Detection : positive = manipulated (label 1); predicted manipulated iff score > 0.5 (official TruFor rule);
            ROC-AUC from the continuous score.
Localization (main, neutral protocol): pred = map > 0.5 (strict, as in official metrics.py), GT = mask_L >= 128.
            No best-of-map/inverted-map, no boundary exclusion. Macro = mean of per-image metrics,
            pooled = TP/FP/FN summed over all pixels of all images. Per-image 0/0 convention: a metric whose
            denominator is 0 is 1 if prediction and GT are both empty, otherwise 0.
Official TruFor protocol (diagnostic only): TruFor/TruFor_train_test/metrics.py imported unmodified,
            computeLocalizationMetrics(map, gt) -> (F1_best, F1_th); macro nanmean, as in its usage example.
Usage: .venv-eval/bin/python code/phaseD/metrics.py <project_root>"""
import csv, importlib.util, json, os, sys, traceback
from multiprocessing import Pool
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PR_T = 0.5
GT_T = 128
ROW = "TruFor (official pretrained, zero-shot on TGIF; no TGIF fine-tuning)"
G = {}


# ------------------------------------------------------------------ metric functions (the only ones used below)
def counts(p, g):
    tp = int(np.count_nonzero(p & g)); fp = int(np.count_nonzero(p & ~g)); fn = int(np.count_nonzero(~p & g))
    return tp, fp, fn, int(p.size) - tp - fp - fn


def ratio(num, den, both_empty):
    return num / den if den > 0 else (1.0 if both_empty else 0.0)


def loc_from_counts(tp, fp, fn):
    """Precision, recall, F1, Dice, IoU from pixel counts (scalars or arrays summed beforehand)."""
    e = (tp + fp == 0) and (tp + fn == 0)  # prediction and GT both empty
    f1 = ratio(2 * tp, 2 * tp + fp + fn, e)
    return dict(precision=ratio(tp, tp + fp, e), recall=ratio(tp, tp + fn, e), f1=f1,
                dice=ratio(2 * tp, 2 * tp + fp + fn, e), iou=ratio(tp, tp + fp + fn, e))


def loc(p, g):
    return loc_from_counts(*counts(p, g)[:3])


def unit_tests():
    a = lambda *v: np.array(v, bool)
    r = loc(a(1, 1, 0, 0), a(1, 0, 1, 0))
    assert r == dict(precision=.5, recall=.5, f1=.5, dice=.5, iou=1 / 3), r
    r = loc(a(1, 1, 1, 0), a(1, 0, 0, 0))
    assert np.isclose(r["precision"], 1 / 3) and r["recall"] == 1 and np.isclose(r["f1"], .5) and np.isclose(r["iou"], 1 / 3)
    assert loc(a(0, 0), a(0, 0)) == dict(precision=1., recall=1., f1=1., dice=1., iou=1.)  # both empty
    assert loc(a(0, 0), a(1, 0)) == dict(precision=0., recall=0., f1=0., dice=0., iou=0.)  # empty prediction
    assert loc(a(1, 0), a(0, 0)) == dict(precision=0., recall=0., f1=0., dice=0., iou=0.)  # empty GT
    rng = np.random.default_rng(0)
    for _ in range(200):  # Dice == F1 identity and Dice = 2 IoU / (1 + IoU)
        p, g = rng.random(500) < rng.random(), rng.random(500) < rng.random()
        r = loc(p, g)
        assert r["f1"] == r["dice"] and np.isclose(r["dice"], 2 * r["iou"] / (1 + r["iou"]))
    return "passed: hand-computed cases, both-empty=1, empty-pred=0, empty-GT=0, F1==Dice, Dice=2IoU/(1+IoU)"


# ------------------------------------------------------------------ per-image worker
def init(root, tgif, official_path):
    spec = importlib.util.spec_from_file_location("trufor_official_metrics", official_path)
    off = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(off)
    traceback.print_exc = lambda *a, **k: None  # official code prints a traceback when it returns NaN; we count NaNs
    G.update(root=root, tgif=tgif, off=off)


def per_image(r):
    import warnings
    warnings.simplefilter("ignore")
    m = np.load(os.path.join(G["root"], r["npz"]))["map"].astype(np.float32)
    g = np.asarray(Image.open(os.path.join(G["tgif"], r["mask_file"])).convert("L")) >= GT_T
    assert m.shape == g.shape
    p = m > PR_T
    tp, fp, fn, tn = counts(p, g)
    tp2, fp2, fn2, _ = counts(m >= PR_T, g)  # README/config rule, recorded only to quantify the > vs >= deviation
    out = dict(image_id=r["image_id"], n_px=g.size, gt_px=int(g.sum()), pred_px=int(p.sum()),
               tp=tp, fp=fp, fn=fn, tn=tn, tp_ge=tp2, fp_ge=fp2, fn_ge=fn2, **loc_from_counts(tp, fp, fn))
    # sanity on the real mask with the final metric code
    sg, se = loc(g, g), loc(np.zeros_like(g), g)
    out["san_gt_vs_gt_all_1"] = all(v == 1 for v in sg.values())
    out["san_empty_pred_all_0"] = all(v == 0 for v in se.values()) if g.any() else None
    # official TruFor protocol (diagnostic)
    f1_best, f1_th = G["off"].computeLocalizationMetrics(m, g)
    out["off_f1_best"], out["off_f1_th"] = float(f1_best), float(f1_th)
    return out


# ------------------------------------------------------------------ aggregation helpers
def detection(y, s):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
        confusion_matrix, balanced_accuracy_score
    yp = (s > PR_T).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
    ones = np.ones_like(y)
    return dict(accuracy=accuracy_score(y, yp), precision=precision_score(y, yp, zero_division=0),
                recall=recall_score(y, yp), f1=f1_score(y, yp), roc_auc=roc_auc_score(y, s),
                balanced_accuracy=balanced_accuracy_score(y, yp), tn=tn, fp=fp, fn=fn, tp=tp,
                n=len(y), n_pos=int(y.sum()), n_neg=int((1 - y).sum()),
                baseline_all_manip_accuracy=accuracy_score(y, ones), baseline_all_manip_f1=f1_score(y, ones))


def localization(d):
    mac = {f"macro_{k}": d[k].mean() for k in ("precision", "recall", "f1", "dice", "iou")}
    poo = {f"pooled_{k}": v for k, v in loc_from_counts(int(d.tp.sum()), int(d.fp.sum()), int(d.fn.sum())).items()}
    poo_ge = loc_from_counts(int(d.tp_ge.sum()), int(d.fp_ge.sum()), int(d.fn_ge.sum()))
    return dict(n_images=len(d), **mac, **poo,
                n_gt_empty=int((d.gt_px == 0).sum()), n_pred_empty=int((d.pred_px == 0).sum()),
                n_both_empty=int(((d.gt_px == 0) & (d.pred_px == 0)).sum()),
                official_f1_th_nanmean=np.nanmean(d.off_f1_th), official_f1_best_nanmean=np.nanmean(d.off_f1_best),
                official_n_nan_th=int(d.off_f1_th.isna().sum()), official_n_nan_best=int(d.off_f1_best.isna().sum()),
                sens_pred_ge_pooled_f1=poo_ge["f1"], sens_pred_ge_pooled_iou=poo_ge["iou"])


def fmt(v):
    return f"{v:.4f}"


def md_table(header, rows):
    s = "| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n"
    return s + "".join("| " + " | ".join(str(x) for x in r) + " |\n" for r in rows)


def main():
    root = os.path.abspath(sys.argv[1])
    cfg = json.load(open(os.path.join(root, "config/config.json")))
    tgif = os.path.join(root, cfg["tgif_root"])
    rdir = os.path.join(root, cfg["resamples_dir"])
    D, T = os.path.join(root, "outputs/phaseD"), os.path.join(root, "outputs/tables")
    os.makedirs(D, exist_ok=True); os.makedirs(T, exist_ok=True)
    integ = json.load(open(os.path.join(D, "integrity.json")))
    assert integ["all_ok"], "integrity checks did not pass; run integrity.py first"
    print("metric unit tests:", unit_tests(), flush=True)

    res = pd.read_csv(os.path.join(root, "outputs/inference/results.csv"), dtype={"label": int})
    man = pd.read_csv(os.path.join(root, cfg["manifest_csv"]), keep_default_na=False)
    res = res.merge(man[["image_id", "category", "mask_file"]], on="image_id", validate="1:1")
    rs = {k: pd.read_csv(os.path.join(rdir, f), keep_default_na=False) for k, f in enumerate(cfg["resample_csvs"], 1)}

    # ---------------------------------------------------------- per-image localization pass (cached)
    pi_path = os.path.join(D, "per_image_localization.csv")
    manip = res[res.label == 1]
    if os.path.exists(pi_path) and set(pd.read_csv(pi_path).image_id) == set(manip.image_id):
        print(f"using cached {pi_path}", flush=True)
    else:
        off = os.path.join(root, "TruFor/TruFor_train_test/metrics.py")
        with Pool(8, init, (root, tgif, off)) as pool:
            out = list(tqdm(pool.imap_unordered(per_image, manip.to_dict("records"), chunksize=4),
                            total=len(manip), desc="localization", mininterval=10, file=sys.stdout))
        pd.DataFrame(out).sort_values("image_id").to_csv(pi_path, index=False, float_format="%.10g")
    pi = pd.read_csv(pi_path).merge(res[["image_id", "generator", "mask_type", "category"]], on="image_id")
    san = dict(n_images=len(pi), gt_vs_gt_all_metrics_1=bool(pi.san_gt_vs_gt_all_1.all()),
               empty_pred_all_metrics_0=bool(pi.san_empty_pred_all_0.dropna().astype(bool).all()),
               n_empty_gt_skipped_in_empty_pred_check=int(pi.san_empty_pred_all_0.isna().sum()))
    print("sanity on real masks:", san, flush=True)
    assert san["gt_vs_gt_all_metrics_1"] and san["empty_pred_all_metrics_0"]

    # ---------------------------------------------------------- per resample
    det_rows, det_diag, loc_rows, loc_split, flags = [], [], [], [], []
    for k, r in rs.items():
        d = res[res.image_id.isin(r.image_id)]
        assert len(d) == 3575
        y, s = d.label.to_numpy(), d.score.to_numpy()
        det_rows.append(dict(resample=k, **detection(y, s)))
        for gen in ("SD", "PS"):
            x = d[d.generator.isin(["real", gen])]
            m = detection(x.label.to_numpy(), x.score.to_numpy())
            det_diag.append(dict(resample=k, subset=f"{gen} vs real", n_pos=m["n_pos"], n_neg=m["n_neg"],
                                 roc_auc=m["roc_auc"], recall=m["recall"], accuracy=m["accuracy"], f1=m["f1"]))
        L = pi[pi.image_id.isin(r.image_id)]
        assert len(L) == 3300
        loc_rows.append(dict(resample=k, **localization(L)))
        for col in ("generator", "mask_type", ["generator", "mask_type"], "category"):
            for key, x in L.groupby(col):
                key = key if isinstance(key, tuple) else (key,)
                loc_split.append(dict(resample=k, split="+".join(col) if isinstance(col, list) else col,
                                      group=" / ".join(map(str, key)), **localization(x)))
    det, detd = pd.DataFrame(det_rows), pd.DataFrame(det_diag)
    locr, locs = pd.DataFrame(loc_rows), pd.DataFrame(loc_split)

    # ---------------------------------------------------------- suspicious-number checks
    for name, df in (("detection", det), ("localization", locr)):
        for c in df.columns:
            if c == "resample" or not np.issubdtype(df[c].dtype, np.number):
                continue
            v = df[c].to_numpy(float)
            if np.isnan(v).any():
                flags.append(f"{name}.{c}: NaN")
            elif not c.startswith(("n_", "official_n_", "tn", "fp", "fn", "tp")) and c not in ("n", "baseline_all_manip_accuracy", "baseline_all_manip_f1") \
                    and np.all(v == v[0]):
                flags.append(f"{name}.{c}: identical across resamples ({v[0]})")
    for c in ("roc_auc",):
        if (det[c] < 0.5).any() or (detd[c] < 0.5).any():
            flags.append(f"{c} below 0.5")
    print("suspicious-number flags:", flags or "none", flush=True)

    # ---------------------------------------------------------- mean ± std (sample std, ddof=1, n=3)
    def ms(df, cols, by=None):
        g = df.groupby(by) if by else [((), df)]
        rows = []
        for key, x in g:
            row = dict(zip(by, key if isinstance(key, tuple) else (key,))) if by else {}
            for c in cols:
                row[f"{c}_mean"], row[f"{c}_std"] = x[c].mean(), x[c].std(ddof=1)
            rows.append(row)
        return pd.DataFrame(rows)

    DC = ["accuracy", "precision", "recall", "f1", "roc_auc", "balanced_accuracy",
          "baseline_all_manip_accuracy", "baseline_all_manip_f1"]
    LC = [c for c in locr.columns if c.startswith(("macro_", "pooled_", "official_f1", "sens_"))]
    det_ms, loc_ms = ms(det, DC), ms(locr, LC)
    detd_ms = ms(detd, ["roc_auc", "recall", "accuracy", "f1"], ["subset"])
    locs_ms = ms(locs, [c for c in LC if not c.startswith("sens_")], ["split", "group"])

    # ---------------------------------------------------------- save CSVs
    det.to_csv(os.path.join(T, "detection_per_resample_full.csv"), index=False)
    det_ms.to_csv(os.path.join(T, "detection_mean_std_full.csv"), index=False)
    detd.to_csv(os.path.join(T, "diag_detection_sd_ps_vs_real_per_resample.csv"), index=False)
    detd_ms.to_csv(os.path.join(T, "diag_detection_sd_ps_vs_real_mean_std.csv"), index=False)
    locr.to_csv(os.path.join(T, "localization_per_resample_full.csv"), index=False)
    loc_ms.to_csv(os.path.join(T, "localization_mean_std_full.csv"), index=False)
    locs[locs.split != "category"].to_csv(os.path.join(T, "diag_localization_splits_per_resample.csv"), index=False)
    locs_ms[locs_ms.split != "category"].to_csv(os.path.join(T, "diag_localization_splits_mean_std.csv"), index=False)
    locs[locs.split == "category"].to_csv(os.path.join(T, "diag_localization_per_category_per_resample.csv"), index=False)
    locs_ms[locs_ms.split == "category"].to_csv(os.path.join(T, "diag_localization_per_category_mean_std.csv"), index=False)

    # ---------------------------------------------------------- README-format tables (Markdown + CSV)
    md = []
    H1 = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    K1 = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    H2 = ["Model", "Pixel Precision", "Pixel Recall", "Pixel F1", "Dice", "IoU"]
    K2 = ["precision", "recall", "f1", "dice", "iou"]

    def emit(name, title, header, rows):
        md.append(f"### {title}\n\n" + md_table(header, rows))
        with open(os.path.join(T, name + ".csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(header); w.writerows(rows)

    for k in rs:
        dr = det[det["resample"] == k].iloc[0]
        lr = locr[locr["resample"] == k].iloc[0]
        emit(f"table1_detection_resample_{k}", f"Table 1 — Detection, resample_{k}", H1,
             [[ROW] + [fmt(dr[c]) for c in K1]])
        emit(f"table2_localization_macro_resample_{k}", f"Table 2a — Localization (macro), resample_{k}", H2,
             [[ROW] + [fmt(lr["macro_" + c]) for c in K2]])
        emit(f"table2_localization_pooled_resample_{k}", f"Table 2b — Localization (pooled), resample_{k}", H2,
             [[ROW] + [fmt(lr["pooled_" + c]) for c in K2]])
    pm = lambda df, c: f"{fmt(df[c + '_mean'].iloc[0])} ± {fmt(df[c + '_std'].iloc[0])}"
    emit("table1_detection_mean_std", "Table 1 — Detection, mean ± std over 3 resamples", H1,
         [[ROW] + [pm(det_ms, c) for c in K1]])
    emit("table2_localization_macro_mean_std", "Table 2a — Localization (macro), mean ± std over 3 resamples", H2,
         [[ROW] + [pm(loc_ms, "macro_" + c) for c in K2]])
    emit("table2_localization_pooled_mean_std", "Table 2b — Localization (pooled), mean ± std over 3 resamples", H2,
         [[ROW] + [pm(loc_ms, "pooled_" + c) for c in K2]])
    emit("diag_official_trufor_protocol",
         "DIAGNOSTIC — official TruFor protocol (not comparable with other models)",
         ["Resample", "F1 @0.5, best of map/inverted, boundary excluded (macro)",
          "F1 @ oracle per-image threshold, best of map/inverted, boundary excluded (macro)", "NaN images skipped (th / best)"],
         [[f"resample_{k}", fmt(locr.official_f1_th_nanmean[i]), fmt(locr.official_f1_best_nanmean[i]),
           f"{locr.official_n_nan_th[i]} / {locr.official_n_nan_best[i]}"] for i, k in enumerate(rs)] +
         [["mean ± std", pm(loc_ms, "official_f1_th_nanmean"), pm(loc_ms, "official_f1_best_nanmean"), ""]])
    open(os.path.join(T, "tables.md"), "w").write("\n".join(md))
    json.dump(dict(sanity=san, flags=flags, unit_tests=unit_tests()), open(os.path.join(D, "metrics_checks.json"), "w"), indent=1)
    print("\n".join(md))
    print("done", flush=True)


if __name__ == "__main__":
    main()
