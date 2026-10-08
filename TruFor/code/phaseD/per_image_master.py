"""Master per-image file for cross-model comparison: one row per unique image (4,407), key image_id.
Built only from saved outputs (results.csv, manifest, resample CSVs, outputs/phaseD/per_image_localization.csv, the
same per-image data the tables were computed from). Verifies against the published tables in outputs/tables/.
Usage: .venv-eval/bin/python code/phaseD/per_image_master.py <project_root>"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import loc_from_counts, PR_T  # the exact metric code behind the tables

COLS = ["model", "image_id", "source_id", "category", "generator", "mask_type", "variant", "label", "width", "height",
        "file", "mask_file", "in_resample_1", "in_resample_2", "in_resample_3",
        "det_score", "det_pred", "det_correct",
        "loc_iou", "loc_dice", "loc_pixel_precision", "loc_pixel_recall", "loc_pixel_f1",
        "loc_tp", "loc_fp", "loc_fn", "loc_tn"]
LOC = {"iou": "loc_iou", "dice": "loc_dice", "precision": "loc_pixel_precision", "recall": "loc_pixel_recall",
       "f1": "loc_pixel_f1", "tp": "loc_tp", "fp": "loc_fp", "fn": "loc_fn", "tn": "loc_tn"}


def main():
    root = os.path.abspath(sys.argv[1])
    cfg = json.load(open(os.path.join(root, "config/config.json")))
    rdir, T = os.path.join(root, cfg["resamples_dir"]), os.path.join(root, "outputs/tables")
    man = pd.read_csv(os.path.join(root, cfg["manifest_csv"]), keep_default_na=False, dtype=str)
    res = pd.read_csv(os.path.join(root, "outputs/inference/results.csv"), dtype={"image_id": str})
    pi = pd.read_csv(os.path.join(root, "outputs/phaseD/per_image_localization.csv"))

    df = man.drop(columns="resamples").copy()
    for c in ("source_id", "variant", "label", "width", "height"):
        df[c] = df[c].astype(int)
    df.insert(0, "model", "TruFor")
    for k, fn in enumerate(cfg["resample_csvs"], 1):
        ids = set(pd.read_csv(os.path.join(rdir, fn), dtype=str).image_id)
        df[f"in_resample_{k}"] = df.image_id.isin(ids).astype(int)
    df = df.merge(res[["image_id", "score"]].rename(columns={"score": "det_score"}), on="image_id", how="left",
                  validate="1:1")
    df["det_pred"] = (df.det_score > PR_T).astype(int)
    df["det_correct"] = (df.det_pred == df.label).astype(int)
    df = df.merge(pi[["image_id"] + list(LOC)].rename(columns=LOC), on="image_id", how="left", validate="1:1")
    for c in ("loc_tp", "loc_fp", "loc_fn", "loc_tn"):
        df[c] = df[c].astype("Int64")  # integer counts; real images stay empty (<NA>), never 0
    df = df[COLS].sort_values("image_id").reset_index(drop=True)
    out = os.path.join(T, "trufor_per_image.csv")
    df.to_csv(out, index=False)  # float64 written at full round-trip precision
    try:
        import pyarrow  # noqa: F401
        df.to_parquet(os.path.join(T, "trufor_per_image.parquet"), index=False)
        print("parquet written")
    except ImportError:
        print("pyarrow not available: parquet skipped")

    # ------------------------------------------------------------------ verification, from the written file
    d = pd.read_csv(out, keep_default_na=False, na_values={c: [""] for c in COLS if c.startswith("loc_")})
    checks = []

    def check(name, ok, detail=""):
        checks.append(bool(ok))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))

    real, manip = d[d.label == 0], d[d.label == 1]
    check("columns in the specified order", list(d.columns) == COLS)
    check("4,407 rows", len(d) == 4407, len(d))
    check("no duplicate image_id", d.image_id.is_unique)
    meta = [c for c in COLS[:15] if c != "mask_file"]
    check("no missing metadata (all columns except mask_file)", not (d[meta].astype(str) == "").any().any())
    check("mask_file present for all manipulated, empty for all real",
          (manip.mask_file != "").all() and (real.mask_file == "").all())
    check("label/generator counts", (len(real), len(manip)) == (339, 4068), f"real {len(real)}, manipulated {len(manip)}")
    check("det_score finite for all rows, det_pred/det_correct consistent",
          np.isfinite(d.det_score).all() and (d.det_pred == (d.det_score > 0.5)).all()
          and (d.det_correct == (d.det_pred == d.label)).all())
    check("det_score bit-identical to results.csv",
          (d.set_index("image_id").det_score == res.set_index("image_id").score.reindex(d.image_id)).all())
    lc = [c for c in COLS if c.startswith("loc_")]
    check("real images: all loc_* empty (NaN), never 0", real[lc].isna().all().all())
    check("manipulated images: no loc_* missing", manip[lc].notna().all().all())
    check("tp+fp+fn+tn == width*height", ((manip.loc_tp + manip.loc_fp + manip.loc_fn + manip.loc_tn)
                                          == manip.width * manip.height).all())
    rec = manip.apply(lambda r: loc_from_counts(int(r.loc_tp), int(r.loc_fp), int(r.loc_fn)), axis=1, result_type="expand")
    err = max(float(np.abs(rec[k] - manip[LOC[k]]).max()) for k in ("iou", "dice", "precision", "recall", "f1"))
    check("loc_* metrics == recomputed from tp/fp/fn", err < 1e-9, f"max abs diff {err:.1e}")
    check("loc_dice == loc_pixel_f1", (manip.loc_dice == manip.loc_pixel_f1).all())

    # per-resample recomputation vs the published (rounded) tables
    f4 = lambda v: f"{v:.4f}"
    H1 = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    H2 = ["loc_pixel_precision", "loc_pixel_recall", "loc_pixel_f1", "loc_dice", "loc_iou"]
    K2 = ["precision", "recall", "f1", "dice", "iou"]
    per = {"det": [], "mac": [], "poo": []}
    print("\nresample | source | Accuracy Precision Recall F1 ROC-AUC || macro P R F1 Dice IoU || pooled P R F1 Dice IoU")
    for k in (1, 2, 3):
        x = d[d[f"in_resample_{k}"] == 1]
        y, s, p = x.label, x.det_score, x.det_pred
        det = [accuracy_score(y, p), precision_score(y, p), recall_score(y, p), f1_score(y, p), roc_auc_score(y, s)]
        m = x[x.label == 1]
        mac = [m[c].mean() for c in H2]
        pr = loc_from_counts(int(m.loc_tp.sum()), int(m.loc_fp.sum()), int(m.loc_fn.sum()))
        poo = [pr[c] for c in K2]
        per["det"].append(det); per["mac"].append(mac); per["poo"].append(poo)
        mine = [f4(v) for v in det + mac + poo]
        pub = (pd.read_csv(os.path.join(T, f"table1_detection_resample_{k}.csv"), dtype=str).iloc[0, 1:].tolist()
               + pd.read_csv(os.path.join(T, f"table2_localization_macro_resample_{k}.csv"), dtype=str).iloc[0, 1:].tolist()
               + pd.read_csv(os.path.join(T, f"table2_localization_pooled_resample_{k}.csv"), dtype=str).iloc[0, 1:].tolist())
        print(f"   {k}     | master | {' '.join(mine[:5])} || {' '.join(mine[5:10])} || {' '.join(mine[10:])}")
        print(f"   {k}     | table  | {' '.join(pub[:5])} || {' '.join(pub[5:10])} || {' '.join(pub[10:])}")
        check(f"resample_{k}: 3,575 rows / 3,300 manipulated; all 15 numbers match the published table to 4 decimals",
              len(x) == 3575 and len(m) == 3300 and mine == pub)
    for key, name in (("det", "table1_detection_mean_std"), ("mac", "table2_localization_macro_mean_std"),
                      ("poo", "table2_localization_pooled_mean_std")):
        a = np.array(per[key])
        mine = [f"{f4(mu)} ± {f4(sd)}" for mu, sd in zip(a.mean(0), a.std(0, ddof=1))]
        pub = pd.read_csv(os.path.join(T, name + ".csv"), dtype=str).iloc[0, 1:].tolist()
        check(f"{name}: mean ± std (ddof=1) matches", mine == pub, " | ".join(mine))
    print("\nALL CHECKS PASSED" if all(checks) else "\nSOME CHECKS FAILED")
    sys.exit(0 if all(checks) else 1)


if __name__ == "__main__":
    main()
