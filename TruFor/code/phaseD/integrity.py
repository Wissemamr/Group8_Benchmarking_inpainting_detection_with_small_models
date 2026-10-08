"""Phase D step 1: integrity checks on the saved inference outputs, before any metric. Read-only on outputs/inference.
Usage: .venv-eval/bin/python code/phaseD/integrity.py <project_root>"""
import csv, hashlib, json, math, os, sys
from collections import Counter
from multiprocessing import Pool
import numpy as np
from PIL import Image
from tqdm import tqdm

G = {}
checks = []


def init(root_, tgif_, man_):
    G.update(root=root_, tgif=tgif_, man=man_)


def inspect(r):
    out = dict(image_id=r["image_id"], err="")
    try:
        z = np.load(os.path.join(G['root'], r["npz"]))
        H, W = int(r["orig_h"]), int(r["orig_w"])
        if set(z.files) != {"map", "conf", "imgsize"}:
            out["err"] += f"keys {z.files};"
        m, c = z["map"], z["conf"]
        for nm, a in (("map", m), ("conf", c)):
            if a.dtype != np.float16: out["err"] += f"{nm} dtype {a.dtype};"
            if a.shape != (H, W): out["err"] += f"{nm} shape {a.shape} != {(H, W)};"
            if not np.isfinite(a).all(): out["err"] += f"{nm} non-finite;"
            if a.min() < 0 or a.max() > 1: out["err"] += f"{nm} out of [0,1];"
        if tuple(z["imgsize"]) not in ((H, W), (W, H)): out["err"] += f"imgsize {z['imgsize']};"
        out["imgsize_order"] = "HW" if tuple(z["imgsize"]) == (H, W) else "WH"
        if abs(float(m.astype(np.float64).max()) - float(r["map_max"])) > 1e-3: out["err"] += "map_max mismatch;"
        out["n_map_eq_0.5"] = int((m == np.float16(0.5)).sum())
        mf = G['man'][r["image_id"]]["mask_file"]
        if r["label"] == "1":
            mk = Image.open(os.path.join(G['tgif'], mf))
            if mk.size != (W, H): out["err"] += f"mask size {mk.size} != {(W, H)};"
        elif mf:
            out["err"] += "real image has mask;"
    except Exception as e:
        out["err"] += f"EXC {type(e).__name__}: {e}"
    return out




def main():
    global checks
    root = os.path.abspath(sys.argv[1])
    cfg = json.load(open(os.path.join(root, "config/config.json")))
    tgif = os.path.join(root, cfg["tgif_root"])
    rdir = os.path.join(root, cfg["resamples_dir"])
    

    def check(name, ok, detail=""):
        checks.append(dict(check=name, ok=bool(ok), detail=str(detail)))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


    res = list(csv.DictReader(open(os.path.join(root, "outputs/inference/results.csv"), newline="")))
    man = {r["image_id"]: r for r in csv.DictReader(open(os.path.join(root, cfg["manifest_csv"]), newline=""))}
    E = cfg["expected"]

    # ---------------------------------------------------------------- results.csv
    check("results.csv has 4,407 rows", len(res) == E["unique_files"], len(res))
    check("all status == ok", all(r["status"] == "ok" for r in res), Counter(r["status"] for r in res))
    check("error column empty everywhere", all(r["error"] == "" for r in res))
    for col in ("image_id", "file", "npz"):
        c = Counter(r[col] for r in res)
        check(f"no duplicate {col}", max(c.values()) == 1, f"{len(c)} unique")
    byid = {r["image_id"]: r for r in res}
    check("image_id set == manifest", set(byid) == set(man), f"manifest {len(man)}")
    check("file/label/generator/mask_type agree with manifest",
          all(all(byid[i][k] == man[i][k] for k in ("file", "label", "generator", "mask_type")) for i in man))
    check("orig size == manifest width/height",
          all((byid[i]["orig_w"], byid[i]["orig_h"]) == (man[i]["width"], man[i]["height"]) for i in man))
    check("map size == orig size (CSV)", all((r["map_w"], r["map_h"]) == (r["orig_w"], r["orig_h"]) for r in res))
    sc = np.array([float(r["score"]) for r in res]); lg = np.array([float(r["det_logit"]) for r in res])
    check("scores and logits finite", np.isfinite(sc).all() and np.isfinite(lg).all())
    check("scores in [0,1]", (sc >= 0).all() and (sc <= 1).all(), f"min {sc.min():.3g} max {sc.max():.6g}")
    check("score == sigmoid(det_logit)", np.abs(sc - 1 / (1 + np.exp(-lg))).max() < 1e-6,
          f"max abs diff {np.abs(sc - 1 / (1 + np.exp(-lg))).max():.2e}")
    check("scores exactly 0.5 (threshold ties)", True, f"{int((sc == 0.5).sum())} images")

    # ---------------------------------------------------------------- resamples
    meta = json.load(open(os.path.join(rdir, cfg["resamples_json"])))
    sets = {}
    for k, fn in enumerate(cfg["resample_csvs"], 1):
        p = os.path.join(rdir, fn)
        sha = hashlib.sha256(open(p, "rb").read()).hexdigest()
        check(f"{fn} sha256 matches resamples.json", sha == meta["resamples"][str(k)]["csv_sha256"])
        rs = list(csv.DictReader(open(p, newline="")))
        ids = [r["image_id"] for r in rs]
        sets[k] = set(ids)
        check(f"{fn}: 3,575 rows, no duplicates", len(rs) == E["rows_per_resample"] and len(set(ids)) == len(ids),
              f"{len(rs)} rows / {len(set(ids))} unique")
        missing = [i for i in ids if i not in byid]
        check(f"{fn}: every row exists in results.csv", not missing, f"{len(missing)} missing")
        check(f"{fn}: file/label/generator/mask_type agree with results.csv",
              all(all(r[c] == byid[r["image_id"]][c] for c in ("file", "label", "generator", "mask_type")) for r in rs))
        g = Counter(r["generator"] for r in rs); gm = Counter((r["generator"], r["mask_type"]) for r in rs)
        check(f"{fn}: composition real 275 / SD 1650 / PS 1650, 825 per generator x mask type",
              g == {"real": 275, "SD": 1650, "PS": 1650} and all(v == 825 for (gg, _), v in gm.items() if gg != "real"),
              dict(g))
        check(f"{fn}: label == (generator != real)", all(r["label"] == ("0" if r["generator"] == "real" else "1") for r in rs))
    for a, b in ((1, 2), (1, 3), (2, 3)):
        check(f"resample_{a} != resample_{b} (overlap)", sets[a] != sets[b], f"{len(sets[a] & sets[b])} shared images")
    check("union of resamples covers all 4,407 images", set().union(*sets.values()) == set(byid),
          len(set().union(*sets.values())))


    # ---------------------------------------------------------------- every map file
    with Pool(8, init, (root, tgif, man)) as p:
        per = list(tqdm(p.imap(inspect, res, chunksize=20), total=len(res), desc='maps', mininterval=5, file=sys.stdout))
    bad = [x for x in per if x["err"]]
    check("every npz readable, keys map/conf/imgsize, float16, shape == (orig_h, orig_w), finite, in [0,1]",
          not bad, f"{len(bad)} bad" + (f": {bad[:5]}" if bad else ""))
    check("every manipulated image has a mask of the same size; reals have none", not bad)
    check("imgsize order", True, dict(Counter(x.get("imgsize_order") for x in per)))
    check("map pixels exactly == 0.5 (where > vs >= differ)", True, f"{sum(x.get('n_map_eq_0.5', 0) for x in per)} px total")
    OUT = os.path.join(root, "outputs/phaseD")
    ok = all(c["ok"] for c in checks)
    json.dump(dict(all_ok=ok, checks=checks), open(os.path.join(OUT, "integrity.json"), "w"), indent=1)
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
