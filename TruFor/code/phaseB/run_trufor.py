"""Run the official pretrained TruFor (trufor_ph3) on CPU over a list of images.

Preprocessing and post-processing are identical to TruFor_train_test/test.py: the official TestDataset
(full-resolution RGB / 256, no resize), softmax[1] for the anomaly map, sigmoid for confidence and score.
Only image paths reach this script; it never opens a GT mask (mask columns are dropped on read).

Per image it writes <out>/maps/<file>.npz (np.savez_compressed: map/conf as float16, imgsize; written to a temp
file then renamed, so a crash never leaves a partial .npz) and one row in <out>/results.csv with status, sizes,
timings, the score and all scalars at full float64 precision (repr).

- Resumable: an image is skipped if results.csv has a status=ok row for it AND its .npz exists. Failed images
  (status=error rows) are retried on the next run; downstream code uses the status=ok row per image.
- Crash-safe: an exception on one image is logged (status=error + message) and the run continues.
- Disk: free space is checked before every write; below config storage.min_free_disk_gb the run stops cleanly
  (exit code 3) with nothing half-written.
- Progress: tqdm bar on stderr, per-image lines in --log, and <out>/progress.json rewritten after every image.

Usage: python run_trufor.py <project_root> <rows.csv> <out_dir> [--log FILE] [--save-f32]
  --save-f32  also save the float32 maps under <out>/maps_f32/ (smoke test only, for the float16 check)."""
import argparse, csv, datetime, json, logging, os, shutil, signal, sys, time, types
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from tqdm import tqdm

ap = argparse.ArgumentParser()
ap.add_argument("root"); ap.add_argument("rows"); ap.add_argument("out")
ap.add_argument("--log", default=None)
ap.add_argument("--save-f32", action="store_true")
args = ap.parse_args()
root = os.path.abspath(args.root)
out = os.path.abspath(args.out)
cfg = json.load(open(os.path.join(root, "config/config.json")))
tgif = os.path.join(root, cfg["tgif_root"])
min_free = cfg["storage"]["min_free_disk_gb"] * 1e9
os.makedirs(out, exist_ok=True)

handlers = [logging.StreamHandler(sys.stdout)]
if args.log:
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    handlers.append(logging.FileHandler(args.log, mode="a"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=handlers)
log = logging.getLogger()


class LowDisk(Exception):
    pass


def check_disk():
    free = shutil.disk_usage(out).free
    if free < min_free:
        raise LowDisk(f"free disk {free / 1e9:.2f} GB < {min_free / 1e9:.1f} GB")
    return free


def now():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# SIGTERM (and SIGHUP, unless nohup already ignores it) end the run through the normal clean-up path
def _term(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")
signal.signal(signal.SIGTERM, _term)
if signal.getsignal(signal.SIGHUP) != signal.SIG_IGN:
    signal.signal(signal.SIGHUP, _term)

# ---- model (same audited load as code/phaseA2/check_checkpoint.py)
repo = os.path.join(root, "TruFor/TruFor_train_test")
ckpt_path = os.path.join(root, "weights/extracted/weights/trufor.pth.tar")
cwd = os.getcwd()
os.chdir(repo)  # update_config reads lib/config/<exp>.yaml relative to cwd
sys.path.insert(0, repo)
from lib.config import config, update_config
from lib.utils import get_model
from dataset.dataset_test import TestDataset
update_config(config, types.SimpleNamespace(experiment="trufor_ph3", gpu=-1, opts=["TEST.MODEL_FILE", ckpt_path]))
os.chdir(cwd)
import pickletools, zipfile
ALLOWED = {"collections OrderedDict", "torch._utils _rebuild_tensor_v2", "torch FloatStorage", "torch LongStorage",
           "numpy.core.multiarray scalar", "numpy dtype", "_codecs encode"}
z = zipfile.ZipFile(ckpt_path)
found, last = set(), []
for op, arg, _ in pickletools.genops(z.read([n for n in z.namelist() if n.endswith("data.pkl")][0])):
    if op.name in ("SHORT_BINUNICODE", "BINUNICODE", "UNICODE"): last = (last + [arg])[-2:]
    if op.name == "GLOBAL": found.add(arg)
    if op.name == "STACK_GLOBAL": found.add(" ".join(last))
assert found <= ALLOWED, f"unexpected pickle globals: {found - ALLOWED}"
model = get_model(config)
model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["state_dict"], strict=True)
model.eval()

# ---- inputs: image paths only, each image exactly once
MASK_COLS = {"mask_file"}
rows = [{k: v for k, v in r.items() if k not in MASK_COLS} for r in csv.DictReader(open(args.rows, newline=""))]
assert all("masks/" not in r["file"] for r in rows), "a GT mask path was given as model input"
assert len({r["image_id"] for r in rows}) == len({r["file"] for r in rows}) == len(rows), "duplicate images in input"
res_path = os.path.join(out, "results.csv")
prog_path = os.path.join(out, "progress.json")
FIELDS = ["image_id", "file", "label", "generator", "mask_type", "status", "error", "orig_w", "orig_h", "proc_h",
          "proc_w", "map_h", "map_w", "score", "det_logit", "map_mean", "map_max", "conf_mean", "t_load_s",
          "t_infer_s", "t_save_s", "npz", "torch_threads", "finished_at"]


def npz_path(r):
    return os.path.join(out, "maps", r["file"] + ".npz")


prev = list(csv.DictReader(open(res_path, newline=""))) if os.path.exists(res_path) else []
done = {p["image_id"] for p in prev if p.get("status", "ok") == "ok" and os.path.exists(npz_path(p))}
new_file = not os.path.exists(res_path)
fres = open(res_path, "a", newline="")
w = csv.DictWriter(fres, fieldnames=FIELDS)
if new_file:
    w.writeheader()

todo = [r for r in rows if r["image_id"] not in done]
paths = [os.path.join(tgif, r["file"]) for r in todo]
ds = TestDataset(list_img=paths)
started, t_start = now(), time.time()
n_ok, n_fail, failed_ids, last_id = 0, 0, [], None


def write_progress(status, note=""):
    el = time.time() - t_start
    attempted = n_ok + n_fail
    rate = attempted / el if el > 0 and attempted else 0.0
    remaining = len(todo) - attempted
    p = dict(status=status, note=note, total=len(rows), done_ok=len(done) + n_ok, skipped_at_start=len(done),
             failed_this_session=n_fail, failed_ids_this_session=failed_ids[-50:], remaining=remaining,
             images_per_s=rate, s_per_image=(1 / rate if rate else None),
             eta_s=(remaining / rate if rate else None), elapsed_s=el, last_image=last_id,
             free_disk_gb=shutil.disk_usage(out).free / 1e9, started=started, updated=now(), pid=os.getpid(),
             torch_threads=torch.get_num_threads(), results_csv=res_path)
    tmp = prog_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(p, f, indent=1)
    os.replace(tmp, prog_path)


log.info(f"START {len(rows)} images, {len(done)} already saved, {len(todo)} to run; torch threads="
         f"{torch.get_num_threads()}; free disk {shutil.disk_usage(out).free / 1e9:.1f} GB; pid {os.getpid()}")
status, note, code = "finished", "", 0
bar = tqdm(total=len(rows), initial=len(done), unit="img", dynamic_ncols=False, ncols=120, mininterval=2,
           smoothing=0.05, file=sys.stderr)
bar.set_postfix(fail=0)
try:
    check_disk()
    write_progress("running")
    with torch.no_grad():
        for i, r in enumerate(todo):
            last_id = r["image_id"]
            try:
                t0 = time.perf_counter()
                rgb, path = ds[i]
                rgb = rgb.unsqueeze(0)  # batch of 1, as DataLoader(batch_size=1) in test.py
                t1 = time.perf_counter()
                pred, conf, det, _ = model(rgb, save_np=False)
                conf = torch.sigmoid(torch.squeeze(conf, 0))[0].numpy()
                det_logit = float(det.item())
                score = float(torch.sigmoid(det).item())
                pred = F.softmax(torch.squeeze(pred, 0), dim=0)[1].numpy()
                t2 = time.perf_counter()
                with Image.open(path) as im:
                    ow, oh = im.size
                npz = npz_path(r)
                os.makedirs(os.path.dirname(npz), exist_ok=True)
                check_disk()
                tmp = npz + ".tmp.npz"
                np.savez_compressed(tmp, map=pred.astype(np.float16), conf=conf.astype(np.float16),
                                    imgsize=np.array(rgb.shape[2:]))
                os.replace(tmp, npz)
                if args.save_f32:
                    p32 = os.path.join(out, "maps_f32", r["file"] + ".npz")
                    os.makedirs(os.path.dirname(p32), exist_ok=True)
                    check_disk()
                    np.savez_compressed(p32, map=pred, conf=conf)
                t3 = time.perf_counter()
                row = dict(status="ok", error="", orig_w=ow, orig_h=oh, proc_h=rgb.shape[2], proc_w=rgb.shape[3],
                           map_h=pred.shape[0], map_w=pred.shape[1], score=repr(score), det_logit=repr(det_logit),
                           map_mean=repr(float(pred.astype(np.float64).mean())), map_max=repr(float(pred.max())),
                           conf_mean=repr(float(conf.astype(np.float64).mean())), t_load_s=repr(t1 - t0),
                           t_infer_s=repr(t2 - t1), t_save_s=repr(t3 - t2), npz=os.path.relpath(npz, root))
                n_ok += 1
                msg = f"OK   {r['image_id']} score={score:.6f} infer={t2 - t1:.2f}s"
            except (LowDisk, KeyboardInterrupt):
                raise
            except Exception as e:
                n_fail += 1
                failed_ids.append(r["image_id"])
                row = dict(status="error", error=f"{type(e).__name__}: {e}"[:500])
                msg = f"FAIL {r['image_id']} {row['error']}"
            row.update(image_id=r["image_id"], file=r["file"], label=r["label"], generator=r["generator"],
                       mask_type=r["mask_type"], torch_threads=torch.get_num_threads(), finished_at=now())
            check_disk()
            w.writerow(row)
            fres.flush()
            os.fsync(fres.fileno())
            log.info(f"[{len(done) + n_ok}/{len(rows)} ok, {n_fail} failed] {msg}")
            bar.update(1)
            bar.set_postfix(fail=n_fail)
            write_progress("running")
except LowDisk as e:
    status, note, code = "stopped_low_disk", str(e), 3
except KeyboardInterrupt as e:
    status, note, code = "interrupted", str(e) or "KeyboardInterrupt", 130
except Exception as e:
    status, note, code = "crashed", f"{type(e).__name__}: {e}", 1
    log.exception("unexpected error outside the per-image loop")
finally:
    bar.close()
    fres.close()
    if status == "finished" and n_fail:
        status = "finished_with_failures"
    write_progress(status, note)
    log.info(f"END status={status} {note} | ok this session {n_ok}, failed {n_fail}, "
             f"saved total {len(done) + n_ok}/{len(rows)}")
sys.exit(code)
