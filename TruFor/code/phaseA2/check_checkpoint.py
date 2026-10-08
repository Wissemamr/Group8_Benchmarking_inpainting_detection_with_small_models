"""Phase A2: build the official TruFor model (trufor_ph3 config), load the official checkpoint on CPU
with strict key matching, and count parameters. No inference is run.
Usage: python check_checkpoint.py <project_root>"""
import collections, json, os, sys, types
import torch

root = os.path.abspath(sys.argv[1])
repo = os.path.join(root, "TruFor/TruFor_train_test")
ckpt_path = os.path.join(root, "weights/extracted/weights/trufor.pth.tar")
os.chdir(repo)  # update_config reads lib/config/<exp>.yaml relative to cwd
sys.path.insert(0, repo)
from lib.config import config, update_config
from lib.utils import get_model

update_config(config, types.SimpleNamespace(experiment="trufor_ph3", gpu=-1, opts=["TEST.MODEL_FILE", ckpt_path]))


# The checkpoint holds a numpy scalar, which weights_only=True rejects in torch 2.2 (add_safe_globals
# only exists from 2.4). Audit the pickle's globals without executing it, then load in full only if
# each one is on the allowlist.
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
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
sd = ckpt["state_dict"]
model = get_model(config)
res = model.load_state_dict(sd, strict=True)
model.eval()

msd = model.state_dict()
mismatch = [k for k in sd if sd[k].shape != msd[k].shape or not torch.equal(sd[k], msd[k])]
params = list(model.named_parameters())
by_module = collections.Counter()
for n, p in params:
    by_module[n.split(".")[0]] += p.numel()
out = dict(
    checkpoint=ckpt_path, checkpoint_keys=sorted(ckpt.keys()), epoch=ckpt.get("epoch"),
    torch=torch.__version__, device="cpu",
    strict_load=dict(missing=list(res.missing_keys), unexpected=list(res.unexpected_keys)),
    tensors_in_checkpoint=len(sd), tensors_not_equal_after_load=mismatch,
    checkpoint_numel=sum(v.numel() for v in sd.values()),
    params_total=sum(p.numel() for _, p in params),
    params_trainable=sum(p.numel() for _, p in params if p.requires_grad),
    buffers_total=sum(b.numel() for b in model.buffers()),
    params_by_top_module=dict(by_module.most_common()),
)
print(json.dumps(out, indent=1))
os.makedirs(os.path.join(root, "outputs/phaseA2"), exist_ok=True)
json.dump(out, open(os.path.join(root, "outputs/phaseA2/checkpoint_check.json"), "w"), indent=1)
