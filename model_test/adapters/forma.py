"""ForMa, through the group's ForMa/inference.py: OpenCV RGB, linear resize to 512x512, /255,
2-class logits at 1/4 resolution, foreground softmax nearest-resized to the original size, > 0.5.

Detection score  ForMa has no detection head. As in the group's ForMa/evaluate_detection.py, the score
                 is the predicted manipulated-area fraction (share of pixels > 0.5); "inpainted" when
                 > 0 (any pixel flagged; the group sets no default threshold, so read the AUC). The
                 mean of the top 1% of the map is kept as the extra column top1pct_score.
Selective scan   uses the fastest backend importable in the env (CUDA kernels, Triton, or the plain
                 PyTorch fallback); the one used is recorded, since it changes the GPU timing a lot.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
FORMA = ROOT / "ForMa"
sys.path.insert(0, str(FORMA))

# ForMa/models/mamba2/ssd_minimal.py imports `ssd_combined` whenever Triton is importable (Linux with a
# CUDA PyTorch), but that file is not in the repo. Its function (selective_scan_chunk_fn) is imported by the
# model and never called, so a placeholder that raises if called is enough; results are unchanged and the
# Triton cross-scan kernels (csm_triton.py, which are in the repo) stay usable.
MAMBA2_STUB = False
if importlib.util.find_spec("triton") is not None and not (FORMA / "models" / "mamba2" / "ssd_combined.py").exists():
    _stub = types.ModuleType("ssd_combined")

    def _not_in_repo(*args, **kwargs):
        raise NotImplementedError("ForMa's mamba2 ssd_combined Triton kernels are not in the repo")

    _stub.mamba_chunk_scan_combined = _not_in_repo
    sys.modules["ssd_combined"] = sys.modules["models.mamba2.ssd_combined"] = _stub
    MAMBA2_STUB = True

import inference as forma_inference  # noqa: E402  (ForMa/inference.py)


class Model:
    name = "ForMa"
    protocol = ("VMamba + Noiseprint, 512x512 input; map = softmax foreground, nearest resize, > 0.5; "
                "score = predicted area fraction (group definition)")
    score_kind = "derived (predicted area fraction, as ForMa/evaluate_detection.py)"
    det_threshold = 0.0

    def __init__(self, device: str, opts: dict):
        self.device = device
        weights = Path(opts.get("forma_weights") or FORMA / "ForMa_weights" / "ForMa_weights.pth")
        if not weights.is_file():
            raise FileNotFoundError(f"{weights} missing: run python model_test/setup_models.py --only forma")
        self.model = forma_inference.load_model(weights, device)
        from models import csm_triton, csms6s  # noqa: E402  (ForMa/models, imported by load_model)
        if opts.get("forma_no_triton"):
            csm_triton.WITH_TRITON = False  # read at call time by cross_scan_fn / cross_merge_fn
        self.backends = {"triton": bool(csm_triton.WITH_TRITON),
                         "selective_scan_oflex": bool(csms6s.WITH_SELECTIVESCAN_OFLEX),
                         "selective_scan_core": bool(csms6s.WITH_SELECTIVESCAN_CORE),
                         "selective_scan_mamba": bool(csms6s.WITH_SELECTIVESCAN_MAMBA),
                         "mamba2_ssd_combined_stub": MAMBA2_STUB}

    def modules(self):
        return {"ForMa (VMamba + Noiseprint)": self.model}

    @torch.inference_mode()
    def predict(self, path: Path, sw) -> dict:
        with sw("preprocess"):
            tensor, image = forma_inference.preprocess_image(path, 512)
            tensor = tensor.to(self.device)
        with sw("forward"):
            out = forma_inference.run_inference(self.model, tensor)
        with sw("postprocess"):
            prob, pred = forma_inference.prediction_maps(out, image.size, 0.5)
            flat = np.sort(prob.ravel())
            top1pct = float(flat[-max(1, flat.size // 100):].mean())
            score = float(pred.mean())
        return {"score": score, "map": prob.astype(np.float32), "pred": pred,
                "extra": {"top1pct_score": top1pct}}
