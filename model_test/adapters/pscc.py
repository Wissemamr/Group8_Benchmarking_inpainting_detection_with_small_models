"""PSCC-Net, as in the official test.py (github.com/proteus1991/PSCC-Net): full-resolution RGB / 255
(no resize), HRNet backbone -> NLCDetection head (sigmoid mask, upsampled bilinearly to the input
size) and DetectionHead (softmax, class 1 = forged).
Detection: score = P(forged), predicted "inpainted" when > 0.5 (= the official argmax).
Localization: mask > threshold, 0.5 by default (`--pscc-threshold` changes it).

Code and checkpoints come from an existing PSCC-Net folder: --pscc-root, the env var PSCC_ROOT,
a PSCC-TGIF / PSCC-Net folder inside or next to this repo, ~/projects/PSCC-TGIF, or the official
clone made by setup_models.py (model_test/third_party/PSCC-Net).
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from setup_models import find_pscc_root  # noqa: E402


def _load(net, path: Path, device: str):
    state = torch.load(str(path), map_location=torch.device(device))
    # checkpoints were saved from nn.DataParallel: strip the "module." prefix and load strictly
    state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
    net.load_state_dict(state, strict=True)


class Model:
    name = "PSCC-Net"
    protocol = "official test.py, full resolution; mask = sigmoid map > 0.5; score = softmax P(forged) > 0.5"
    score_kind = "native (detection head)"
    det_threshold = 0.5

    def __init__(self, device: str, opts: dict):
        root = Path(opts["pscc_root"]) if opts.get("pscc_root") else find_pscc_root()
        if root is None:
            raise FileNotFoundError("no PSCC-Net folder found: pass --pscc-root or run python model_test/setup_models.py --only pscc")
        self.root, self.device = root, device
        self.threshold = float(opts.get("pscc_threshold") or 0.5)
        if not hasattr(np, "int"):
            np.int = int  # the official HRNet code uses np.int, removed in numpy 1.24; same meaning
        cwd = os.getcwd()
        os.chdir(root)  # HRNet's init reads the ImageNet weights path relative to the repo
        sys.path.insert(0, str(root))
        # HRNet's init_weights calls torch.load without map_location (the ImageNet weights, overwritten
        # by the PSCC checkpoints below); load them into CPU memory so construction works without a GPU
        torch_load = torch.load

        def _load_to_cpu(*a, **k):
            k.setdefault("map_location", "cpu")
            return torch_load(*a, **k)

        torch.load = _load_to_cpu
        try:
            from models.seg_hrnet import get_seg_model
            from models.seg_hrnet_config import get_hrnet_cfg
            from utils.config import get_pscc_args
            from models.NLCDetection import NLCDetection
            from models.detection_head import DetectionHead
            args = get_pscc_args()
            self.fenet = get_seg_model(get_hrnet_cfg())
            self.segnet = NLCDetection(args)
            self.clsnet = DetectionHead(args)
        finally:
            torch.load = torch_load
            os.chdir(cwd)
        ck = root / "checkpoint"
        for net, name in ((self.fenet, "HRNet"), (self.segnet, "NLCDetection"), (self.clsnet, "DetectionHead")):
            _load(net, ck / f"{name}_checkpoint" / f"{name}.pth", device)
            net.to(device).eval()

    def modules(self):
        return {"HRNet backbone": self.fenet, "NLCDetection (localization head)": self.segnet,
                "DetectionHead": self.clsnet}

    @torch.no_grad()
    def predict(self, path: Path, sw) -> dict:
        with sw("preprocess"):
            img = Image.open(path)
            img = img.convert("RGB") if img.mode != "RGBA" else _rgba_to_rgb(img)
            x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255).permute(2, 0, 1)[None].to(self.device)
        with sw("forward"):
            feat = self.fenet(x)
            mask = self.segnet(feat)[0]
            logit = self.clsnet(feat)
        with sw("postprocess"):
            mask = F.interpolate(mask, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=True)
            prob = mask[0, 0].cpu().numpy().astype(np.float32)
            score = float(torch.softmax(logit, dim=1)[0, 1].item())
        return {"score": score, "map": prob, "pred": prob > self.threshold,
                "extra": {"pscc_root": str(self.root), "mask_threshold": self.threshold}}


def _rgba_to_rgb(img: Image.Image) -> Image.Image:
    """Official load_vdata.rgba2rgb: composite on white."""
    a = np.asarray(img).astype(np.float32)
    alpha = a[..., 3:4] / 255.0
    rgb = a[..., :3] * alpha + (1.0 - alpha) * 255.0
    return Image.fromarray(rgb.astype(np.uint8))
