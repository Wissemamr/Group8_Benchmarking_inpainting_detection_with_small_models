"""TruFor, as in the group's TruFor/ benchmark and the official test.py (experiment trufor_ph3):
full-resolution RGB / 256 (no resize), anomaly map = softmax[1], detection score = sigmoid(det).
Predicted "inpainted" when score > 0.5 and pixels when map > 0.5 (official rules). On a GPU, cuDNN
is configured from the TruFor config, as the official test.py does.
Needs TruFor/TruFor (official code) and TruFor/weights (setup_models.py --only trufor).
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
REPO = ROOT / "TruFor" / "TruFor" / "TruFor_train_test"
CKPT = ROOT / "TruFor" / "weights" / "extracted" / "weights" / "trufor.pth.tar"


class Model:
    name = "TruFor"
    protocol = "official trufor_ph3, full resolution; map = softmax[1] > 0.5; score = sigmoid(det) > 0.5"
    score_kind = "native (integrity score)"
    det_threshold = 0.5

    def __init__(self, device: str, opts: dict):
        if not REPO.is_dir() or not CKPT.is_file():
            raise FileNotFoundError("TruFor code or weights missing: run python model_test/setup_models.py --only trufor")
        self.device = device
        cwd = os.getcwd()
        os.chdir(REPO)  # update_config reads lib/config/<experiment>.yaml relative to the working directory
        sys.path.insert(0, str(REPO))
        try:
            from lib.config import config, update_config
            from lib.utils import get_model
            from dataset.dataset_test import TestDataset
            gpu = int(device.split(":")[1]) if device.startswith("cuda:") else (0 if device == "cuda" else -1)
            update_config(config, types.SimpleNamespace(experiment="trufor_ph3", gpu=gpu,
                                                        opts=["TEST.MODEL_FILE", str(CKPT)]))
        finally:
            os.chdir(cwd)
        if device != "cpu":
            torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK
            torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
            torch.backends.cudnn.enabled = config.CUDNN.ENABLED
        self.cudnn = {"benchmark": bool(config.CUDNN.BENCHMARK), "deterministic": bool(config.CUDNN.DETERMINISTIC)}
        self.TestDataset = TestDataset
        state = torch.load(str(CKPT), map_location=torch.device(device), weights_only=False) \
            if "weights_only" in torch.load.__code__.co_varnames else torch.load(str(CKPT), map_location=torch.device(device))
        self.model = get_model(config)
        self.model.load_state_dict(state["state_dict"], strict=True)
        self.model = self.model.to(device).eval()

    def modules(self):
        return {"TruFor (SegFormer/CMX + Noiseprint++)": self.model}

    @torch.no_grad()
    def predict(self, path: Path, sw) -> dict:
        with sw("preprocess"):
            rgb, _ = self.TestDataset(list_img=[str(path)])[0]
            rgb = rgb.unsqueeze(0).to(self.device)
        with sw("forward"):
            pred, conf, det, _ = self.model(rgb, save_np=False)
        with sw("postprocess"):
            score = float(torch.sigmoid(det).item())
            prob = F.softmax(torch.squeeze(pred, 0), dim=0)[1].cpu().numpy().astype(np.float32)
            conf_mean = float(torch.sigmoid(torch.squeeze(conf, 0))[0].mean().item())
        return {"score": score, "map": prob, "pred": prob > 0.5,
                "extra": {"confidence_mean": conf_mean, "cudnn": self.cudnn}}
