"""AEROBLADE, exactly as in AEROBLADE/ (the group benchmark): SD2 autoencoder, image resized to
512x512 (bicubic), reconstruction with latent seed 1, 8-bit PNG round trip, LPIPS (VGG layer 2).

Detection score  -AEROBLADE distance (spatial mean of the unpadded layer-2 map; higher = generated).
                 Predicted "inpainted" when score > -0.0521, the cross-fitted Youden threshold of the
                 full TGIF benchmark run (AEROBLADE has no threshold of its own).
Localization     reflect-padded layer-2 map, upsampled to the original size, normalized per image
                 (1st-99th percentile), inverted (low error = inpainted) and binarized with Otsu.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "AEROBLADE" / "src"))

from aeroblade_bench import imaging  # noqa: E402
from aeroblade_bench.autoencoders import load_autoencoder, reconstruct  # noqa: E402
from aeroblade_bench.lpips_map import LPIPSLayerMap, upsample  # noqa: E402
from aeroblade_bench.metrics import normalize, otsu_threshold  # noqa: E402
from torchvision.transforms.v2.functional import to_pil_image  # noqa: E402

DET_THRESHOLD = -0.0521


class Model:
    name = "AEROBLADE"
    protocol = ("SD2 VAE reconstruction at 512x512 + LPIPS layer 2; score = -mean distance; "
                "map = 1 - percentile-normalized error, Otsu")
    score_kind = "native (-reconstruction distance)"
    det_threshold = DET_THRESHOLD

    def __init__(self, device: str, opts: dict):
        self.device = device
        self.size = 512
        self.ae = load_autoencoder("SD2", device)
        self.lpips = LPIPSLayerMap(2, pad=16).to(device)

    def modules(self):
        return {"SD2 autoencoder": self.ae, "LPIPS VGG16 (to layer 2)": self.lpips}

    @torch.inference_mode()
    def predict(self, path: Path, sw) -> dict:
        with sw("preprocess"):
            img = Image.open(path).convert("RGB")
            w, h = img.size
            # plain tensor: torchvision's Image subclass breaks the FLOP counter
            x = imaging.to_tensor(img.resize((self.size, self.size), Image.BICUBIC)).as_subclass(torch.Tensor)
            x = x.to(self.device)
        with sw("forward_autoencoder"):
            rec = reconstruct(self.ae, imaging.pad_to_multiple(x), 1)[..., :self.size, :self.size]
        with sw("png_round_trip"):
            rec = imaging.to_tensor(to_pil_image(rec[0].cpu())).as_subclass(torch.Tensor).to(self.device)
        with sw("forward_lpips"):
            dist = float(self.lpips(x, rec, pad=0).mean())
            err = self.lpips(x, rec, pad=16)[0, 0].cpu().numpy()
        with sw("postprocess"):
            err = upsample(err, (h, w))
            score_map = 1.0 - normalize(err, "percentile", (1.0, 99.0))
            pred = score_map > otsu_threshold(score_map)
        return {"score": -dist, "map": score_map, "pred": pred, "extra": {"aeroblade_distance": dist}}
