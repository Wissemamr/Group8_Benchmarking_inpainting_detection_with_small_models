"""Image and mask I/O. Tensors follow AEROBLADE's ImageFolder: RGB, float32 in [0, 1]."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as tf
from PIL import Image
from torchvision.transforms.v2.functional import to_pil_image

# AEROBLADE's ImageFolder default transform.
_TO_TENSOR = tf.Compose([tf.ToImage(), tf.ToDtype(torch.float32, scale=True)])


def load_image(path: Path, image_size: int | str, box: Optional[Sequence[int]] = None) -> Image.Image:
    """RGB image: cropped to `box` for 'crop', unchanged for 'native', else resized to
    image_size x image_size (bicubic)."""
    img = Image.open(path).convert("RGB")
    if image_size == "crop":
        img = img.crop(tuple(box))
    elif image_size != "native":
        img = img.resize((image_size, image_size), Image.BICUBIC)
    return img


def to_tensor(img: Image.Image) -> torch.Tensor:
    return _TO_TENSOR(img).unsqueeze(0)


def pad_to_multiple(x: torch.Tensor, multiple: int = 8) -> torch.Tensor:
    """Reflect-pad bottom/right so H and W are multiples of `multiple` (the AE's stride)."""
    h, w = x.shape[-2:]
    pad_h, pad_w = -h % multiple, -w % multiple
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x


def save_png(x: torch.Tensor, path: Path) -> None:
    """Save a (1, 3, H, W) tensor in [0, 1] as 8-bit PNG, the way AEROBLADE stores reconstructions.
    Written via a temporary file so an interrupted run never leaves a truncated cache entry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    to_pil_image(x[0]).save(tmp)
    tmp.replace(path)


def save_npy(array: np.ndarray, path: Path) -> None:
    """np.save via a temporary file (see save_png)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.npy")
    np.save(tmp, array)
    tmp.replace(path)


def load_mask(
    path: Path, size_wh: tuple[int, int], threshold: int, box: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Boolean GT mask (True = inpainted), cropped to `box` if given, at `size_wh`
    (nearest-neighbour resized if needed)."""
    mask = Image.open(path).convert("L")
    if box is not None:
        mask = mask.crop(tuple(box))
    if mask.size != size_wh:
        mask = mask.resize(size_wh, Image.NEAREST)
    return np.asarray(mask) > threshold
