"""Autoencoder loading and reconstruction.

Adapted from AEROBLADE, `compute_reconstructions` in aeroblade/src/aeroblade/image.py. Kept as is:
the three AEs, the fp16 weight variants AEROBLADE loads, input scaling to [-1, 1], sampling the
latent with `retrieve_latents`, `force_not_quantize=True` for the VQ model, and the clamp back to
[0, 1] (the caller saves an 8-bit PNG, as AEROBLADE does). Changed so it runs on CPU:
  * only the AE sub-model is loaded, not the whole img2img pipeline (no UNet/text-encoder
    download, no `enable_model_cpu_offload()`, which needs CUDA);
  * compute is fp32 (the fp16 weights are upcast);
  * each image gets a fresh generator seeded with `seed`, so its reconstruction does not depend
    on its position in the dataset (AEROBLADE threads one generator through its DataLoader).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.models import AutoencoderKL, VQModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (
    retrieve_latents,
)
from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class AESpec:
    repo_id: str
    subfolder: str
    variant: Optional[str]
    sha256: Optional[str] = None  # of the weights file; checked before loading when set

    @property
    def weights_file(self) -> str:
        suffix = f".{self.variant}" if self.variant else ""
        return f"{self.subfolder}/diffusion_pytorch_model{suffix}.safetensors"


# The AEs of AEROBLADE's scripts/run_aeroblade.py.
AUTOENCODERS = {
    "SD1": AESpec("CompVis/stable-diffusion-v1-1", "vae", "fp16"),
    # stabilityai/stable-diffusion-2-base is no longer downloadable (HTTP 401). This mirror's VAE
    # is byte-identical to independent 2023-2025 mirrors of it, and to the VAE of
    # stable-diffusion-2-inpainting, the model that generated TGIF's SD2 images.
    "SD2": AESpec(
        "sd2-community/stable-diffusion-2-base",
        "vae",
        "fp16",
        sha256="3e4c08995484ee61270175e9e7a072b66a6e4eeb5f0c266667fe1f45b90daf9a",
    ),
    "KD2.1": AESpec("kandinsky-community/kandinsky-2-1", "movq", None),
}


def _check_sha256(spec: AESpec) -> None:
    path = hf_hub_download(spec.repo_id, spec.weights_file)
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != spec.sha256:
        raise RuntimeError(
            f"{spec.repo_id}/{spec.weights_file}: sha256 {digest.hexdigest()} does not match the "
            f"expected {spec.sha256}; the mirror may have changed."
        )


def load_autoencoder(name: str, device: str = "cpu") -> torch.nn.Module:
    spec = AUTOENCODERS[name]
    if spec.sha256 is not None:
        _check_sha256(spec)
    cls = VQModel if spec.subfolder == "movq" else AutoencoderKL
    ae = cls.from_pretrained(
        spec.repo_id,
        subfolder=spec.subfolder,
        variant=spec.variant,
        use_safetensors=True,
        torch_dtype=torch.float32,
    )
    return ae.to(device=device, dtype=torch.float32).eval()


@torch.inference_mode()
def reconstruct(ae: torch.nn.Module, images: torch.Tensor, seed: int = 1) -> torch.Tensor:
    """AE reconstruction of (N, 3, H, W) images in [0, 1]; H and W must be multiples of 8."""
    generator = torch.Generator().manual_seed(seed)
    latents = retrieve_latents(ae.encode(images * 2.0 - 1.0), generator=generator)
    if isinstance(ae, VQModel):
        rec = ae.decode(latents, force_not_quantize=True, return_dict=False)[0]
    else:
        rec = ae.decode(latents, return_dict=False)[0]
    return (rec / 2 + 0.5).clamp(0, 1)
