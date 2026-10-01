"""Single-image ForMa inference with optional localization evaluation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch

if __package__:
    from .metrics import binary_mask, localization_metrics
else:
    from metrics import binary_mask, localization_metrics

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / 'ForMa_weights' / 'ForMa_weights.pth'


def load_model(weights_path=DEFAULT_WEIGHTS, device='cpu'):
    """Load a trusted full checkpoint; fail on missing or mismatched weights."""
    if __package__:
        from .models.vmamba_pixelshuf_modals import Forensic_Vmamba
    else:
        from models.vmamba_pixelshuf_modals import Forensic_Vmamba
    model = Forensic_Vmamba(noiseprint_path=None)
    # Full checkpoint includes Noiseprint and optimizer metadata. PyTorch 1.13's
    # weights_only loader cannot read it. Only use trusted checkpoint files.
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    state = checkpoint.get('model_state_dict', checkpoint)
    if state and all(k.startswith('module.') for k in state):
        state = {k[7:]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def preprocess_image(image_path, input_size=512):
    """Upstream preprocessing: OpenCV RGB, linear resize, float32 / 255."""
    if input_size < 32 or input_size % 32:
        raise ValueError('Input size must be a positive multiple of 32.')
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode image: {image_path}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(resized.astype(np.float32) / 255).permute(2, 0, 1)
    return tensor.unsqueeze(0), Image.fromarray(image)


@torch.inference_mode()
def run_inference(model, image_tensor):
    output = model(image_tensor)
    if output.ndim != 4 or output.shape[:2] != (1, 2):
        raise ValueError(f'Expected [1, 2, H, W] logits, got {tuple(output.shape)}')
    if not torch.isfinite(output).all():
        raise ValueError('Model returned non-finite logits.')
    return output


def prediction_maps(output, size, threshold=0.5):
    """Class-1 softmax, threshold, nearest resize; ties select background.

    Nearest-resized float probabilities reproduce the saved binary mask.
    At threshold 0.5 this matches upstream argmax then nearest resize.
    """
    if not 0 <= threshold <= 1:
        raise ValueError('Threshold must be between 0 and 1.')
    probability = output.softmax(dim=1)[0, 1].cpu().numpy()
    probability = np.asarray(Image.fromarray(probability).resize(size, Image.Resampling.NEAREST))
    return probability, probability > threshold


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image', type=Path, required=True)
    parser.add_argument('--mask', type=Path, help='Binary GT: white/1 = manipulated.')
    parser.add_argument('--weights', type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results' / 'single_image')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--input-size', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args(argv)
    for path in (args.image, args.weights, args.mask):
        if path is not None and not path.is_file():
            parser.error(f'File not found: {path}')
    if not 0 <= args.threshold <= 1 or args.threads < 1:
        parser.error('Threshold must be in [0, 1]; threads must be positive.')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu')
                          if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable; use --device cpu.')
    torch.set_num_threads(args.threads)
    tensor, image = preprocess_image(args.image, args.input_size)
    target = None
    if args.mask:
        with Image.open(args.mask) as mask_image:
            target = binary_mask(np.asarray(mask_image.convert('L')))
        if target.shape != (image.height, image.width):
            parser.error('Ground-truth dimensions must match the input image.')
    print(f'Loading checkpoint on {device}...', flush=True)
    model = load_model(args.weights, device)
    print('All checkpoint parameters loaded. Running inference...', flush=True)
    tensor = tensor.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = run_inference(model, tensor)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    probability, prediction = prediction_maps(output, image.size, args.threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(prediction.astype(np.uint8) * 255).save(args.output_dir / 'prediction.png')
    np.save(args.output_dir / 'probabilities.npy', probability)
    Image.fromarray(np.round(probability * 255).astype(np.uint8)).save(args.output_dir / 'probabilities.png')
    overlay = np.asarray(image).copy()
    overlay[prediction] = (0.55 * overlay[prediction] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    Image.fromarray(overlay).save(args.output_dir / 'overlay.png')
    report = dict(
        image=str(args.image.resolve()), image_sha256=sha256(args.image),
        mask=str(args.mask.resolve()) if args.mask else None,
        mask_sha256=sha256(args.mask) if args.mask else None,
        weights=str(args.weights.resolve()), weights_sha256=sha256(args.weights),
        device=str(device), torch_version=torch.__version__, numpy_version=np.__version__,
        opencv_version=cv2.__version__, threads=args.threads, input_size=args.input_size,
        original_size=list(image.size), output_shape=list(output.shape),
        threshold=args.threshold, foreground_class=1,
        preprocessing='RGB; OpenCV INTER_LINEAR square resize; float32 / 255',
        postprocessing='class-1 softmax > threshold; nearest resize to original resolution',
        forward_seconds=elapsed, timing_note='Single cold forward; excludes loading, preprocessing and saving.',
        parameters=sum(p.numel() for p in model.parameters()),
        metrics=localization_metrics(prediction, target) if target is not None else None,
    )
    from importlib import import_module
    prefix = f'{__package__}.' if __package__ else ''
    scan = import_module(prefix + 'models.csms6s')
    cross = import_module(prefix + 'models.csm_triton')
    report['available_backends'] = dict(triton=cross.WITH_TRITON,
        selective_scan_oflex=scan.WITH_SELECTIVESCAN_OFLEX,
        selective_scan_core=scan.WITH_SELECTIVESCAN_CORE,
        selective_scan_mamba=scan.WITH_SELECTIVESCAN_MAMBA)
    (args.output_dir / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2) if target is not None else 'No ground truth supplied; localization scores not computed.')
    print(f'Forward: {elapsed:.2f}s. Results: {args.output_dir.resolve()}')
    return report


if __name__ == '__main__':
    main()
