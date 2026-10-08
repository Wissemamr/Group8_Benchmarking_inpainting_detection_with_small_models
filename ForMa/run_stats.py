"""Per-invocation timing artifacts; optional Python call profiling."""
from contextlib import contextmanager
import cProfile
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import pstats
import time
import uuid

import numpy as np


class RunStats:
    def __init__(self, directory, metadata, profile=False):
        self.directory = Path(directory) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8])
        self.directory.mkdir(parents=True, exist_ok=False)
        self.info = dict(metadata, run_id=self.directory.name, started_at_utc=datetime.now(timezone.utc).isoformat(),
                         status='running', profiled=profile)
        self.stages = {}
        self.rows = []
        self.rss = []
        self.process = None
        try:
            import psutil
            self.process = psutil.Process()
        except ImportError:
            pass
        self.profiler = cProfile.Profile() if profile else None

    def sample_memory(self):
        if self.process is not None:
            try:
                self.rss.append(self.process.memory_info().rss)
            except OSError:
                pass

    def __enter__(self):
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.sample_memory()
        if self.profiler:
            self.profiler.enable()
        return self

    @contextmanager
    def phase(self, name):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.stages.setdefault(name, []).append(time.perf_counter() - started)
            self.sample_memory()

    def record_image(self, row):
        self.rows.append(row)
        path = self.directory / 'images.csv'
        with path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if len(self.rows) == 1:
                writer.writeheader()
            writer.writerow(row)

    def __exit__(self, exc_type, exc, traceback):
        if self.profiler:
            self.profiler.disable()
        elapsed = time.perf_counter() - self.start
        cpu = time.process_time() - self.cpu_start
        self.sample_memory()
        versions = {}
        for package in ['torch', 'numpy', 'pandas', 'Pillow', 'opencv-python', 'opencv-python-headless', 'psutil']:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        summaries = {}
        for name, values in self.stages.items():
            summaries[name] = dict(count=len(values), total_seconds=sum(values), mean_seconds=float(np.mean(values)),
                                   median_seconds=float(np.median(values)), p95_seconds=float(np.percentile(values, 95)),
                                   min_seconds=min(values), max_seconds=max(values))
        forward = self.stages.get('forward', [])
        self.info.update(status='failed' if exc_type else 'completed', error=str(exc) if exc_type else None,
                         completed_at_utc=datetime.now(timezone.utc).isoformat(), wall_seconds=elapsed,
                         process_cpu_seconds=cpu, newly_saved_images=len(self.rows),
                         saved_images_per_wall_second=len(self.rows) / elapsed if self.rows else None,
                         forward_only_images_per_second=len(forward) / sum(forward) if forward and sum(forward) else None,
                         max_sampled_rss_bytes=max(self.rss) if self.rss else None,
                         stages=summaries, python=platform.python_version(), platform=platform.platform(),
                         processor=platform.processor(), logical_cpu_count=os.cpu_count(), packages=versions,
                         timing_notes='Invocation excludes interpreter/import startup and report export. Stages are non-overlapping; other orchestration remains in wall time. Throughput counts newly saved images only. Forward statistics are across images, not repeated same-image latency trials.',
                         memory_notes='RSS sampled at phase boundaries, not a continuous peak; includes the Python process, model and workbook. Null if psutil unavailable. CUDA allocator peaks, when present, cover the measured loop after warmup and exclude non-PyTorch allocations.',
                         profile_notes='cProfile measures Python calls and adds overhead; it is not a GPU kernel trace or a clean latency benchmark.')
        if self.profiler:
            self.profiler.dump_stats(str(self.directory / 'profile.pstats'))
            with (self.directory / 'profile.txt').open('w', encoding='utf-8') as stream:
                pstats.Stats(self.profiler, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(80)
        (self.directory / 'run.json').write_text(json.dumps(self.info, indent=2, allow_nan=False), encoding='utf-8')
        print(f'Run report: {self.directory / "run.json"}', flush=True)
        if forward:
            summary = summaries['forward']
            print(f'Forward: mean {summary["mean_seconds"]:.3f}s | median {summary["median_seconds"]:.3f}s | '
                  f'p95 {summary["p95_seconds"]:.3f}s. Newly saved: {len(self.rows)}; wall time: {elapsed:.1f}s.', flush=True)
        return False
