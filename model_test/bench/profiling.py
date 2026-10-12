"""Resource measurement shared by every model runner.

Works in any of the model environments (Python 3.7+, any PyTorch from 1.8). Everything optional
degrades to "unavailable" instead of failing:
  * process memory      psutil if installed, else the OS API (Windows psapi / Linux /proc)
  * GPU sampling        `nvidia-smi` polled every 100 ms in a thread: utilization, memory used by
                        all processes, power, SM clock, temperature (energy = integrated power)
  * PyTorch GPU memory  torch.cuda peak allocated / reserved (this process only)
  * FLOPs               torch.utils.flop_counter (PyTorch >= 2.1, counts 2 FLOPs per multiply-add
                        of matmuls and convolutions); per-model fallbacks are up to the adapter
  * kernel profile      torch.profiler: per-operator CPU / CUDA time, calls, memory, FLOPs
"""

from __future__ import annotations

import csv
import ctypes
import gzip
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch


# ---------------------------------------------------------------- process memory

def _mem_windows():
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    ctypes.windll.psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.WorkingSetSize, c.PeakWorkingSetSize


def _mem_linux():
    cur = peak = 0
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                cur = int(line.split()[1]) * 1024
            elif line.startswith("VmHWM:"):
                peak = int(line.split()[1]) * 1024
    return cur, peak


def process_memory():
    """(current, peak since process start) resident memory in bytes; (None, None) if unknown."""
    try:
        if sys.platform == "win32":
            return _mem_windows()
        if sys.platform.startswith("linux"):
            return _mem_linux()
        import psutil  # macOS and others
        return psutil.Process().memory_info().rss, None
    except Exception:  # noqa: BLE001 - measurement must never break a run
        return None, None


class RSSSampler(threading.Thread):
    """Samples current resident memory every `interval` s; peak per phase via `peak_between`."""

    def __init__(self, interval: float = 0.05):
        super().__init__(daemon=True)
        self.interval, self.samples, self._halt = interval, [], threading.Event()

    def run(self):
        while not self._halt.is_set():
            cur, _ = process_memory()
            if cur is not None:
                self.samples.append((time.perf_counter(), cur))
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=2)

    def peak_between(self, t0: float, t1: float):
        vals = [v for t, v in self.samples if t0 <= t <= t1]
        return max(vals) if vals else None


# ---------------------------------------------------------------- GPU sampling (nvidia-smi)

SMI_FIELDS = ["utilization.gpu", "utilization.memory", "memory.used", "memory.total", "power.draw",
              "clocks.sm", "temperature.gpu"]


class GPUSampler(threading.Thread):
    """Polls nvidia-smi (all processes on the GPU, not only this one)."""

    def __init__(self, gpu_index: int = 0, interval_ms: int = 100):
        super().__init__(daemon=True)
        self.samples, self.error, self.proc = [], None, None
        self.cmd = ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader,nounits",
                    f"-lms={interval_ms}", f"-i={gpu_index}"]

    def run(self):
        if shutil.which("nvidia-smi") is None:
            self.error = "nvidia-smi not found"
            return
        try:
            self.proc = subprocess.Popen(self.cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         universal_newlines=True, bufsize=1)
            for line in self.proc.stdout:
                vals = [v.strip() for v in line.split(",")]
                if len(vals) != len(SMI_FIELDS):
                    continue
                row = {"t": time.perf_counter()}
                for k, v in zip(SMI_FIELDS, vals):
                    try:
                        row[k] = float(v)
                    except ValueError:
                        row[k] = None
                self.samples.append(row)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"

    def stop(self):
        if self.proc is not None:
            self.proc.terminate()
        self.join(timeout=3)

    def summary(self, t0: float, t1: float) -> dict:
        rows = [r for r in self.samples if t0 <= r["t"] <= t1]
        if not rows:
            return {"available": False, "reason": self.error or "no samples in this phase"}

        def col(k):
            return [r[k] for r in rows if r.get(k) is not None]

        def stat(k, f):
            v = col(k)
            return f(v) if v else None

        power = [(r["t"], r["power.draw"]) for r in rows if r.get("power.draw") is not None]
        energy = sum((t2 - t1_) * (p1 + p2) / 2 for (t1_, p1), (t2, p2) in zip(power, power[1:])) if len(power) > 1 else None
        return {"available": True, "n_samples": len(rows),
                "util_gpu_mean_pct": stat("utilization.gpu", lambda v: sum(v) / len(v)),
                "util_gpu_max_pct": stat("utilization.gpu", max),
                "util_mem_mean_pct": stat("utilization.memory", lambda v: sum(v) / len(v)),
                "vram_used_max_mib": stat("memory.used", max),
                "vram_total_mib": stat("memory.total", max),
                "power_mean_w": stat("power.draw", lambda v: sum(v) / len(v)),
                "power_max_w": stat("power.draw", max),
                "energy_j": energy,
                "sm_clock_mean_mhz": stat("clocks.sm", lambda v: sum(v) / len(v)),
                "temperature_max_c": stat("temperature.gpu", max)}

    def save(self, path: Path, t_origin: float) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s"] + SMI_FIELDS)
            for r in self.samples:
                w.writerow([round(r["t"] - t_origin, 3)] + [r.get(k) for k in SMI_FIELDS])


# ---------------------------------------------------------------- timing

def sync(device: str) -> None:
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


class Stopwatch:
    """`with sw("forward"):` accumulates wall time per stage, synchronizing CUDA at both ends."""

    def __init__(self, device: str):
        self.device, self.times = device, {}

    def __call__(self, name: str):
        sw = self

        class _Ctx:
            def __enter__(self):
                sync(sw.device)
                self.t = time.perf_counter()

            def __exit__(self, *exc):
                sync(sw.device)
                sw.times[name] = sw.times.get(name, 0.0) + time.perf_counter() - self.t
                return False

        return _Ctx()


# ---------------------------------------------------------------- FLOPs

class FlopRecorder:
    """Drop-in for Stopwatch: counts FLOPs inside every stage whose name starts with "forward"
    (pre- and post-processing are not model compute and are left out). PyTorch >= 2.1."""

    METHOD = "torch.utils.flop_counter (2 x multiply-adds of matmul / conv / attention, forward stages only)"

    def __init__(self):
        try:
            from torch.utils.flop_counter import FlopCounterMode
            self._mode, self.error = FlopCounterMode, None
        except ImportError:
            self._mode, self.error = None, f"unavailable (torch {torch.__version__} has no torch.utils.flop_counter)"
        self.by_stage, self.by_op = {}, {}

    def __call__(self, name: str):
        rec = self

        class _Ctx:
            def __enter__(self):
                self.fc = None
                if rec._mode is not None and name.startswith("forward") and rec.error is None:
                    self.fc = rec._mode(display=False)
                    self.fc.__enter__()

            def __exit__(self, *exc):
                if self.fc is not None:
                    self.fc.__exit__(*exc)
                    rec.by_stage[name] = rec.by_stage.get(name, 0) + int(self.fc.get_total_flops())
                    for module, ops in self.fc.get_flop_counts().items():
                        if module == "Global":
                            for op, n in ops.items():
                                k = str(op).split(".")[-1]
                                rec.by_op[k] = rec.by_op.get(k, 0) + int(n)
                return False

        return _Ctx()

    def result(self) -> dict:
        if self.error:
            return {"flops": None, "method": self.error}
        return {"flops": sum(self.by_stage.values()), "method": self.METHOD, "by_stage": dict(self.by_stage),
                "by_op": dict(self.by_op)}


def count_flops(predict) -> dict:
    """FLOPs of one prediction: `predict(recorder)` must route its stages through the recorder."""
    rec = FlopRecorder()
    try:
        predict(rec)
        return rec.result()
    except Exception as e:  # noqa: BLE001
        return {"flops": None, "method": f"failed: {type(e).__name__}: {str(e)[:200]}"}


# ---------------------------------------------------------------- kernel / operator profile

def _attr(evt, *names):
    for n in names:
        v = getattr(evt, n, None)
        if v is not None:
            return v
    return None


def torch_profile(fn, n_calls: int, device: str, out_dir: Path, trace: bool = True) -> dict:
    """Profiles n_calls of fn() with torch.profiler; writes an operator table (CSV + text) and a trace."""
    try:
        from torch.profiler import ProfilerActivity, profile, record_function
    except ImportError:
        return {"available": False, "reason": "torch.profiler unavailable"}
    acts = [ProfilerActivity.CPU]
    on_gpu = str(device).startswith("cuda")
    if on_gpu:
        acts.append(ProfilerActivity.CUDA)
    kwargs = dict(activities=acts, record_shapes=True, profile_memory=True)
    try:
        prof_ctx = profile(with_flops=True, **kwargs)
    except TypeError:
        prof_ctx = profile(**kwargs)
    with prof_ctx as prof:
        for i in range(n_calls):
            with record_function(f"image_{i}"):
                fn(i)
        sync(device)
    rows = []
    for e in prof.key_averages():
        rows.append({"op": e.key, "calls": e.count,
                     "cpu_total_ms": (e.cpu_time_total or 0) / 1e3,
                     "self_cpu_ms": (e.self_cpu_time_total or 0) / 1e3,
                     "gpu_total_ms": (_attr(e, "device_time_total", "cuda_time_total") or 0) / 1e3,
                     "self_gpu_ms": (_attr(e, "self_device_time_total", "self_cuda_time_total") or 0) / 1e3,
                     "self_gpu_mem_mb": (_attr(e, "self_device_memory_usage", "self_cuda_memory_usage") or 0) / 2**20,
                     "flops": _attr(e, "flops") or 0})
    key = "self_gpu_ms" if on_gpu else "self_cpu_ms"
    rows.sort(key=lambda r: -r[key])
    with open(out_dir / "torch_ops.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["op"])
        w.writeheader()
        w.writerows(rows)
    sort_by = "self_cuda_time_total" if on_gpu else "self_cpu_time_total"
    try:
        table = prof.key_averages().table(sort_by=sort_by, row_limit=40)
    except Exception:  # noqa: BLE001 - newer torch renamed the cuda keys
        table = prof.key_averages().table(sort_by="self_device_time_total" if on_gpu else sort_by, row_limit=40)
    (out_dir / "torch_ops.txt").write_text(table, encoding="utf-8")
    trace_file = None
    if trace:
        tmp = out_dir / "trace.json"
        prof.export_chrome_trace(str(tmp))
        trace_file = out_dir / "trace.json.gz"
        with open(tmp, "rb") as src, gzip.open(trace_file, "wb") as dst:
            shutil.copyfileobj(src, dst)
        tmp.unlink()
    total_gpu = sum(r["self_gpu_ms"] for r in rows)
    total_cpu = sum(r["self_cpu_ms"] for r in rows)
    return {"available": True, "n_calls": n_calls, "sorted_by": key,
            "self_gpu_ms_total": total_gpu, "self_cpu_ms_total": total_cpu,
            "top_ops": rows[:15], "trace": trace_file.name if trace_file else None}


# ---------------------------------------------------------------- environment

def environment() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
            "cpu": platform.processor(), "cpu_logical_cores": os.cpu_count(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "cuda_available": torch.cuda.is_available(), "conda_env": os.environ.get("CONDA_DEFAULT_ENV")}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu=p.name, gpu_memory_gb=round(p.total_memory / 2**30, 2),
                    gpu_compute_capability=f"{p.major}.{p.minor}", gpu_multiprocessors=p.multi_processor_count)
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,power.limit",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout
            info["nvidia_smi"] = out.strip()
        except Exception:  # noqa: BLE001
            pass
    try:
        if sys.platform == "win32":
            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong), ("c", ctypes.c_ulonglong),
                            ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            info["ram_gb"] = round(ms.ullTotalPhys / 2**30, 1)
        else:
            with open("/proc/meminfo") as f:
                info["ram_gb"] = round(int(f.readline().split()[1]) / 2**20, 1)
    except Exception:  # noqa: BLE001
        pass
    return info


def write_json(path: Path, obj) -> None:
    def default(o):
        try:
            return float(o)
        except Exception:  # noqa: BLE001
            return str(o)
    path.write_text(json.dumps(obj, indent=1, default=default), encoding="utf-8")
