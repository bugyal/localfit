"""Local hardware probing, and ingestion of the user's own benchmark logs.

Probing is best-effort by design and says what it could not determine rather than
guessing. Three families are covered because they are the three that show up in
these threads, and each fails differently:

- **NVIDIA**: ``nvidia-smi`` is authoritative when present. Absent, no NVIDIA card.
- **AMD**: ``rocm-smi`` / ``rocminfo``. Absent on a machine that clearly has an AMD
  card usually means the driver stack is not installed, which is itself worth
  reporting because it predicts which engine will work.
- **Apple Silicon**: ``sysctl``. Unified memory means the distinction between VRAM
  and RAM collapses, so the probe reports the shared pool and says so.

The point of probing is to save the user typing, not to be clever. Anything it
cannot resolve becomes an explicit question in the report rather than a silent
default, because a fit plan computed from a wrong VRAM figure is worse than no
plan.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .hardware import Gpu, GPUS, resolve as resolve_gpu


@dataclass
class ProbeResult:
    """What the machine actually is, as far as we could tell."""

    gpus: list[Gpu] = field(default_factory=list)
    system_ram_gb: float = 0.0
    platform: str = ""
    cuda_version: str = ""
    rocm_version: str = ""
    notes: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    """Things that could not be determined, and therefore must be confirmed."""

    def to_dict(self) -> dict:
        return {
            "platform": self.platform,
            "system_ram_gb": round(self.system_ram_gb, 1),
            "cuda_version": self.cuda_version,
            "rocm_version": self.rocm_version,
            "gpus": [
                {
                    "key": g.key,
                    "name": g.name,
                    "arch": g.arch,
                    "vram_gb": g.vram_gb,
                    "bandwidth_gbs": g.bandwidth_gbs,
                    "unified_memory": g.unified_memory,
                }
                for g in self.gpus
            ],
            "notes": self.notes,
            "unknown": self.unknown,
        }


def _run(cmd: list[str], timeout: int = 20) -> str:
    """Run a probe command, returning stdout or empty on any failure.

    Probes are allowed to fail: a missing tool is information, not an error.
    """
    if not shutil.which(cmd[0]):
        return ""
    try:
        done = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout or ""


# Maps a substring of the reported device name onto a catalogue key. Ordered: the
# more specific patterns come first so "4060 Ti" does not match "4060".
_NAME_PATTERNS: tuple[tuple[str, str], ...] = (
    ("4060 ti", "rtx-4060-ti-16g"),
    ("4060", "rtx-4060-8g"),
    ("3060", "rtx-3060-12g"),
    ("3080", "rtx-3080-10g"),
    ("3090", "rtx-3090-24g"),
    ("4090", "rtx-4090-24g"),
    ("5090", "rtx-5090-32g"),
    ("1660", "gtx-1660-ti-6g"),
    ("t4", "tesla-t4-16g"),
    ("v100", "tesla-v100-32g"),
    ("p40", "tesla-p40-24g"),
    ("p4", "tesla-p4-8g"),
    ("7900 xtx", "rx-7900-xtx-24g"),
    ("7900", "rx-7900-xtx-24g"),
    ("9070", "rx-9070-xt-16g"),
    ("6800", "rx-6800-xt-16g"),
    ("6700", "rx-6700-xt-12g"),
    ("rx 580", "rx-580-8g"),
    ("a770", "arc-a770-16g"),
    ("b580", "arc-b580-12g"),
)


def match_gpu_by_name(name: str, vram_gb: float = 0.0) -> Gpu | None:
    """Find a catalogue entry for a reported device name.

    Falls back to a synthetic entry when the name is recognised as a GPU but is
    not in the catalogue: an unknown-but-real card with a correct VRAM figure
    yields a useful plan, whereas refusing to plan at all helps nobody.
    """
    lowered = (name or "").lower()
    for pattern, key in _NAME_PATTERNS:
        if pattern in lowered:
            gpu = GPUS[key]
            # A 12 GB card reporting as a "4060" is a 3060 12GB or similar; the
            # VRAM figure is more trustworthy than the substring.
            if vram_gb and abs(gpu.vram_gb - vram_gb) > 2.0:
                return _synthetic(name, vram_gb, gpu)
            return gpu
    if vram_gb > 0:
        return _synthetic(name, vram_gb)
    return None


def _synthetic(name: str, vram_gb: float, like: Gpu) -> Gpu:
    """A catalogue-shaped entry for a card we can measure but not identify."""
    return Gpu(
        key=f"detected-{abs(hash(name)) % 100000}-{int(vram_gb)}g",
        name=f"{name.strip()} ({vram_gb:.0f} GB, unlisted)",
        arch=like.arch,
        vram_gb=float(vram_gb),
        # Bandwidth is the one figure we cannot measure without running a
        # benchmark, so it is inherited from the closest known architecture and
        # flagged as an assumption rather than presented as fact.
        bandwidth_gbs=like.bandwidth_gbs,
        tensor_cores=like.tensor_cores,
        bf16_native=like.bf16_native,
        fp16_native=like.fp16_native,
        int8_dp4a=like.int8_dp4a,
        unified_memory=like.unified_memory,
        note=(
            "Not in the catalogue. Architecture and bandwidth are inherited from "
            f"the closest known part ({like.name}); override with --bandwidth if "
            "you know the real figure."
        ),
    )


def probe(allow_commands: bool = True) -> ProbeResult:
    """Detect what this machine has."""
    result = ProbeResult()
    import platform as _platform

    result.platform = _platform.system()

    if result.platform == "Darwin":
        result.system_ram_gb = _sysctl_int("hw.memsize") / 1e9
        chip = _sysctl_string("machdep.cpu.brand_string") or ""
        result.notes.append(f"Apple platform, chip reported as {chip or 'unknown'}.")
        apple = _apple_gpu(chip, result.system_ram_gb)
        if apple:
            result.gpus.append(apple)
        else:
            result.unknown.append(
                "Could not map this Apple chip to a catalogue entry; pass --gpu "
                "explicitly (see `localfit hardware`)."
            )
        return result

    # ---- system RAM, for hybrid plans ------------------------------------
    if allow_commands:
        meminfo = _run(["cat", "/proc/meminfo"])
        if meminfo:
            match = re.search(r"MemTotal:\s+(\d+)", meminfo)
            if match:
                result.system_ram_gb = int(match.group(1)) * 1024 / 1e9

    # ---- NVIDIA ----------------------------------------------------------
    smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    if smi:
        for line in smi.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            name = parts[0]
            try:
                vram = float(parts[1]) / 1024.0
            except ValueError:
                continue
            gpu = match_gpu_by_name(name, vram)
            if gpu:
                result.gpus.append(gpu)
            else:
                result.unknown.append(f"NVIDIA device {name!r} ({vram:.0f} GB) unrecognised.")
        cuda = _run(["nvcc", "--version"])
        if cuda:
            match = re.search(r"release (\d+\.\d+)", cuda)
            if match:
                result.cuda_version = match.group(1)
        else:
            result.notes.append(
                "nvidia-smi found but nvcc is not on PATH: the runtime driver is "
                "installed, the CUDA toolkit is not. llama.cpp prebuilt CUDA "
                "binaries need the driver only; building from source needs both."
            )
    else:
        result.notes.append(
            "nvidia-smi not found, so no NVIDIA card was detected. If you have "
            "one, the driver is missing."
        )

    # ---- AMD -------------------------------------------------------------
    rocm = _run(["rocm-smi", "--showproductname", "--showmeminfo", "vram", "--csv"])
    if rocm:
        result.gpus.extend(_parse_rocm_smi(rocm, result))
        version = _run(["rocminfo"])
        if version:
            match = re.search(r"ROCm\s+(\d+\.\d+)", version)
            result.rocm_version = match.group(1) if match else "detected"
    else:
        result.notes.append(
            "rocm-smi not found. On an AMD card this usually means ROCm is not "
            "installed; Vulkan remains an option and behaves differently."
        )

    if not result.gpus:
        result.unknown.append(
            "No accelerator was detected. Pass --gpu explicitly, or use "
            "`localfit hardware` to see the catalogue keys."
        )
    return result


def _parse_rocm_smi(text: str, result: ProbeResult) -> list[Gpu]:
    """Read the CSV emitted by rocm-smi.

    The column layout varies between ROCm releases, so this looks for the VRAM
    figure rather than trusting a fixed position.
    """
    found: list[Gpu] = []
    for line in text.strip().splitlines():
        lowered = line.lower()
        if "card" not in lowered and "gpu" not in lowered:
            continue
        name_match = re.search(r"card\d+[,\s]+(.+?)(?:,|$)", line)
        name = name_match.group(1).strip() if name_match else ""
        vram_match = re.search(r"(\d{3,6})\s*(?:MB|mb)", line)
        vram_gb = int(vram_match.group(1)) / 1024.0 if vram_match else 0.0
        if not name or not vram_gb:
            continue
        gpu = match_gpu_by_name(name, vram_gb)
        if gpu:
            found.append(gpu)
        else:
            result.unknown.append(f"AMD device {name!r} ({vram_gb:.0f} GB) unrecognised.")
    return found


def _apple_gpu(chip: str, ram_gb: float) -> Gpu | None:
    """Map an Apple chip string onto the catalogue."""
    lowered = (chip or "").lower()
    if "m3 ultra" in lowered:
        return GPUS["apple-m3-ultra-192g"]
    if "m4 max" in lowered:
        return GPUS["apple-m4-max-64g"]
    if "m4 pro" in lowered:
        return GPUS["apple-m4-pro-24g"]
    if lowered.startswith("apple m4") or "m4" in lowered:
        return GPUS["apple-m4-16g"]
    if "m2 max" in lowered:
        return GPUS["apple-m2-max-64g"]
    if "m1" in lowered:
        return GPUS["apple-m1-16g"]
    return None


def _sysctl_int(key: str) -> float:
    out = _run(["sysctl", "-n", key])
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def _sysctl_string(key: str) -> str:
    return _run(["sysctl", "-n", key]).strip()


# ---------------------------------------------------------------------------
# Ingestion of the user's own logs: the 'measure' path
# ---------------------------------------------------------------------------

# llama.cpp prints its timing summary to stderr at the end of a run, and the
# numbers people quote in threads come from here. Patterns are permissive because
# the format has changed across releases.
TIMING_PATTERNS = {
    "eval_tok_s": re.compile(r"eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) runs?.*?([\d.]+)\s*tokens per second", re.S),
    "prompt_tok_s": re.compile(r"prompt eval time\s*=\s*[\d.]+ ms\s*/\s*(\d+) tokens.*?([\d.]+)\s*tokens per second", re.S),
    "tokens": re.compile(r"eval time.*?/\s*(\d+) runs?", re.S),
}

# llama.cpp also logs what it actually loaded, which is better evidence than the
# flags the user believes they passed.
LOAD_PATTERNS = {
    "n_gpu_layers": re.compile(r"n_gpu_layers\s*=\s*(-?\d+)"),
    "layers_offloaded": re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU"),
    "tensor_split": re.compile(r"tensor_split\s*=\s*([\d.,\s]+)"),
    "main_gpu": re.compile(r"main_gpu\s*=\s*(\d+)"),
    "context": re.compile(r"n_ctx\s*=\s*(\d+)"),
    "model_path": re.compile(r"load_model:\s*loading model '([^']+)'"),
    "kv_cache": re.compile(r"KV self size\s*=\s*([\d.]+)\s*MiB"),
}

# The per-device memory report is how you see the imbalance from the log rather
# than from a memory monitor.
DEVICE_MEMORY = re.compile(
    r"(CUDA|ROCm|Vulkan|Metal|SYCL)(\d+)\s+model buffer size\s*=\s*([\d.]+)\s*MiB"
)
DEVICE_TOTAL = re.compile(
    r"ggml_(CUDA|ROCm|Vulkan|Metal|SYCL)\d*\s*.*?total size\s*=\s*([\d.]+)\s*MiB"
)


@dataclass
class Measurement:
    """A row read out of one of the user's runs."""

    source: str = ""
    model_path: str = ""
    eval_tok_s: float = 0.0
    prompt_tok_s: float = 0.0
    layers_offloaded: int = 0
    layers_total: int = 0
    tensor_split: str = ""
    main_gpu: str = ""
    context: int = 0
    kv_cache_mib: float = 0.0
    per_device_mib: list[tuple[str, float]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_useful(self) -> bool:
        return self.eval_tok_s > 0 or self.per_device_mib

    def to_dict(self) -> dict:
        return {
            "model_path": self.model_path,
            "eval_tok_s": self.eval_tok_s,
            "prompt_tok_s": self.prompt_tok_s,
            "layers_offloaded": self.layers_offloaded,
            "layers_total": self.layers_total,
            "tensor_split": self.tensor_split,
            "main_gpu": self.main_gpu,
            "context": self.context,
            "kv_cache_mib": round(self.kv_cache_mib, 1),
            "per_device_mib": [[n, round(v, 1)] for n, v in self.per_device_mib],
            "warnings": self.warnings,
        }


def parse_log(text: str, source: str = "") -> Measurement:
    """Extract a measurement from llama.cpp or vLLM output.

    Deliberately tolerant: this reads logs produced by tools whose output format
    is not a stable interface, so a partial read that yields the tok/s figure is
    still valuable, and anything missing is reported as missing.
    """
    result = Measurement(source=source)

    # llama.cpp: "eval time = 1234.56 ms /   128 runs   (  9.64 ms per token, 103.68 tokens per second)"
    match = re.search(
        r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+runs.*?([\d.]+)\s*tokens per second",
        text,
        re.S,
    )
    if match:
        result.eval_tok_s = float(match.group(3))
    else:
        # vLLM and some wrappers print a plain throughput line.
        alt = re.search(r"(\d+\.?\d*)\s*(?:tok/s|tokens/s|tokens per second)", text)
        if alt:
            result.eval_tok_s = float(alt.group(1))

    match = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s+tokens.*?([\d.]+)\s*tokens per second",
        text,
        re.S,
    )
    if match:
        result.prompt_tok_s = float(match.group(3))

    for name, pattern in LOAD_PATTERNS.items():
        found = pattern.search(text)
        if not found:
            continue
        raw = found.group(1)
        if name == "layers_offloaded":
            result.layers_offloaded = int(found.group(1))
            result.layers_total = int(found.group(2))
        elif name == "n_gpu_layers" and not result.layers_total:
            value = int(raw)
            result.layers_offloaded = value
        elif name == "tensor_split":
            result.tensor_split = raw.strip().rstrip(",")
        elif name == "main_gpu":
            result.main_gpu = raw
        elif name == "context":
            result.context = int(raw)
        elif name == "model_path":
            result.model_path = raw
        elif name == "kv_cache":
            result.kv_cache_mib = float(raw)

    for match in DEVICE_MEMORY.finditer(text):
        backend, index, mib = match.group(1), match.group(2), float(match.group(3))
        result.per_device_mib.append((f"{backend}{index}", mib))

    if result.layers_total and result.layers_offloaded < result.layers_total:
        result.warnings.append(
            f"Only {result.layers_offloaded}/{result.layers_total} layers were "
            "offloaded: part of this run was CPU-bound, so the tok/s figure "
            "understates what this card can do on a smaller model."
        )
    if result.eval_tok_s <= 0:
        result.warnings.append(
            "No tokens-per-second figure found in the log. llama.cpp prints its "
            "timing summary to stderr at the end of a run — capture that too."
        )
    if len(result.per_device_mib) >= 2:
        values = [v for _, v in result.per_device_mib]
        spread = max(values) - min(values)
        if spread > 512:
            result.warnings.append(
                f"Device buffers differ by {spread:.0f} MiB. That is the imbalance "
                "this tool diagnoses: see the multi-GPU section of the report."
            )
    return result


def parse_log_file(path: str | Path) -> Measurement:
    text = Path(path).read_text(errors="ignore")
    return parse_log(text, source=str(path))


def contribute_row(measurement: Measurement, out_path: str | Path) -> dict:
    """Format a measurement as a contribution row.

    The output is a plain JSON object the user can post or keep. Nothing is
    uploaded anywhere — this tool has no network path, which is the point: the
    only way a row leaves the machine is if the person reading it chooses to send
    it.
    """
    row = measurement.to_dict()
    row["schema"] = "localfit.measurement/1"
    Path(out_path).write_text(json.dumps(row, indent=2) + "\n")
    return row


def summarise_probe(result: ProbeResult) -> str:
    """A short human summary of what was detected."""
    if not result.gpus:
        return "no accelerator detected"
    counts: dict[str, int] = {}
    for gpu in result.gpus:
        counts[gpu.name] = counts.get(gpu.name, 0) + 1
    parts = [f"{n}x {name}" if n > 1 else name for name, n in counts.items()]
    return ", ".join(parts)
