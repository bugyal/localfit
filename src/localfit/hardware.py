"""The hardware catalogue.

Memory bandwidth is the number that matters. Decode speed is bandwidth-bound at
batch 1: producing a token requires streaming every active weight once, so
``bandwidth / bytes-per-token`` is a ceiling no scheduler can beat. A card's
bandwidth is therefore the single most predictive spec for how a local model will
feel, which is why the catalogue is organised around it rather than around
teraflops (which govern prefill, not the interactive experience).

Sources: vendor datasheets and specifications, as published for each part. Where
a vendor gives several bandwidth figures the marketing maximum is used, which
makes the tok/s ceiling optimistic — the measured anchors in ``perf.py`` are what
correct the estimate downward.

Consumer and legacy parts are included deliberately: they are the segment asking
these questions, and the segment vendor documentation is worst for. A card the
vendor has forgotten is exactly where community-measured data is the only data
that exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Architecture families, used for kernel caveats that are architectural rather
# than per-card. The P40 and the 7900 XTX ask opposite questions of llama.cpp,
# and the answer is a property of the family, not the card.
NV_PASCAL = "nvidia-pascal"
NV_TURING = "nvidia-turing"
NV_AMPERE = "nvidia-ampere"
NV_ADA = "nvidia-ada"
NV_BLACKWELL = "nvidia-blackwell"
NV_VOLTA = "nvidia-volta"
AMD_RDNA1 = "amd-rdna1"
AMD_RDNA2 = "amd-rdna2"
AMD_RDNA3 = "amd-rdna3"
AMD_RDNA4 = "amd-rdna4"
AMD_GCN = "amd-gcn"
INTEL_XE = "intel-xe"
INTEL_XE2 = "intel-xe2"
APPLE_M = "apple-silicon"
CPU_ONLY = "cpu"


@dataclass(frozen=True)
class Gpu:
    """One accelerator."""

    key: str
    name: str
    arch: str
    vram_gb: float
    bandwidth_gbs: float

    # Capabilities that change what runs well rather than whether it runs.
    tensor_cores: bool = False
    bf16_native: bool = False
    fp16_native: bool = True
    """Pascal emulates FP16 in FP32; that is a real 20% decode cost."""
    int8_dp4a: bool = True
    """dp4a is the INT8 dot-product instruction. Pascal lacks it, which is why
    IQ quants are historically slow there."""
    unified_memory: bool = False
    """Apple Silicon and some iGPUs share system RAM; offload is not a tradeoff."""

    host_ram_gb: float = 0.0
    """System memory, for the CPU-only entries where ``vram_gb`` is meaningless.

    Without this a CPU-only plan has zero capacity and reports every model as
    impossible, which is wrong: for a CPU plan the budget is system RAM.
    """

    note: str = ""
    sources: tuple[str, ...] = field(default_factory=tuple)


GPUS: dict[str, Gpu] = {
    g.key: g
    for g in (
        # ---- NVIDIA: datacenter and legacy --------------------------------
        Gpu(
            "tesla-p40-24g", "NVIDIA Tesla P40", NV_PASCAL, 24.0, 346.0,
            tensor_cores=False, bf16_native=False, fp16_native=False,
            int8_dp4a=False,
            note=(
                "24 GB for very little money, no tensor cores, no native FP16/BF16. "
                "The reference card for 'lots of VRAM, old silicon'."
            ),
            sources=("nvidia-p40-datasheet: 346 GB/s",),
        ),
        Gpu(
            "tesla-p4-8g", "NVIDIA Tesla P4", NV_PASCAL, 8.0, 192.0,
            tensor_cores=False, bf16_native=False, fp16_native=False,
            int8_dp4a=False,
            sources=("nvidia-p4-datasheet: 192 GB/s",),
        ),
        Gpu(
            "tesla-v100-32g", "NVIDIA Tesla V100", NV_VOLTA, 32.0, 900.0,
            tensor_cores=True, bf16_native=True,
            sources=("nvidia-v100-datasheet: 900 GB/s HBM2",),
        ),
        Gpu(
            "tesla-t4-16g", "NVIDIA Tesla T4", NV_TURING, 16.0, 320.0,
            tensor_cores=True, bf16_native=True,
            sources=("nvidia-t4-datasheet: 320 GB/s",),
        ),
        # ---- NVIDIA: consumer ---------------------------------------------
        Gpu("gtx-1660-ti-6g", "NVIDIA GTX 1660 Ti", NV_TURING, 6.0, 288.0,
            tensor_cores=False, bf16_native=False,
            sources=("vendor specs: 288 GB/s",)),
        Gpu("rtx-3060-12g", "NVIDIA RTX 3060 12GB", NV_AMPERE, 12.0, 360.0,
            tensor_cores=True, bf16_native=True,
            sources=("vendor specs: 360 GB/s",)),
        Gpu("rtx-3080-10g", "NVIDIA RTX 3080 10GB", NV_AMPERE, 10.0, 760.0,
            tensor_cores=True, bf16_native=True,
            sources=("vendor specs: 760 GB/s",)),
        Gpu("rtx-3090-24g", "NVIDIA RTX 3090", NV_AMPERE, 24.0, 936.0,
            tensor_cores=True, bf16_native=True,
            note="NVLink available on some SKUs, which changes multi-GPU behaviour.",
            sources=("vendor specs: 936 GB/s", "getdeploying: 936 GB/s")),
        Gpu("rtx-4060-8g", "NVIDIA RTX 4060 8GB", NV_ADA, 8.0, 272.0,
            tensor_cores=True, bf16_native=True,
            sources=("vendor specs: 272 GB/s",)),
        Gpu("rtx-4060-ti-16g", "NVIDIA RTX 4060 Ti 16GB", NV_ADA, 16.0, 288.0,
            tensor_cores=True, bf16_native=True,
            note="16 GB at a slow 288 GB/s — capacity without bandwidth.",
            sources=("vendor specs: 288 GB/s",)),
        Gpu("rtx-4090-24g", "NVIDIA RTX 4090", NV_ADA, 24.0, 1008.0,
            tensor_cores=True, bf16_native=True,
            sources=("vendor specs: 1008 GB/s",)),
        Gpu("rtx-5090-32g", "NVIDIA RTX 5090", NV_BLACKWELL, 32.0, 1792.0,
            tensor_cores=True, bf16_native=True,
            sources=("vendor specs: 1792 GB/s",)),
        # ---- AMD ----------------------------------------------------------
        Gpu("rx-580-8g", "AMD Radeon RX 580", AMD_GCN, 8.0, 256.0,
            tensor_cores=False, bf16_native=False, fp16_native=True,
            sources=("vendor specs: 256 GB/s",)),
        Gpu("rx-6700-xt-12g", "AMD Radeon RX 6700 XT", AMD_RDNA2, 12.0, 384.0,
            tensor_cores=False, bf16_native=True,
            sources=("vendor specs: 384 GB/s",)),
        Gpu("rx-6800-xt-16g", "AMD Radeon RX 6800 XT", AMD_RDNA2, 16.0, 512.0,
            tensor_cores=False, bf16_native=True,
            sources=("vendor specs: 512 GB/s",)),
        Gpu("rx-7900-xtx-24g", "AMD Radeon RX 7900 XTX", AMD_RDNA3, 24.0, 960.0,
            tensor_cores=False, bf16_native=True,
            note=(
                "Strong bandwidth for the price. ROCm support trails CUDA, and "
                "ROCm version gates which kernels are available."
            ),
            sources=("vendor specs: 960 GB/s", "xhinker medium: 24GB @ 960 GB/s")),
        Gpu("rx-9070-xt-16g", "AMD Radeon RX 9070 XT", AMD_RDNA4, 16.0, 640.0,
            tensor_cores=False, bf16_native=True,
            sources=("localaimaster bandwidth table: 640 GB/s",)),
        # ---- Intel --------------------------------------------------------
        Gpu("arc-a770-16g", "Intel Arc A770 16GB", INTEL_XE, 16.0, 560.0,
            tensor_cores=False, bf16_native=True,
            note="512 GB/s on the 8 GB model; 560 GB/s on the 16 GB Limited Edition.",
            sources=("pcmag: up to 560 GB/s", "localaimaster: 560 GB/s")),
        Gpu("arc-b580-12g", "Intel Arc B580 12GB", INTEL_XE2, 12.0, 456.0,
            tensor_cores=False, bf16_native=True,
            sources=("localaimaster bandwidth table: 456 GB/s",)),
        # ---- Apple Silicon ------------------------------------------------
        Gpu("apple-m1-16g", "Apple M1", APPLE_M, 16.0, 68.25,
            bf16_native=True, unified_memory=True,
            note="Unified memory: the GPU can address system RAM, so 'offload' is not a tradeoff.",
            sources=("apple: 68 GB/s",)),
        Gpu("apple-m2-max-64g", "Apple M2 Max", APPLE_M, 64.0, 400.0,
            bf16_native=True, unified_memory=True,
            sources=("apple: 400 GB/s",)),
        Gpu("apple-m4-16g", "Apple M4", APPLE_M, 16.0, 120.0,
            bf16_native=True, unified_memory=True,
            sources=("apple newsroom: M4 at 120 GB/s",)),
        Gpu("apple-m4-pro-24g", "Apple M4 Pro", APPLE_M, 24.0, 273.0,
            bf16_native=True, unified_memory=True,
            sources=("apple newsroom: 273 GB/s",)),
        Gpu("apple-m4-max-64g", "Apple M4 Max", APPLE_M, 64.0, 546.0,
            bf16_native=True, unified_memory=True,
            note="Apple family 9 GPU: IQ quants approach K-quant speed here.",
            sources=("apple newsroom: up to 546 GB/s",)),
        Gpu("apple-m3-ultra-192g", "Apple M3 Ultra", APPLE_M, 192.0, 800.0,
            bf16_native=True, unified_memory=True,
            sources=("apple newsroom: over 800 GB/s",)),
        # ---- CPU-only, for the honest baseline -----------------------------
        Gpu("cpu-ddr4-dual", "CPU only (dual-channel DDR4-3200)", CPU_ONLY, 0.0, 51.2,
            unified_memory=True, host_ram_gb=64.0,
            note="Roughly 25 GB/s per channel. The floor every plan is compared against.",
            sources=("ddr4-3200: 25.6 GB/s per channel",)),
        Gpu("cpu-ddr5-dual", "CPU only (dual-channel DDR5-5600)", CPU_ONLY, 0.0, 89.6,
            unified_memory=True, host_ram_gb=64.0,
            sources=("ddr5-5600: 44.8 GB/s per channel",)),
        Gpu("cpu-ddr5-oct", "CPU only (8-channel DDR5-5600)", CPU_ONLY, 0.0, 358.4,
            unified_memory=True, host_ram_gb=512.0,
            note="Server platforms: eight channels is why big-RAM CPU inference is viable.",
            sources=("ddr5-5600: 44.8 GB/s per channel x8",)),
    )
}

# What people actually say, mapped to catalogue keys.
GPU_ALIASES = {
    "p40": "tesla-p40-24g",
    "tesla p40": "tesla-p40-24g",
    "p4": "tesla-p4-8g",
    "v100": "tesla-v100-32g",
    "t4": "tesla-t4-16g",
    "1660": "gtx-1660-ti-6g",
    "1660 ti": "gtx-1660-ti-6g",
    "3060": "rtx-3060-12g",
    "3080": "rtx-3080-10g",
    "3090": "rtx-3090-24g",
    "4060": "rtx-4060-8g",
    "4060 ti": "rtx-4060-ti-16g",
    "4090": "rtx-4090-24g",
    "5090": "rtx-5090-32g",
    "rx580": "rx-580-8g",
    "580": "rx-580-8g",
    "6700xt": "rx-6700-xt-12g",
    "6800xt": "rx-6800-xt-16g",
    "7900xtx": "rx-7900-xtx-24g",
    "a770": "arc-a770-16g",
    "b580": "arc-b580-12g",
    "m1": "apple-m1-16g",
    "m2 max": "apple-m2-max-64g",
    "m4": "apple-m4-16g",
    "m4 pro": "apple-m4-pro-24g",
    "m4 max": "apple-m4-max-64g",
    "m3 ultra": "apple-m3-ultra-192g",
}


def resolve(key: str) -> Gpu:
    """Look up a GPU by catalogue key or a common way of writing its name."""
    slug = (key or "").strip().lower()
    if slug in GPUS:
        return GPUS[slug]
    if slug in GPU_ALIASES:
        return GPUS[GPU_ALIASES[slug]]
    # try progressively looser matching: normalise punctuation then alias again
    squashed = slug.replace("-", " ").replace("_", " ").replace("  ", " ").strip()
    for alias, target in GPU_ALIASES.items():
        if squashed == alias:
            return GPUS[target]
    for candidate, gpu in GPUS.items():
        if squashed == gpu.name.lower() or squashed in gpu.name.lower():
            return gpu
    raise KeyError(key)


def suggest_keys(key: str) -> list[str]:
    """Candidate keys for a 'did you mean' message."""
    slug = (key or "").strip().lower()
    tokens = [t for t in slug.replace("-", " ").split() if t]
    hits = [
        g.key
        for g in GPUS.values()
        if any(t in g.key or t in g.name.lower() for t in tokens)
    ]
    return sorted(hits)[:6] or sorted(GPUS)[:6]
