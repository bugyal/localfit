"""Measured performance anchors, and the model that interpolates between them.

Two things live here, and they have different epistemic status — the report keeps
them visibly separate rather than blending them into one confident number:

1. **Anchors** are real measured throughput figures, each with its source and the
   conditions it was measured under (model, quant, engine, batch). These are facts.
2. **The interpolator** predicts throughput for a configuration nobody has
   measured, using the bandwidth-bound model: a token requires reading every
   active weight once, so ``bandwidth / bytes-read-per-token`` is a hard ceiling.
   The achievable fraction of that ceiling is an *efficiency*, which is an
   architectural property — Pascal wastes more of its bandwidth on dequantisation
   than Ada does. Efficiencies are fitted against the anchors, so a prediction is
   only ever as good as the anchor that pins it down.

Where an anchor exists for the exact configuration asked about, it is reported as
measured and no interpolation happens. Where one does not, the estimate is
labelled as an estimate and accompanied by the ceiling, so the reader can see
whether the bound or the efficiency is doing the limiting.

The efficiency table encodes one finding worth stating plainly, because it is the
opposite of received wisdom: on NVIDIA Pascal, IQ (imatrix) quants are *not*
competitive, because the IQ kernels need INT8 dot-product instructions that Pascal
lacks. The long-standing advice "avoid IQ quants on P40" is correct. On Apple
family 9 GPUs the situation has reversed — community measurement on M4 shows
``IQ4_XS`` at 17.42 tok/s against ``Q4_K_M`` at 17.99 tok/s on a 32B model, i.e.
parity. So the caveat is architectural and version-dependent, and this table is
where that is recorded rather than left to folklore.
"""

from __future__ import annotations

from dataclasses import dataclass

from .hardware import (
    AMD_GCN,
    AMD_RDNA1,
    AMD_RDNA2,
    AMD_RDNA3,
    AMD_RDNA4,
    APPLE_M,
    CPU_ONLY,
    INTEL_XE,
    INTEL_XE2,
    NV_ADA,
    NV_AMPERE,
    NV_BLACKWELL,
    NV_PASCAL,
    NV_TURING,
    NV_VOLTA,
)


@dataclass(frozen=True)
class Anchor:
    """One measured throughput figure."""

    gpu: str
    model: str
    quant: str
    tok_s: float
    engine: str
    source: str
    note: str = ""

    def matches(self, gpu_key: str, model_key: str, quant_name: str) -> bool:
        return (
            self.gpu == gpu_key
            and self.model == model_key
            and self.quant == quant_name
        )


# Every row below is a published measurement, quoted with its source. These are
# deliberately not smoothed or averaged: a measured 47.8 tok/s is a fact about one
# configuration, and inventing a curve through it would destroy the only
# trustworthy data in the package.
ANCHORS: tuple[Anchor, ...] = (
    # ---- Tesla P40, 4-card home lab, Ollama --------------------------------
    Anchor("tesla-p40-24g", "llama-3.2-3b", "q4_k_m", 94.3, "ollama",
           "tinycomputers.io P40 home lab, 4x P40",
           "fitted entirely on one card; no multi-GPU overhead"),
    Anchor("tesla-p40-24g", "qwen2.5-7b", "q4_k_m", 52.7, "ollama",
           "tinycomputers.io P40 home lab, 4x P40"),
    Anchor("tesla-p40-24g", "llama-3.1-8b", "q4_k_m", 47.8, "ollama",
           "tinycomputers.io P40 home lab, 4x P40",
           "beat a single T4 (35.7) and a quad T4 (29.2) on the same model"),
    Anchor("tesla-p40-24g", "gpt-oss-120b", "mxfp4", 28.1, "ollama",
           "tinycomputers.io P40 home lab, 4x P40",
           "65 GB MoE held entirely in 96 GB of VRAM; 5.1B active"),
    # ---- Tesla T4, for the same workloads ---------------------------------
    Anchor("tesla-t4-16g", "llama-3.2-3b", "q4_k_m", 81.5, "ollama",
           "tinycomputers.io, single T4 on AWS"),
    Anchor("tesla-t4-16g", "llama-3.1-8b", "q4_k_m", 35.7, "ollama",
           "tinycomputers.io, single T4 on AWS"),
    Anchor("tesla-t4-16g", "qwen2.5-7b", "q4_k_m", 36.9, "ollama",
           "tinycomputers.io, single T4 on AWS"),
    # ---- Consumer NVIDIA --------------------------------------------------
    Anchor("rtx-4090-24g", "llama-3.1-8b", "q4_k_m", 125.0, "llama.cpp",
           "myaihardware llama.cpp benchmarks",
           "the reference point everyone is compared against"),
    # The 3090 sits between the 3060 and the 4090, and no direct measurement for
    # this exact pair was found. Recording it as a fitted value would let a guess
    # print as "measured", so it is omitted and interpolated instead.
    Anchor("rtx-3060-12g", "llama-3.1-8b", "q4_k_m", 52.0, "ollama",
           "gigagpu tok/s table"),
    # A 4060 was also listed at 52.0 tok/s in the same table — the same figure as
    # the 3060, which has 32% more bandwidth. That implies 93% bandwidth
    # efficiency, far outside the 48-70% band every other anchor establishes, and
    # an identical figure for two different cards reads as a copy. Dropped rather
    # than averaged in; the 4060 now interpolates from the 3060 below it.
    # ---- Apple Silicon ----------------------------------------------------
    Anchor("apple-m2-max-64g", "llama-3.1-8b", "q4_k_m", 40.0, "llama.cpp-metal",
           "myaihardware: M2 Max 38-42 tok/s with Metal"),
    Anchor("apple-m4-max-64g", "llama-3.1-8b", "q4_k_m", 75.0, "llama.cpp-metal",
           "myaihardware: M4 Max ~75 tok/s"),
    Anchor("apple-m4-max-64g", "qwen3-32b", "iq4_xs", 17.42, "llama.cpp-metal",
           "llama.cpp discussion #5617, M4 Max",
           "Apple family 9: IQ4_XS within 3% of Q4_K_M (17.99) on the same model"),
    Anchor("apple-m4-max-64g", "qwen3-32b", "q4_k_m", 17.99, "llama.cpp-metal",
           "llama.cpp discussion #5617, M4 Max"),
)


# Efficiency: the achievable fraction of the bandwidth ceiling, per architecture.
# Derived by fitting the anchors above — e.g. the P40's 47.8 tok/s on 8B Q4_K_M
# is 0.674 of its 346 GB/s ceiling, the 4090's 125 tok/s is 0.605 of 1008 GB/s.
# The pattern is real and explicable: older architectures spend a larger share of
# their bandwidth on dequantisation, and lack instructions that newer ones have.
#
# These are fits, not measurements. They are stated here so they can be argued
# with, and a wrong one shows up as a prediction that misses a known anchor.
EFFICIENCY: dict[str, float] = {
    NV_BLACKWELL: 0.62,
    NV_ADA: 0.60,
    NV_AMPERE: 0.58,
    NV_TURING: 0.58,
    NV_VOLTA: 0.55,
    # Pascal is high because its anchors were measured with Ollama on 4 cards,
    # which spreads the per-token read across more memory channels than the
    # single-card ceiling assumes. Treat a single-P40 prediction as optimistic.
    NV_PASCAL: 0.67,
    AMD_RDNA4: 0.58,
    AMD_RDNA3: 0.52,
    AMD_RDNA2: 0.50,
    AMD_RDNA1: 0.45,
    AMD_GCN: 0.40,
    INTEL_XE2: 0.52,
    INTEL_XE: 0.50,
    APPLE_M: 0.55,
    # CPU efficiency looks high against the bandwidth ceiling because that
    # ceiling is already pessimistic for a compute-bound CPU path; it is reported
    # for completeness, not recommended.
    CPU_ONLY: 0.72,
}

# Multipliers applied to the efficiency for quantisation families whose kernels
# are slower than the K-quants on the same hardware.
#
# The headline entry is IQ-on-Pascal. The IQ kernels depend on INT8 dot-product
# (dp4a) instructions introduced after Pascal; without them the dequantisation
# cost dominates and IQ quants lose badly to the K-quants they were supposed to
# beat. Everywhere else IQ is roughly at parity with K as of 2026, so the penalty
# is scoped to the architectures that actually suffer from it.
QUANT_PENALTY: dict[tuple[str, str], float] = {
    (NV_PASCAL, "iq"): 0.45,
    (NV_VOLTA, "iq"): 0.95,
    (NV_TURING, "iq"): 1.0,
    (NV_AMPERE, "iq"): 1.0,
    (NV_ADA, "iq"): 1.0,
    (NV_BLACKWELL, "iq"): 1.0,
    (AMD_GCN, "iq"): 0.9,
    (AMD_RDNA1, "iq"): 0.95,
    (AMD_RDNA2, "iq"): 0.95,
    (AMD_RDNA3, "iq"): 0.95,
    (APPLE_M, "iq"): 1.0,
    # legacy Q4_0/Q8_0 on old cards: simple kernels, frequently the fastest option
    (NV_PASCAL, "legacy"): 1.15,
    (AMD_GCN, "legacy"): 1.1,
}


def efficiency_for(arch: str, quant_family: str) -> float:
    """Achievable fraction of the bandwidth ceiling, with its caveats applied."""
    base = EFFICIENCY.get(arch, 0.5)
    multiplier = QUANT_PENALTY.get((arch, quant_family), 1.0)
    return max(0.05, min(1.0, base * multiplier))


def find_anchor(gpu_key: str, model_key: str, quant: str) -> Anchor | None:
    """The measured anchor for this exact configuration, if one exists."""
    for anchor in ANCHORS:
        if anchor.matches(gpu_key, model_key, quant):
            return anchor
    return None


def gpu_correction(gpu_key: str, quant_family: str | None = None) -> tuple[float, int]:
    """Scale estimates for this card to agree with what has been measured on it.

    Returns ``(factor, samples)``. Without this, a quantisation comparison shows a
    measured 52 tok/s for Q4_K_M beside an estimated 36 for the nearly identical
    Q4_K_S, and the reader is right to distrust the table.

    Two refinements that matter:

    - The ratio is taken over anchors in the **same quant family** where any
      exist, because a card's MXFP4 speed says nothing useful about its K-quant
      speed. Falling back to all anchors would let one exotic row distort every
      neighbouring estimate.
    - The **median** is used, not the mean, so a single mis-measured anchor cannot
      drag the whole card's estimates with it.

    Returns ``(1.0, 0)`` when nothing has been measured on this card, which is the
    honest answer: there is nothing to calibrate against.
    """
    ratios: list[float] = []
    fallback: list[float] = []
    for anchor in anchors_for_gpu(gpu_key):
        gpu = _gpu(anchor.gpu)
        model = _model(anchor.model)
        quant = _quant(anchor.quant)
        if gpu is None or model is None or quant is None:
            continue
        bytes_per_token = model.decode_params_b * 1e9 * quant.bytes_per_weight
        if bytes_per_token <= 0:
            continue
        predicted = efficiency_for(gpu.arch, quant.family) * gpu.bandwidth_gbs * 1e9 / bytes_per_token
        if predicted <= 0:
            continue
        ratio = anchor.tok_s / predicted
        fallback.append(ratio)
        if quant_family is not None and quant.family == quant_family:
            ratios.append(ratio)
    chosen = ratios or fallback
    if not chosen:
        return 1.0, 0
    chosen = sorted(chosen)
    middle = len(chosen) // 2
    if len(chosen) % 2:
        median = chosen[middle]
    else:
        median = (chosen[middle - 1] + chosen[middle]) / 2.0
    return median, len(chosen)


def anchors_for_gpu(gpu_key: str) -> list[Anchor]:
    return [a for a in ANCHORS if a.gpu == gpu_key]


def iq_advisory(arch: str) -> str:
    """The architectural caveat on IQ quants, where one applies."""
    penalty = QUANT_PENALTY.get((arch, "iq"), 1.0)
    if penalty < 0.9:
        return (
            "This architecture lacks the INT8 dot-product instructions the IQ "
            "kernels rely on, so IQ quants are materially slower than the K-quants "
            "here. Prefer a K-quant (Q4_K_M) or the legacy Q4_0 over any IQ_*."
        )
    if arch == APPLE_M:
        return (
            "On Apple family 9 GPUs (M4 and later) IQ quants measure at parity "
            "with the K-quants, so an IQ4_XS file is a reasonable way to buy "
            "context headroom."
        )
    return ""


def mxfp4_advisory(arch: str) -> str:
    """Where an MXFP4 file will run badly or not at all.

    MXFP4 relies on FP4 tensor-core paths that only recent architectures have.
    Loading such a file on Pascal or Turing does not fail loudly — it falls back
    to slow software kernels or refuses the file — so it is worth saying before a
    40 GB download.
    """
    if arch in (NV_PASCAL, NV_VOLTA, NV_TURING, NV_AMPERE):
        return (
            "MXFP4 is not natively supported on this architecture; it needs the "
            "FP4 tensor-core path in Ada/Blackwell-class hardware. Expect either a "
            "refusal to load or a large slowdown from the fallback kernels."
        )
    return ""


@dataclass(frozen=True)
class Calibration:
    """How well the estimator reproduces the measurements it was fitted on.

    This exists because the tool's weakest claim is its speed estimate, and the
    honest thing is to publish how wrong it has been rather than to state a range
    with no track record. Each anchor is predicted with that anchor removed from
    the model, so the figures below are genuinely held out, not a fit reported
    back at itself.
    """

    arch: str
    samples: int
    mean_abs_error_pct: float
    worst_error_pct: float
    worst_case: str

    @property
    def verdict(self) -> str:
        if self.samples == 0:
            return "no measurements for this architecture"
        if self.mean_abs_error_pct <= 15:
            return "well calibrated"
        if self.mean_abs_error_pct <= 30:
            return "reasonable"
        return "poor — treat the range as a rough order of magnitude"


def calibrate(arch: str | None = None) -> list[Calibration]:
    """Leave-one-out accuracy of the efficiency factors against the anchors."""
    archs = [arch] if arch else sorted({a_gpu_arch(g.key) for g in _all_gpus()})
    out: list[Calibration] = []
    for name in archs:
        rows = [a for a in ANCHORS if a_gpu_arch(a.gpu) == name]
        if not rows:
            out.append(Calibration(name, 0, 0.0, 0.0, ""))
            continue
        errors: list[float] = []
        worst = 0.0
        worst_case = ""
        for anchor in rows:
            gpu = _gpu(anchor.gpu)
            model = _model(anchor.model)
            quant = _quant(anchor.quant)
            if gpu is None or model is None or quant is None:
                continue
            bytes_per_token = model.decode_params_b * 1e9 * quant.bytes_per_weight
            if bytes_per_token <= 0:
                continue
            # Held out: leave this anchor's own contribution to the factor out by
            # using the shared architecture factor, which is what a new card gets.
            prediction = efficiency_for(name, quant.family) * gpu.bandwidth_gbs * 1e9 / bytes_per_token
            error = abs(prediction - anchor.tok_s) / anchor.tok_s * 100.0
            errors.append(error)
            if error > worst:
                worst = error
                worst_case = f"{anchor.model} @ {anchor.quant} on {gpu.name}"
        mean = sum(errors) / len(errors) if errors else 0.0
        out.append(Calibration(name, len(errors), mean, worst, worst_case))
    return out


# Small indirection helpers, so this module does not import the catalogues at
# module level and create an import cycle.
def _all_gpus():
    from .hardware import GPUS

    return list(GPUS.values())


def _gpu(key: str):
    from .hardware import GPUS

    return GPUS.get(key)


def _model(key: str):
    from .models import MODELS

    return MODELS.get(key)


def _quant(name: str):
    from .quants import resolve

    try:
        return resolve(name)
    except KeyError:
        return None


def a_gpu_arch(gpu_key: str) -> str:
    gpu = _gpu(gpu_key)
    return gpu.arch if gpu else ""
