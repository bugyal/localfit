"""The fit plan: memory arithmetic, then speed.

Memory first, because it is the question that has a definite answer. A model
either fits in the memory you have at the context you asked for, or it does not,
and that is pure arithmetic:

    weights    = params x bytes-per-weight           (from the real bpw, not the label)
    kv cache   = kv_bytes_per_token x context
    overhead   = compute buffers, CUDA context, fragmentation

The overhead term is the one people omit and then get bitten by. llama.cpp
allocates a fixed-size compute graph that grows with batch and context, and the
driver takes a few hundred MB before any model loads. On a 24 GB card that is
noise; on an 8 GB card it decides whether the plan works. It is budgeted here
rather than hoped away.

Speed second, because it is a range and not a number. The bandwidth ceiling is
hard; the achievable fraction of it is a fitted efficiency (see ``perf``); and
where a real measurement exists for the exact configuration, that measurement is
reported instead of the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import perf, quants
from .hardware import Gpu
from .models import Model

# Fixed driver and context overhead, in GB. A CUDA context plus the smallest
# usable compute graph lands in this range; it is not zero and claiming otherwise
# produces plans that OOM on load.
DRIVER_OVERHEAD_GB = 0.30

# Compute-graph buffers scale with the batch and context actually used. This is a
# per-1000-tokens-of-context allowance, calibrated so that a 32k-context run on a
# 24 GB card budgets around 0.6 GB of graph.
GRAPH_GB_PER_1K_CONTEXT = 0.018

# Fraction of the model that must be resident on the GPU for the plan to be worth
# calling "fits". Offloading only the last few layers to a card with the rest in
# system RAM technically runs but is dominated by the PCIe transfer, so below this
# the plan is reported as a hybrid with an explicit warning.
FULL_OFFLOAD_THRESHOLD = 0.99

LOW_HEADROOM_GB = 1.0
"""Below this much spare memory, a plan that technically fits is likely to fail on
a machine whose card also drives a display — the driver and compositor take a few
hundred megabytes that no memory total accounts for."""


@dataclass
class Split:
    """One GPU's share of a multi-GPU plan."""

    gpu: Gpu
    index: int
    bytes_assigned: int
    layers_assigned: int
    is_main: bool = False
    note: str = ""

    @property
    def gb_assigned(self) -> float:
        return self.bytes_assigned / 1e9

    @property
    def headroom_gb(self) -> float:
        return self.gpu.vram_gb - self.gb_assigned


@dataclass
class FitPlan:
    """The answer to 'will this run, and how fast'."""

    gpu: Gpu
    model: Model
    quant_name: str
    quant: quants.Quant
    context: int
    batch: int = 1

    weights_gb: float = 0.0
    kv_gb: float = 0.0
    overhead_gb: float = 0.0
    total_gb: float = 0.0
    capacity_gb: float = 0.0

    fits: bool = False
    resident_fraction: float = 1.0
    """Share of the model held in GPU memory; <1 means a hybrid with CPU spill."""

    gpus: list[Split] = field(default_factory=list)
    tok_s_low: float = 0.0
    tok_s_mid: float = 0.0
    tok_s_high: float = 0.0
    ceiling_tok_s: float = 0.0
    measured: perf.Anchor | None = None
    prefill_note: str = ""
    warnings: list[str] = field(default_factory=list)
    quality_delta: float | None = None

    @property
    def headroom_gb(self) -> float:
        return self.capacity_gb - self.total_gb

    @property
    def fits_with_room(self) -> bool:
        return self.fits and self.headroom_gb >= 0.5


def weight_gb(model: Model, quant: quants.Quant) -> float:
    """File and resident size of the weights, in decimal GB.

    Decimal, because model files and VRAM are both sold in decimal GB; using GiB
    here makes plans disagree with what the download says and with what
    ``nvidia-smi`` reports.
    """
    return model.params_b * 1e9 * quant.bytes_per_weight / 1e9


def kv_gb(model: Model, context: int, dtype_bytes: float = 2.0) -> float:
    """KV cache size for a context length, in decimal GB."""
    return model.kv_bytes_per_token(dtype_bytes) * context / 1e9


def usable_memory_gb(gpu: Gpu) -> float:
    """The memory a plan may actually use on this device.

    For an accelerator that is VRAM. For a CPU-only entry it is system RAM, which
    is the whole point of the entry: reporting zero capacity would make every CPU
    plan answer "does not fit", which is not an answer.
    """
    if gpu.vram_gb > 0:
        return gpu.vram_gb
    return gpu.host_ram_gb


def overhead_gb(context: int) -> float:
    """Driver plus compute-graph allowance for a given context."""
    return DRIVER_OVERHEAD_GB + GRAPH_GB_PER_1K_CONTEXT * (context / 1000.0)


def plan(
    gpu: Gpu,
    model: Model,
    quant_name: str,
    context: int = 8192,
    batch: int = 1,
    headroom_gb: float = 0.0,
) -> FitPlan:
    """Build a single-GPU fit plan.

    ``headroom_gb`` is memory the user knows is already spoken for (a display, a
    desktop session, another process). Ignoring it is how a plan that "fits"
    fails on a card that has a monitor attached.
    """
    quant = quants.resolve(quant_name)
    weights = weight_gb(model, quant)
    kv = kv_gb(model, context)
    over = overhead_gb(context)
    total = weights + kv + over
    capacity = max(0.0, usable_memory_gb(gpu) - headroom_gb)

    fits = total <= capacity
    resident = 1.0 if fits else max(0.0, min(1.0, (capacity - kv - over) / weights))
    if weights <= 0:
        resident = 1.0

    result = FitPlan(
        gpu=gpu,
        model=model,
        quant_name=quant.name,
        quant=quant,
        context=context,
        batch=batch,
        weights_gb=weights,
        kv_gb=kv,
        overhead_gb=over,
        total_gb=total,
        capacity_gb=capacity,
        fits=fits,
        resident_fraction=resident,
        quality_delta=quant.ppl_delta,
    )

    if not gpu.unified_memory and 0.0 < resident < FULL_OFFLOAD_THRESHOLD:
        result.warnings.append(
            f"Only {resident:.0%} of the weights fit in GPU memory; the rest runs "
            "from system RAM over PCIe. Expect a large speed penalty and a "
            "context-length ceiling — reduce quantisation or context, or add a card."
        )
    if gpu.unified_memory and total > gpu.vram_gb:
        result.warnings.append(
            "Unified memory: the GPU addresses system RAM directly, so this does "
            "not need layer offload — but the total still has to fit in physical RAM."
        )
    if quant.is_lossy_heavy():
        result.warnings.append(
            f"{quant.name} has a measured perplexity penalty of "
            f"+{quant.ppl_delta:.3f} on an 8B reference model. Fine for chat, "
            "risky for code or maths."
        )
    advisory = perf.iq_advisory(gpu.arch)
    if quant.family == "iq" and advisory:
        result.warnings.append(advisory)

    _add_speed(result, model, quant, gpu)
    return result


def _add_speed(result: FitPlan, model: Model, quant: quants.Quant, gpu: Gpu) -> None:
    """Fill in the throughput numbers and the prefill note.

    The bandwidth ceiling is always computed, even when a measurement exists for
    this exact configuration, because the report shows the two side by side and a
    measurement plus the bound it must obey is more informative than either alone.
    """
    bytes_per_token = model.decode_params_b * 1e9 * quant.bytes_per_weight
    if bytes_per_token <= 0 or gpu.bandwidth_gbs <= 0:
        result.ceiling_tok_s = 0.0
        result.prefill_note = "no bandwidth figure available for this device"
        return
    result.ceiling_tok_s = gpu.bandwidth_gbs * 1e9 / bytes_per_token

    anchor = perf.find_anchor(gpu.key, model.key, quant.name)
    if anchor is not None:
        # A measurement outranks any model of the hardware.
        result.measured = anchor
        result.tok_s_mid = anchor.tok_s
        result.tok_s_low = anchor.tok_s * 0.8
        result.tok_s_high = anchor.tok_s * 1.2
    else:
        effective = perf.efficiency_for(gpu.arch, quant.family) * gpu.bandwidth_gbs
        # Where this card has been measured, scale the estimate to agree with
        # those measurements, so a quantisation table does not show a measured
        # figure sitting implausibly beside estimated neighbours.
        correction, samples = perf.gpu_correction(gpu.key, quant.family)
        if samples:
            effective *= correction
        # MoE only reads its active weights, which decode_params_b already
        # reflects; dense reads everything. So the same expression covers both,
        # and the difference is carried by the model rather than special-cased.
        mid = effective * 1e9 / bytes_per_token
        if correction != 1.0:
            result.warnings.append(
                f"Estimate scaled by {correction:.2f}x to agree with "
                f"{samples} measured configuration(s) on {gpu.name}."
            )
        if result.resident_fraction < FULL_OFFLOAD_THRESHOLD and not gpu.unified_memory:
            # CPU-resident layers run an order of magnitude slower; weight the
            # estimate by the share that actually sits on the card.
            mid *= max(0.05, result.resident_fraction)
            result.warnings.append(
                "tok/s estimate scaled down for the CPU-resident share of the model."
            )
        result.tok_s_mid = mid
        result.tok_s_low = mid * 0.70
        result.tok_s_high = mid * 1.30

    # Prefill behaves differently from decode by construction: it is a
    # compute-bound batched matmul, so it does not scale with bandwidth but with
    # tensor throughput, and MoE models suffer badly because every expert the
    # batch touches must be read.
    if model.is_moe:
        result.prefill_note = (
            "MoE: prefill reads every expert the batch activates, so prompt "
            "processing is disproportionately slower than decode and degrades "
            "as concurrency rises. A 5B-active model can be far faster at "
            "generation than at prompt processing."
        )
    else:
        result.prefill_note = (
            "Dense: prefill is compute-bound and typically 20-100x the decode "
            "rate at batch 1; it falls as concurrency rises."
        )
    if quant.family == "iq" and gpu.arch == "nvidia-pascal":
        result.prefill_note += (
            " IQ quants additionally slow prefill on this architecture, which "
            "matches the reported 'good tok/s but degrading prefill' pattern."
        )

    if quant.family == "mxfp4":
        advisory = perf.mxfp4_advisory(gpu.arch)
        if advisory:
            result.warnings.append(advisory)

    # A plan that fits with almost nothing to spare is the most common way a
    # correct answer still fails in practice, because the card also drives a
    # display and the driver takes its own share. Say so rather than let a
    # 0.6 GB margin read as a green light.
    if result.fits and result.headroom_gb < LOW_HEADROOM_GB:
        result.warnings.append(
            f"Only {result.headroom_gb:.2f} GB spare. Headless this is workable; "
            "with a display attached or a desktop session running it will likely "
            "OOM, because the driver and compositor take a few hundred MB. Pass "
            f"--headroom <GB> to plan against the memory you actually have, or drop "
            "a step in the context length."
        )


def largest_fitting_quant(
    gpu: Gpu,
    model: Model,
    context: int = 8192,
    headroom_gb: float = 0.0,
    max_loss: float | None = None,
) -> FitPlan | None:
    """The highest-quality quantisation that fits.

    Walks ``FIT_ORDER`` from the most faithful scheme down and returns the first
    plan that fits. ``max_loss`` caps the acceptable perplexity penalty, because
    "it fits" and "it is good enough" are different questions.
    """
    for name in quants.FIT_ORDER:
        candidate = quants.QUANTS[name]
        if max_loss is not None and candidate.ppl_delta is not None:
            if candidate.ppl_delta > max_loss:
                continue
        candidate_plan = plan(gpu, model, name, context, headroom_gb=headroom_gb)
        if candidate_plan.fits:
            return candidate_plan
    return None


def all_plans(
    gpu: Gpu, model: Model, context: int = 8192, headroom_gb: float = 0.0
) -> list[FitPlan]:
    """A plan for every quantisation, for the comparison table."""
    names = ["f16"] + [n for n in quants.FIT_ORDER if n in quants.QUANTS]
    return [plan(gpu, model, n, context, headroom_gb=headroom_gb) for n in names]


def max_context(gpu: Gpu, model: Model, quant_name: str, headroom_gb: float = 0.0) -> int:
    """The largest context that still fits, capped at the model's training limit.

    Context is the cheapest thing to give up and the most common cause of a plan
    that fits "on paper" but OOMs in practice, so it is worth answering directly.
    """
    quant = quants.resolve(quant_name)
    weights = weight_gb(model, quant)
    capacity = max(0.0, usable_memory_gb(gpu) - headroom_gb)
    per_token = model.kv_bytes_per_token() / 1e9
    # Solve weights + per_token*ctx + overhead(ctx) <= capacity, stepping rather
    # than inverting so the non-linear graph term is handled exactly.
    best = 0
    for step in range(1024, model.context_max + 1, 512):
        if weights + per_token * step + overhead_gb(step) <= capacity:
            best = step
        else:
            break
    return best
