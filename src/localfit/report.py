"""Reports.

The report is ordered by what the reader needs in the order they need it: can it
run, how fast, what to type, and what will go wrong. Two rules hold throughout:

- Every number is labelled with its provenance. A measured figure, an arithmetic
  result and a fitted estimate are three different kinds of claim and are never
  printed in a way that blurs them.
- Where something could not be determined, it is said plainly. A report that
  hides an unknown behind a plausible default is worse than one that admits it.
"""

from __future__ import annotations

from . import perf, quants
from .hardware import Gpu
from .models import Model
from .multigpu import MultiGpuPlan, verify_split
from .planner import FitPlan
from .probe import Measurement, ProbeResult


def _bar(value: float, total: float, width: int = 24) -> str:
    """A crude fill bar, for showing memory composition at a glance."""
    if total <= 0:
        return ""
    filled = int(round(width * max(0.0, min(1.0, value / total))))
    return "#" * filled + "." * (width - filled)


def render_plan(plan: FitPlan, *, title: str = "") -> str:
    """The single-GPU fit plan."""
    lines: list[str] = []
    lines.append(f"# localfit — {plan.model.name} @ {plan.quant_name}")
    if title:
        lines.append(f"  on {title}")
    lines.append("")

    verdict = "FITS" if plan.fits else "DOES NOT FIT"
    lines.append(f"## {verdict}")
    lines.append("")
    if plan.fits:
        lines.append(
            f"{plan.total_gb:.2f} GB needed of {plan.capacity_gb:.2f} GB usable "
            f"on {plan.gpu.name} ({plan.headroom_gb:+.2f} GB headroom)."
        )
    else:
        shortfall = plan.total_gb - plan.capacity_gb
        lines.append(
            f"{plan.total_gb:.2f} GB needed but only {plan.capacity_gb:.2f} GB "
            f"usable on {plan.gpu.name} — short by {shortfall:.2f} GB."
        )
    lines.append("")

    # -- memory composition -------------------------------------------------
    lines.append("### Memory")
    lines.append("")
    lines.append("| component | size | of capacity |")
    lines.append("|---|---|---|")
    cap = plan.capacity_gb or 1.0
    lines.append(
        f"| weights ({plan.quant_name}) | {plan.weights_gb:.2f} GB | "
        f"`{_bar(plan.weights_gb, cap)}` |"
    )
    lines.append(f"| KV cache ({plan.context:,} ctx) | {plan.kv_gb:.2f} GB | `{_bar(plan.kv_gb, cap)}` |")
    lines.append(f"| driver + compute graph | {plan.overhead_gb:.2f} GB | `{_bar(plan.overhead_gb, cap)}` |")
    lines.append(f"| **total** | **{plan.total_gb:.2f} GB** | |")
    lines.append(f"| usable on card | {plan.capacity_gb:.2f} GB | |")
    lines.append("")

    if not plan.fits:
        lines.append(
            f"Only {plan.resident_fraction:.0%} of the weights fit in memory; the "
            "remainder would run from system RAM."
        )
        lines.append("")

    # -- speed --------------------------------------------------------------
    lines.append("### Speed")
    lines.append("")
    if plan.measured is not None:
        lines.append(
            f"**Measured: {plan.measured.tok_s:.1f} tok/s** "
            f"({plan.measured.model}, {plan.measured.quant}, {plan.measured.engine})"
        )
        lines.append(f"  source: {plan.measured.source}")
        if plan.measured.note:
            lines.append(f"  note: {plan.measured.note}")
        lines.append("")
        lines.append(
            f"The bandwidth ceiling for this configuration is "
            f"{plan.ceiling_tok_s:.1f} tok/s, so the measurement is using "
            f"{plan.measured.tok_s / plan.ceiling_tok_s:.0%} of what the hardware "
            "allows."
        )
    else:
        lines.append(
            f"Estimated **{plan.tok_s_low:.0f}-{plan.tok_s_high:.0f} tok/s** "
            f"(midpoint {plan.tok_s_mid:.0f})."
        )
        lines.append("")
        lines.append(f"- bandwidth ceiling for this config: {plan.ceiling_tok_s:.1f} tok/s")
        lines.append(
            f"- the estimate applies an efficiency factor for {plan.gpu.arch}, "
            "fitted against measured anchors; it is an estimate, not a measurement"
        )
        confidence = perf.calibrate(plan.gpu.arch)
        if confidence:
            record = confidence[0]
            if record.samples == 0:
                lines.append(
                    f"- **no measured runs exist for {plan.gpu.arch}**, so this "
                    "figure is unvalidated. Treat it as an order of magnitude."
                )
            else:
                lines.append(
                    f"- track record on {plan.gpu.arch}: {record.samples} measured "
                    f"configuration(s), mean error {record.mean_abs_error_pct:.0f}% "
                    f"({record.verdict})"
                )
    lines.append("")
    if plan.prefill_note:
        lines.append(f"**Prefill:** {plan.prefill_note}")
        lines.append("")

    # -- quality ------------------------------------------------------------
    if plan.quality_delta is not None:
        lines.append("### Quality")
        lines.append("")
        if plan.quality_delta == 0:
            lines.append("Reference precision; no quantisation loss.")
        else:
            lines.append(
                f"Perplexity penalty **+{plan.quality_delta:.3f}** versus F16 on an "
                "8B reference model. Lower is better."
            )
        lines.append("")

    # -- warnings -----------------------------------------------------------
    if plan.warnings:
        lines.append("### Watch out")
        lines.append("")
        for warning in plan.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("### Provenance")
    lines.append("")
    lines.append(
        f"- weights: {plan.model.params_b:g}B x {plan.quant.bpw:g} bits/weight (real "
        f"bits-per-weight, not the label) = {plan.weights_gb:.2f} GB"
    )
    lines.append(
        f"- KV cache: 2 x {plan.model.layers} layers x {plan.model.kv_heads} kv_heads "
        f"x {plan.model.head_dim} = {plan.model.kv_bytes_per_token():.0f} B/token"
    )
    if plan.model.is_moe:
        lines.append(
            f"- MoE: {plan.model.params_b:g}B held, {plan.model.active_params_b:g}B "
            "active per token (memory follows the first, speed the second)"
        )
    return "\n".join(lines)


def render_all_plans(plans: list[FitPlan], model: Model, gpu: Gpu, context: int) -> str:
    """The quantisation comparison table."""
    lines = [f"# localfit — {model.name} on {gpu.name} @ {context:,} context", ""]
    lines.append("| quant | bpw | weights | +kv+graph | fits | tok/s | quality |")
    lines.append("|---|---|---|---|---|---|---|")
    for plan in plans:
        fits = "yes" if plan.fits else "no"
        if plan.measured is not None:
            speed = f"{plan.measured.tok_s:.0f} (measured)"
        elif plan.tok_s_mid > 0:
            speed = f"~{plan.tok_s_mid:.0f}"
        else:
            speed = "-"
        delta = plan.quality_delta
        quality = "-" if delta is None else ("ref" if delta == 0 else f"+{delta:.3f} ppl")
        lines.append(
            f"| {plan.quant_name} | {plan.quant.bpw:g} | {plan.weights_gb:.2f} GB | "
            f"{plan.total_gb:.2f} GB | {fits} | {speed} | {quality} |"
        )
    lines.append("")
    fitting = [p for p in plans if p.fits]
    if fitting:
        best = fitting[0]
        lines.append(
            f"Highest-quality scheme that fits: **{best.quant_name}** "
            f"({best.total_gb:.2f} GB, {best.quality_delta or 0:.3f} ppl penalty)."
        )
    else:
        lines.append("Nothing in the catalogue fits this model at this context.")
        largest = max(plans, key=lambda p: p.capacity_gb)
        lines.append(
            f"Largest quantisation still over budget: {largest.quant_name} at "
            f"{largest.total_gb:.2f} GB against {largest.capacity_gb:.2f} GB usable."
        )
    return "\n".join(lines)


def render_multi(plan: MultiGpuPlan) -> str:
    """The multi-GPU plan, with the imbalance explained."""
    lines: list[str] = []
    lines.append(f"# localfit — {plan.model.name} @ {plan.quant_name} across {len(plan.gpus)} GPUs")
    lines.append("")
    status = "FITS" if plan.fits else "DOES NOT FIT"
    lines.append(f"## {status}")
    lines.append("")
    lines.append(
        f"Weights total {plan.total_weights_gb:.2f} GB; free across the cards "
        f"{plan.total_capacity_gb:.2f} GB."
    )
    if plan.cpu_layers:
        lines.append("")
        lines.append(
            f"{plan.gpu_layers} of {plan.model.layers} layers fit on the cards; "
            f"{plan.cpu_layers} would run from system RAM."
        )
    lines.append("")

    lines.append("### Layer assignment")
    lines.append("")
    lines.append("| device | card | layers | assigned | budget | unused |")
    lines.append("|---|---|---|---|---|---|")
    for split in plan.splits:
        main = " (main)" if split.is_main else ""
        unused = split.gpu.vram_gb - split.gb_assigned
        lines.append(
            f"| {split.index} | {split.gpu.name}{main} | {split.layers_assigned} | "
            f"{split.gb_assigned:.2f} GB | {split.gpu.vram_gb:.1f} GB | "
            f"{unused:.2f} GB |"
        )
    lines.append("")

    problems = verify_split(plan)
    if problems:
        lines.append("### Problems")
        lines.append("")
        for problem in problems:
            lines.append(f"- {problem}")
        lines.append("")

    lines.append("### Flags")
    lines.append("")
    lines.append("```")
    for flag in plan.flags:
        lines.append(flag)
    lines.append("```")
    lines.append("")

    if plan.warnings:
        lines.append("### Why it will not balance")
        lines.append("")
        for warning in plan.warnings:
            lines.append(f"- {warning}")
        lines.append("")

    lines.append("### Verify from the load log")
    lines.append("")
    lines.append(
        "Do not trust the flags — read what llama.cpp reports. It prints the "
        "per-device buffer sizes on load, and that is the only authoritative "
        "answer to where the layers actually went:"
    )
    lines.append("")
    lines.append("```")
    lines.append("llama-server <flags above> 2>&1 | grep -E 'buffer size|tensor_split|offloaded'")
    lines.append("```")
    lines.append("")
    lines.append(
        "Then feed that log back: `localfit measure <log>` reports the real "
        "per-device split and turns it into a contribution row."
    )
    return "\n".join(lines)


def render_probe(result: ProbeResult) -> str:
    """What hardware was detected."""
    lines = ["# localfit — detected hardware", ""]
    lines.append(f"- platform: {result.platform}")
    lines.append(f"- system RAM: {result.system_ram_gb:.1f} GB")
    if result.cuda_version:
        lines.append(f"- CUDA: {result.cuda_version}")
    if result.rocm_version:
        lines.append(f"- ROCm: {result.rocm_version}")
    lines.append("")
    if result.gpus:
        lines.append("| card | arch | memory | bandwidth |")
        lines.append("|---|---|---|---|")
        for gpu in result.gpus:
            lines.append(
                f"| {gpu.name} | {gpu.arch} | {gpu.vram_gb:.0f} GB | "
                f"{gpu.bandwidth_gbs:.0f} GB/s |"
            )
        lines.append("")
    else:
        lines.append("No accelerator detected.")
        lines.append("")
    if result.notes:
        lines.append("### Notes")
        lines.append("")
        for note in result.notes:
            lines.append(f"- {note}")
        lines.append("")
    if result.unknown:
        lines.append("### Could not determine")
        lines.append("")
        for item in result.unknown:
            lines.append(f"- {item}")
        lines.append("")
    return "\n".join(lines)


def render_measurement(measurement: Measurement) -> str:
    """A parsed log, and what it says."""
    lines = ["# localfit — measured run", ""]
    if measurement.model_path:
        lines.append(f"- model: {measurement.model_path}")
    if measurement.eval_tok_s:
        lines.append(f"- **generation: {measurement.eval_tok_s:.2f} tok/s** (measured)")
    if measurement.prompt_tok_s:
        lines.append(f"- prompt processing: {measurement.prompt_tok_s:.2f} tok/s")
    if measurement.layers_total:
        lines.append(
            f"- layers offloaded: {measurement.layers_offloaded}/{measurement.layers_total}"
        )
    if measurement.tensor_split:
        lines.append(f"- tensor-split in effect: {measurement.tensor_split}")
    if measurement.main_gpu:
        lines.append(f"- main GPU: {measurement.main_gpu}")
    if measurement.context:
        lines.append(f"- context: {measurement.context:,}")
    if measurement.kv_cache_mib:
        lines.append(f"- KV cache: {measurement.kv_cache_mib:.0f} MiB")
    lines.append("")

    if measurement.per_device_mib:
        lines.append("### Per-device buffers")
        lines.append("")
        lines.append("| device | model buffer |")
        lines.append("|---|---|")
        for name, mib in measurement.per_device_mib:
            lines.append(f"| {name} | {mib:.0f} MiB |")
        lines.append("")
        values = [v for _, v in measurement.per_device_mib]
        lines.append(
            f"Spread between fullest and emptiest device: "
            f"{max(values) - min(values):.0f} MiB."
        )
        lines.append("")

    if measurement.warnings:
        lines.append("### Notes")
        lines.append("")
        for warning in measurement.warnings:
            lines.append(f"- {warning}")
        lines.append("")
    return "\n".join(lines)


def render_hardware_catalogue() -> str:
    """The known cards, so a user can find the right key."""
    from .hardware import GPUS

    lines = ["# localfit — known hardware", ""]
    lines.append("| key | card | arch | memory | bandwidth |")
    lines.append("|---|---|---|---|---|")
    for gpu in sorted(GPUS.values(), key=lambda g: -g.bandwidth_gbs):
        lines.append(
            f"| `{gpu.key}` | {gpu.name} | {gpu.arch} | {gpu.vram_gb:.0f} GB | "
            f"{gpu.bandwidth_gbs:.0f} GB/s |"
        )
    return "\n".join(lines)


def render_model_catalogue() -> str:
    """The known models."""
    from .models import MODELS

    lines = ["# localfit — known models", ""]
    lines.append("| key | model | params | layers | kv heads | max ctx | notes |")
    lines.append("|---|---|---|---|---|---|---|")
    for model in sorted(MODELS.values(), key=lambda m: m.params_b):
        kind = "MoE" if model.is_moe else "dense"
        active = f", {model.active_params_b:g}B active" if model.is_moe else ""
        lines.append(
            f"| `{model.key}` | {model.name} | {model.params_b:g}B {kind}{active} | "
            f"{model.layers} | {model.kv_heads} | {model.context_max:,} | "
            f"{model.note} |"
        )
    return "\n".join(lines)
