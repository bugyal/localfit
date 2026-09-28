"""Multi-GPU planning, and why your two cards will never balance.

This module exists because of one specific, repeatable complaint: you tune
``--tensor-split`` across two unequal cards, and one of them keeps 1-2 GB
unusable no matter what values you pass. The usual conclusion is that the flag is
broken or that the tool is buggy. It is neither, and the mechanism is worth
stating precisely because it explains the whole symptom:

1. **``--tensor-split`` takes ratios, not gigabytes.** ``24,16`` and ``3,2`` are
   the same instruction. Passing absolute GB mostly works by coincidence when the
   numbers are large, which is why the folklore half-works and never quite does.

2. **``--split-mode layer`` assigns whole layers.** llama.cpp walks the layer list
   and hands each GPU a contiguous run sized to its share. A layer is allocated
   atomically, except that oversized layers may *overflow* onto the next device.
   So the granularity of allocation is one layer, not one byte, and the waste on a
   given card is whatever the last layer did not fill — typically hundreds of MB,
   and on large-layer models well over a gigabyte. **This is the real answer to
   'no matter the split values': changing the ratio by less than one layer's worth
   changes nothing at all.** A one-integer nudge can therefore move nothing, or
   move a whole layer (2 GB on these models), which is exactly the reported
   behaviour.

3. **``--main-gpu`` holds the KV cache.** The KV buffer is allocated on one
   device, not split, and it grows with context. Whichever card is "main" loses
   that much more of its budget, which is why the imbalance is asymmetric even
   with identical cards.

4. **A draft model for speculative decoding sits on one device too.** The reported
   ``--spec-draft-device`` experiments belong to this bucket: the draft model and
   its own KV cache consume memory on whichever card holds them, and that memory
   never appears in the tensor-split arithmetic.

So the honest fixes are: size the split to *free* memory minus the KV and graph
budget, prefer ``--main-gpu`` on the largest card, and accept that the last layer
on each card leaves some VRAM unused — then verify from the load log rather than
expecting the numbers to come out even.

None of this is speculative: the layer-granular allocation and the overflow
behaviour are documented behaviour of the ``--fit`` machinery in llama.cpp, and
the ratios-versus-GB semantics are in the flag's own documentation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import planner, quants
from .hardware import Gpu
from .models import Model
from .planner import FitPlan, Split, kv_gb, overhead_gb, weight_gb


@dataclass
class MultiGpuPlan:
    """A fit plan spread across several devices, plus the imbalance diagnosis."""

    gpus: list[Gpu]
    model: Model
    quant_name: str
    context: int
    tensor_split: list[float] = field(default_factory=list)
    """Ratios, as llama.cpp wants them — not gigabytes."""

    splits: list[Split] = field(default_factory=list)
    fits: bool = False
    total_weights_gb: float = 0.0
    total_capacity_gb: float = 0.0
    warnings: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    """Copy-pasteable llama.cpp arguments."""
    imbalance_gb: float = 0.0
    """Spread between the fullest and emptiest card: the waste the user sees."""

    unusable_gb: float = 0.0
    """Total budget no card can use, because allocation is layer-granular."""

    cpu_layers: int = 0
    """Layers that did not fit on any card and must run from system RAM."""

    gpu_layers: int = 0
    """Layers that did fit on the cards."""

    max_overflow_gb: float = 0.0
    """Largest per-card overrun; above zero means a real load would OOM."""

    actual_split: list[float] = field(default_factory=list)
    """Ratios derived from the layers actually assigned, not from memory totals."""


def layer_bytes(model: Model, quant: quants.Quant) -> float:
    """Approximate bytes per transformer layer.

    Weights are not distributed perfectly evenly across layers (embeddings and the
    output head are chunky), but for the purpose of explaining why a split lands
    where it does, an average layer is the right unit — it is the quantum of
    allocation.
    """
    total_bytes = model.params_b * 1e9 * quant.bytes_per_weight
    return total_bytes / max(1, model.layers)


def suggest_tensor_split(
    gpus: list[Gpu], headroom: list[float] | None = None
) -> list[float]:
    """Ratios proportional to each card's *free* memory, not its spec capacity.

    Spec-capacity ratios are the common mistake: they ignore whatever is already
    resident on the card, which is precisely why a 24 GB card with a desktop
    session behaves unlike a bare one.
    """
    headroom = headroom or [0.0] * len(gpus)
    free = [max(0.0, g.vram_gb - h) for g, h in zip(gpus, headroom)]
    total = sum(free)
    if total <= 0:
        return [1.0 for _ in gpus]
    return [f / total for f in free]


def plan_multi(
    gpus: list[Gpu],
    model: Model,
    quant_name: str,
    context: int = 8192,
    headroom: list[float] | None = None,
    main_gpu_index: int | None = None,
) -> MultiGpuPlan:
    """Plan a layer-split across several cards, using layer granularity.

    The allocation walks whole layers onto each card in ratio order, the way
    llama.cpp's layer split does, and records what was actually assigned. That is
    what exposes the imbalance: the assigned total per card is a multiple of the
    layer size, so a card can end up hundreds of MB or gigabytes under the budget
    it was given.
    """
    if not gpus:
        raise ValueError("need at least one GPU")

    quant = quants.resolve(quant_name)
    headroom = headroom or [0.0] * len(gpus)
    ratios = suggest_tensor_split(gpus, headroom)

    if main_gpu_index is None:
        main_gpu_index = max(range(len(gpus)), key=lambda i: gpus[i].vram_gb)

    weights = weight_gb(model, quant)
    kv = kv_gb(model, context)
    over = overhead_gb(context)
    free = [max(0.0, g.vram_gb - h) for g, h in zip(gpus, headroom)]

    result = MultiGpuPlan(
        gpus=list(gpus),
        model=model,
        quant_name=quant.name,
        context=context,
        tensor_split=ratios,
        total_weights_gb=weights,
        total_capacity_gb=sum(free),
    )

    # The KV cache and the compute graph land on the main device, not spread.
    result.flags.append(f"--split-mode layer")
    result.flags.append("--tensor-split " + ",".join(_fmt_ratio(r) for r in ratios))
    result.flags.append(f"--main-gpu {main_gpu_index}")
    result.flags.append("-ngl 99")

    kv_and_over_headroom = kv + over
    per_card_budget = list(free)
    per_card_budget[main_gpu_index] = max(
        0.0, per_card_budget[main_gpu_index] - kv_and_over_headroom
    )

    # Hand out whole layers, largest-budget-first, never exceeding a card's
    # budget. Layers that do not fit stay on the CPU. Greedy-by-budget rather
    # than proportional matters: proportional assignment is what produces the
    # "-0.11 GB" overflow that a real load would turn into an OOM.
    layer_size = layer_bytes(model, quant)
    layer_gb = layer_size / 1e9
    per_card_layers = [0] * len(gpus)
    remaining_gb = list(per_card_budget)
    layers_left = model.layers

    # Largest remaining budget first, so capacity is filled where it exists.
    order = sorted(range(len(gpus)), key=lambda i: -remaining_gb[i])
    for index in order:
        if layers_left <= 0:
            break
        capacity_layers = int(remaining_gb[index] // layer_gb) if layer_gb > 0 else 0
        take = max(0, min(capacity_layers, layers_left))
        per_card_layers[index] = take
        remaining_gb[index] -= take * layer_gb
        layers_left -= take

    # Anything left over runs from system RAM: with -ngl below the layer count,
    # llama.cpp keeps whole layers on the CPU, which is slow but works.
    cpu_layers = layers_left
    result.cpu_layers = cpu_layers
    result.gpu_layers = model.layers - cpu_layers

    # Rebalance pass. The greedy fill above places layers but does not balance
    # what is left over, and the leftover is what users experience as unusable
    # VRAM. Note the total waste is fixed — sum(free) minus the model's total
    # footprint is a constant — so the only thing worth optimising is the
    # *maximum* waste across cards. That is the actual complaint in the evidence:
    # one card showing 1+ GB free while another is full.
    #
    # Whole layers only, and never past a card's budget, so this stays physically
    # realisable with --split-mode layer.
    if layer_gb > 0 and free:
        def max_waste(card_layers: list[int]) -> float:
            worst = 0.0
            for i, n in enumerate(card_layers):
                held = n * layer_gb + ((kv + over) if i == main_gpu_index else 0.0)
                worst = max(worst, free[i] - held)
            return worst

        improved = True
        while improved:
            improved = False
            best_move: tuple[int, int] | None = None
            best_worst = max_waste(per_card_layers)
            for donor in range(len(gpus)):
                if per_card_layers[donor] <= 0:
                    continue
                for taker in range(len(gpus)):
                    if taker == donor:
                        continue
                    # The taker must have room for one more whole layer.
                    held = per_card_layers[taker] * layer_gb + (
                        (kv + over) if taker == main_gpu_index else 0.0
                    )
                    if held + layer_gb > free[taker] + 1e-9:
                        continue
                    trial = list(per_card_layers)
                    trial[donor] -= 1
                    trial[taker] += 1
                    worst = max_waste(trial)
                    # Require a real improvement, not float noise.
                    if worst < best_worst - 1e-6:
                        best_worst = worst
                        best_move = (donor, taker)
            if best_move is not None:
                per_card_layers[best_move[0]] -= 1
                per_card_layers[best_move[1]] += 1
                improved = True

    for index, gpu in enumerate(gpus):
        layers = per_card_layers[index]
        bytes_assigned = layers * layer_size
        if index == main_gpu_index:
            # The KV cache and compute graph live on the main device regardless
            # of how many layers it holds.
            bytes_assigned += int((kv + over) * 1e9)
        split = Split(
            gpu=gpu,
            index=index,
            bytes_assigned=int(bytes_assigned),
            layers_assigned=layers,
            is_main=(index == main_gpu_index),
        )
        result.splits.append(split)

    # It fits only if every layer found a home on some card. The comparison is
    # against each card's whole free memory, because the assignment already
    # includes the KV cache and graph that land on the main card — measuring
    # against the layer budget instead would count the KV cache twice and report
    # a fit as an overflow.
    over_budget = [
        result.splits[i].gb_assigned - free[i] for i in range(len(result.splits))
    ]
    result.max_overflow_gb = max(over_budget) if over_budget else 0.0
    result.fits = cpu_layers == 0 and result.max_overflow_gb <= 1e-9

    # Re-state the split ratio from what was actually assigned. This is the
    # actionable version of the advice: it is derived from the real allocation
    # rather than from memory totals, which is why it can differ from the naive
    # proportional guess.
    actual = [s.layers_assigned for s in result.splits]
    total_assigned = sum(actual)
    if total_assigned > 0:
        result.actual_split = [n / total_assigned for n in actual]

    # Correct the flags now that the allocation is known. `-ngl 99` is the usual
    # advice and is wrong whenever layers spill: asking for more layers than fit
    # is how you turn a slow-but-working load into an OOM.
    result.flags = [
        "--split-mode layer",
        "--tensor-split "
        + ",".join(_fmt_ratio(r) for r in (result.actual_split or ratios)),
        f"--main-gpu {main_gpu_index}",
        f"-ngl {result.gpu_layers}",
    ]

    # The imbalance: what each card was allowed versus what it actually holds.
    # This is the number the user experiences as unusable VRAM. Measured against
    # each card's whole free memory, because the KV cache is part of what the
    # main card holds, not something subtracted from its budget twice.
    if result.splits:
        waste = [
            free[i] - result.splits[i].gb_assigned for i in range(len(result.splits))
        ]
        result.imbalance_gb = max(waste) - min(waste) if waste else 0.0
        result.unusable_gb = sum(max(0.0, w) for w in waste)
        for index, split in enumerate(result.splits):
            if waste[index] >= 1.0:
                split.note = (
                    f"{waste[index]:.2f} GB of this card is unused at "
                    f"layer granularity (a layer is {layer_gb:.2f} GB)"
                )

    _add_diagnosis(result, gpus, model, quant, kv, main_gpu_index, layer_gb)
    return result


def _fmt_ratio(value: float) -> str:
    """Render a ratio the way people write it, without float noise."""
    return f"{value:.3f}".rstrip("0").rstrip(".") or "0"


def _add_diagnosis(
    result: MultiGpuPlan,
    gpus: list[Gpu],
    model: Model,
    quant: quants.Quant,
    kv: float,
    main_gpu_index: int,
    layer_gb: float,
) -> None:
    """Attach the explanations the evidence shows people are missing."""
    if len(gpus) > 1:
        smallest = min(gpus, key=lambda g: g.bandwidth_gbs)
        largest = max(gpus, key=lambda g: g.bandwidth_gbs)
        if largest.bandwidth_gbs > smallest.bandwidth_gbs * 1.5:
            result.warnings.append(
                f"Mixed-bandwidth pair: {largest.name} has "
                f"{largest.bandwidth_gbs / smallest.bandwidth_gbs:.1f}x the bandwidth "
                f"of {smallest.name}. With --split-mode layer the slowest card gates "
                "every token, so total throughput lands near the slow card's rate "
                "rather than the sum. Splitting does not add bandwidth across "
                "unequal cards; it only adds capacity."
            )
        if any(g.arch != gpus[0].arch for g in gpus):
            result.warnings.append(
                "Cards span architectures, so they cannot be treated as one "
                "pool: different kernel sets, different quantisation speeds."
            )
        naive = ",".join(_fmt_ratio(r) for r in result.tensor_split)
        tuned = ",".join(_fmt_ratio(r) for r in (result.actual_split or result.tensor_split))
        warning = (
            f"Tensor-split takes ratios, not gigabytes. The memory-proportional "
            f"guess is {naive}; the ratio that matches the layer breakdown above is "
            f"{tuned}. Use the second one — the first is what leaves a card holding "
            "less than it could."
        )
        if naive != tuned:
            warning += (
                " They differ because layers are indivisible: the split is quantised "
                "to whole layers, so the ideal ratio is never the tidy memory ratio."
            )
        result.warnings.append(warning)
        result.warnings.append(
            f"--main-gpu {main_gpu_index} ({gpus[main_gpu_index].name}) holds the "
            f"{kv:.2f} GB KV cache for this context, which is why its budget is "
            "smaller than its VRAM suggests. Move it if that card is the small one."
        )
        if result.imbalance_gb >= 1.0:
            result.warnings.append(
                f"Expect roughly {result.imbalance_gb:.1f} GB of unusable VRAM "
                "across the pair. Layer assignment happens in whole layers "
                f"({layer_gb:.2f} GB each here), so small ratio changes do nothing "
                "and the next step jumps by a whole layer. This is a property of "
                "layer-granular allocation, not a bug in your flags."
            )
        result.warnings.append(
            "A draft model for speculative decoding is not part of the tensor "
            "split: --spec-draft-device pins it to one card, and its weights plus "
            "its own KV cache come out of that card's budget. Account for it "
            "separately or the load will OOM on the card holding it."
        )
    if quant.family == "iq":
        from .perf import iq_advisory

        advisory = iq_advisory(gpus[0].arch)
        if advisory:
            result.warnings.append(advisory)


def verify_split(plan: MultiGpuPlan) -> list[str]:
    """Problems with a plan, as things to fix rather than numbers to read."""
    problems: list[str] = []
    for split in plan.splits:
        if split.headroom_gb < -1e-9:
            problems.append(
                f"{split.gpu.name} (device {split.index}) is over budget by "
                f"{-split.headroom_gb:.2f} GB — layers assigned exceed free memory."
            )
    if plan.cpu_layers:
        problems.append(
            f"{plan.cpu_layers} of {plan.model.layers} layers did not fit on any "
            "card and would run from system RAM. That works, but generation speed "
            "collapses to CPU rates for those layers, and the estimate below is for "
            f"a full GPU offload. Use a smaller quantisation or lower --ctx to fit "
            "all layers."
        )
    if not plan.fits and not plan.cpu_layers and plan.max_overflow_gb <= 1e-9:
        problems.append(
            f"Model needs {plan.total_weights_gb:.2f} GB of weights but only "
            f"{plan.total_capacity_gb:.2f} GB is free across the cards."
        )
    return problems
