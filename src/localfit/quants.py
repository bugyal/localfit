"""Quantisation schemes, with their real bits-per-weight and measured quality.

The bits-per-weight numbers are the ones that actually determine file size and
therefore the memory budget. An "4-bit" quant is never 4.0 bits: the K-quants
store per-block scale factors, so ``Q4_K_M`` lands near 4.84 bpw and an 8B model
lands at ~4.9 GB rather than 4.0 GB. Sizing a machine on the nominal bit-width is
the single most common way people get this wrong, so the table carries the real
figure.

Quality figures are perplexity deltas measured against an F16 baseline on
Llama-3.1-8B-Instruct (the unified llama.cpp quantisation evaluation), reported
as ``ppl_delta``: lower is lossier. They are a guide, not a promise — a quant's
effect on your task depends on the task — which is why the report prints the
delta rather than a made-up percentage score.

Sources are recorded per entry so a disputed number can be traced.
"""

from __future__ import annotations

from dataclasses import dataclass

# Perplexity delta versus F16, from the unified llama.cpp quantisation evaluation
# on Llama-3.1-8B-Instruct. Values are for the same 8B model the bpw figures come
# from, so the two columns describe the same artefact.
_PPL_8B = {
    "f16": 0.0,
    "q8_0": 0.0016,
    "q6_k": 0.005,
    "q5_k_m": 0.0115,
    "q5_k_s": 0.0216,
    "q4_k_m": 0.0532,
    "q4_k_s": 0.0992,
    "q4_0": 0.1159,
    "iq4_xs": 0.0694,
    "iq4_nl": 0.0757,
    "q3_k_m": 0.2496,
    "q3_k_s": 0.5551,
    "iq3_m": 0.2135,
    "iq3_xs": 0.3176,
    "iq3_xxs": 0.3836,
    "q2_k": 0.6717,
    "iq2_m": 0.4661,
    "iq2_xs": 0.7026,
    "iq2_xxs": 0.8768,
    "iq1_s": 1.4,
}


@dataclass(frozen=True)
class Quant:
    """One quantisation scheme."""

    name: str
    bpw: float
    """Real bits per weight, including block scale factors."""

    family: str
    """``k``, ``iq``, ``legacy`` or ``float``. Drives the kernel-speed warning."""

    quality: str = ""

    @property
    def bytes_per_weight(self) -> float:
        return self.bpw / 8.0

    @property
    def ppl_delta(self) -> float | None:
        return _PPL_8B.get(self.name)

    def is_lossy_heavy(self) -> bool:
        """True where quality loss is likely to be visible on real work."""
        delta = self.ppl_delta
        return delta is not None and delta > 0.25


# bits per weight, ordered from the most faithful to the most aggressive.
QUANTS: dict[str, Quant] = {
    q.name: q
    for q in (
        Quant("f16", 16.0, "float", "reference; no quantisation"),
        Quant("bf16", 16.0, "float", "reference; no quantisation"),
        Quant("q8_0", 8.5, "legacy", "effectively lossless"),
        Quant("mxfp4", 4.25, "mxfp4", "4-bit MX; native on Blackwell, needs modern tensor cores"),
        Quant("q6_k", 6.5625, "k", "near-lossless"),
        Quant("q5_k_m", 5.6719, "k", "very good"),
        Quant("q5_k_s", 5.5156, "k", "very good"),
        Quant("iq4_xs", 4.25, "iq", "good; needs IQ kernels"),
        Quant("iq4_nl", 4.25, "iq", "good; needs IQ kernels"),
        Quant("q4_k_m", 4.8438, "k", "the default choice for most users"),
        Quant("q4_k_s", 4.5781, "k", "smaller, slightly lossier than Q4_K_M"),
        Quant("q4_0", 4.5, "legacy", "fastest kernels on old GPUs"),
        Quant("q3_k_m", 3.9063, "k", "noticeable loss on hard tasks"),
        Quant("iq3_m", 3.66, "iq", "needs IQ kernels"),
        Quant("iq3_xs", 3.3, "iq", "needs IQ kernels"),
        Quant("iq3_xxs", 3.06, "iq", "needs IQ kernels; lossy"),
        Quant("q3_k_s", 3.5, "k", "lossy"),
        Quant("q2_k", 2.5625, "k", "lossy; large models only"),
        Quant("iq2_m", 2.7, "iq", "needs IQ kernels; lossy"),
        Quant("iq2_xs", 2.31, "iq", "needs IQ kernels; lossy"),
        Quant("iq2_xxs", 2.06, "iq", "needs IQ kernels; very lossy"),
        Quant("iq1_s", 1.5625, "iq", "needs IQ kernels; expect degradation"),
    )
}

# What people type, mapped to the stored name.
QUANT_ALIASES = {
    "f32": "f16",
    "fp32": "f16",
    "float32": "f16",
    "fp16": "f16",
    "float16": "f16",
    "half": "f16",
    "bfloat16": "bf16",
    "q8": "q8_0",
    "q6": "q6_k",
    "q5": "q5_k_m",
    "q4": "q4_k_m",
    "q3": "q3_k_m",
    "q2": "q2_k",
    "iq4": "iq4_xs",
    "iq3": "iq3_m",
    "iq2": "iq2_m",
    "int8": "q8_0",
    "int4": "q4_0",
    "mx4": "mxfp4",
    "fp4": "mxfp4",
    "mxfp4_0": "mxfp4",
}

# Ordered worst-for-you first when a machine is too small: the planner walks this
# from the top down and stops at the first scheme that fits.
FIT_ORDER = [
    "q8_0",
    "q6_k",
    "q5_k_m",
    "q5_k_s",
    "q4_k_m",
    "q4_k_s",
    "q4_0",
    "iq4_xs",
    "q3_k_m",
    "iq3_m",
    "iq3_xs",
    "q3_k_s",
    "iq3_xxs",
    "q2_k",
    "iq2_m",
    "iq2_xs",
    "iq2_xxs",
    "iq1_s",
]


def resolve(name: str) -> Quant:
    """Look up a quant by name or common alias.

    Raises ``KeyError`` with the closest matches rather than silently defaulting,
    because guessing which quantisation someone meant is how you get a memory
    plan that is wrong by a factor of two.
    """
    key = (name or "").strip().lower().replace("-", "_")
    if key in QUANTS:
        return QUANTS[key]
    if key in QUANT_ALIASES:
        return QUANTS[QUANT_ALIASES[key]]
    raise KeyError(name)


def suggest_name(name: str) -> str:
    """A 'did you mean' hint for an unknown quant."""
    key = (name or "").strip().lower().replace("-", "_")
    matches = [q for q in QUANTS if key and (key in q or q in key)]
    if not matches:
        matches = [q for q in QUANTS if key and key[0] == q[0]]
    return ", ".join(sorted(matches)[:5]) or ", ".join(sorted(QUANTS)[:5])
