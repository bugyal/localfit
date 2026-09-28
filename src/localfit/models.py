"""Model architectures and the KV-cache arithmetic they imply.

The KV cache is where most hand estimates go wrong, and it is the reason a model
that "fits" at 8K context dies at 32K. Its size is

    bytes = 2 (K and V) x layers x kv_heads x head_dim x context x dtype_bytes

and the important subtlety is ``kv_heads``, not ``heads``: grouped-query
attention (GQA) stores far fewer key/value heads than query heads, which is why
modern models with GQA need a fraction of the cache a same-size older model does.
Using ``heads`` instead of ``kv_heads`` overestimates the cache by the group
factor — a 8x error on a typical 8-to-1 GQA model.

Dense and mixture-of-experts sizes are kept separate for throughput: an MoE model
must *hold* all its weights but only *reads* the active ones per token, and since
decode is bandwidth-bound those are two different numbers. Conflating them is why
MoE models are routinely predicted to be far slower than they are.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    """One model, as published."""

    key: str
    name: str
    params_b: float
    """Total parameters in billions."""

    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    context_max: int
    active_params_b: float = 0.0
    """For an MoE, the parameters touched per token. Zero means dense."""

    mla_latent_dim: int = 0
    """Non-zero for models using Multi-head Latent Attention (DeepSeek V2/V3/R1).

    MLA caches a single compressed latent vector per token per layer instead of
    per-head K and V, so the GQA arithmetic above overestimates its cache by an
    order of magnitude. Roughly, the cached width is the latent dimension plus a
    small decoupled-RoPE term, not ``2 x kv_heads x head_dim``.
    """

    note: str = ""

    @property
    def is_moe(self) -> bool:
        return 0.0 < self.active_params_b < self.params_b

    @property
    def decode_params_b(self) -> float:
        """Parameters that must be read to emit one token."""
        return self.active_params_b if self.active_params_b else self.params_b

    @property
    def gqa_group(self) -> int:
        """Query heads per KV head. >1 means the cache is compressed."""
        return max(1, self.heads // max(1, self.kv_heads))

    def kv_bytes_per_token(self, dtype_bytes: float = 2.0) -> float:
        """KV cache bytes for a single token, both K and V.

        Two formulas, because there are two mechanisms. GQA stores K and V for a
        reduced number of heads; MLA stores one compressed latent per layer plus
        the decoupled-RoPE key term. Applying the GQA formula to an MLA model
        inflates its cache by the head-to-latent ratio, which is how a tool ends
        up telling someone a 671B model needs more memory than they have.
        """
        if self.mla_latent_dim:
            # Latent plus the shared RoPE component; the RoPE term is per-head on
            # the key only, which is the 1x head_dim addition below.
            return self.layers * (self.mla_latent_dim + self.head_dim) * dtype_bytes
        return 2.0 * self.layers * self.kv_heads * self.head_dim * dtype_bytes


# Parameters are the published figures, rounded to the same precision vendors use.
# Head geometry follows each family's config; where a family ships several sizes
# the smallest and largest are catalogued and the rest interpolate sensibly.
MODELS: dict[str, Model] = {
    m.key: m
    for m in (
        Model("llama-3.2-3b", "Llama 3.2 3B", 3.2, 28, 24, 8, 128, 131072,
              note="GQA 3:1. Fits a 6 GB card at Q4."),
        Model("llama-3.1-8b", "Llama 3.1 8B", 8.0, 32, 32, 8, 128, 131072,
              note="The community reference point; most measured anchors use it."),
        Model("qwen2.5-7b", "Qwen 2.5 7B", 7.6, 28, 28, 4, 128, 131072),
        Model("mistral-7b", "Mistral 7B", 7.2, 32, 32, 8, 128, 32768),
        Model("gemma-2-9b", "Gemma 2 9B", 9.2, 42, 16, 8, 256, 8192,
              note="Wide head_dim; large KV cache for its size."),
        Model("llama-3.1-70b", "Llama 3.1 70B", 70.6, 80, 64, 8, 128, 131072,
              note="Dense 70B is the classic 'does not fit on one card' wall."),
        Model("qwen3-32b", "Qwen 3 32B", 32.8, 64, 64, 8, 128, 131072),
        Model("qwen3-27b", "Qwen 3.8 Flash 27B", 27.0, 48, 24, 4, 128, 131072,
              note="From evidence #2 for this build. GQA 6:1; head geometry read "
              "from the family config, not from a model card, so treat the KV figure "
              "as indicative."),
        Model("qwen3-14b", "Qwen 3 14B", 14.8, 48, 40, 8, 128, 131072),
        # MoE: hold everything, read a fraction.
        Model("gpt-oss-120b", "GPT-OSS 120B (MoE)", 117.0, 36, 64, 8, 64, 131072,
              active_params_b=5.1,
              note="5.1B active per token: needs 117B of memory but decodes like a "
              "5B. head_dim is 64, unusually small, which keeps its KV cache cheap."),
        Model("qwen3-30b-a3b", "Qwen 3 30B-A3B (MoE)", 30.5, 48, 32, 4, 128, 131072,
              active_params_b=3.3,
              note="3.3B active. Fast on modest bandwidth, large memory footprint."),
        Model("mixtral-8x7b", "Mixtral 8x7B (MoE)", 46.7, 32, 32, 8, 128, 32768,
              active_params_b=12.9,
              note="12.9B active of 46.7B held."),
        Model("deepseek-v3-671b", "DeepSeek V3 671B (MoE)", 671.0, 61, 128, 128, 128,
              131072, active_params_b=37.0,
              mla_latent_dim=512,
              note="37B active. Uses MLA, not GQA: the cache stores a 512-dim "
              "compressed latent rather than per-head K/V, so it is far smaller "
              "than the head count suggests."),
    )
}

MODEL_ALIASES = {
    "llama3.2:3b": "llama-3.2-3b",
    "llama-3.2-3b": "llama-3.2-3b",
    "llama3.1:8b": "llama-3.1-8b",
    "llama-3.1-8b": "llama-3.1-8b",
    "llama3:8b": "llama-3.1-8b",
    "8b": "llama-3.1-8b",
    "70b": "llama-3.1-70b",
    "qwen2.5:7b": "qwen2.5-7b",
    "mistral:7b": "mistral-7b",
    "gemma2:9b": "gemma-2-9b",
    "qwen3:32b": "qwen3-32b",
    "qwen3:14b": "qwen3-14b",
    "qwen3.8:27b": "qwen3-27b",
    "qwen3.8-27b": "qwen3-27b",
    "gpt-oss:120b": "gpt-oss-120b",
    "gpt-oss-120b": "gpt-oss-120b",
}


def resolve(key: str) -> Model:
    """Look up a model by catalogue key or a common way of writing its name."""
    slug = (key or "").strip().lower()
    if slug in MODELS:
        return MODELS[slug]
    if slug in MODEL_ALIASES:
        return MODELS[MODEL_ALIASES[slug]]
    for candidate, model in MODELS.items():
        if slug == model.name.lower() or slug in model.name.lower().replace(" ", ""):
            return model
    raise KeyError(key)


def suggest_keys(key: str) -> list[str]:
    slug = (key or "").strip().lower()
    numeric = "".join(ch for ch in slug if ch.isdigit())
    hits = [m.key for m in MODELS.values() if numeric and numeric in m.key]
    return sorted(hits)[:6] or sorted(MODELS)[:6]
