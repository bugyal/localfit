"""Tests for the memory arithmetic and the fit verdict.

These matter most, because a wrong fit verdict is the one error a user cannot
detect before downloading a model. The numbers asserted here are hand-computed
from the published architecture figures, so a change to the formula that is not
also a change to the physics will fail the suite.
"""

from __future__ import annotations

import pytest

from localfit import hardware, models, planner, quants


class TestKvArithmetic:
    def test_kv_cache_scales_with_kv_heads_not_heads(self):
        """The classic error: using query heads inflates the cache by the GQA group."""
        model = models.MODELS["llama-3.1-8b"]
        # 2 (K and V) x 32 layers x 8 kv_heads x 128 head_dim x 2 bytes = 131072
        assert model.kv_bytes_per_token() == pytest.approx(131_072.0)
        # Using 32 query heads instead would be 4x too large.
        wrong = 2.0 * 32 * 32 * 128 * 2.0
        assert wrong == 4 * model.kv_bytes_per_token()

    def test_mla_model_uses_the_latent_formula(self):
        """DeepSeek's MLA caches a compressed latent, not per-head K/V."""
        model = models.MODELS["deepseek-v3-671b"]
        assert model.mla_latent_dim == 512
        # layers x (latent + rope term) x 2 bytes
        expected = 61 * (512 + 128) * 2.0
        assert model.kv_bytes_per_token() == pytest.approx(expected)
        # The GQA formula would overstate it enormously.
        gqa = 2.0 * 61 * 128 * 128 * 2.0
        assert gqa > 30 * model.kv_bytes_per_token()

    def test_gqa_group_reported(self):
        assert models.MODELS["qwen3-27b"].gqa_group == 6
        assert models.MODELS["llama-3.1-70b"].gqa_group == 8


class TestWeightArithmetic:
    def test_bpw_is_real_not_the_label(self):
        """An '4-bit' quant is not 4 bits per weight; the scale factors count."""
        q4 = quants.resolve("q4_k_m")
        assert q4.bpw == pytest.approx(4.8438)
        assert q4.bpw > 4.0

    def test_8b_q4_lands_near_the_published_file_size(self):
        """A real Q4_K_M 8B GGUF is about 4.9 GB; the arithmetic must agree."""
        gb = planner.weight_gb(models.MODELS["llama-3.1-8b"], quants.resolve("q4_k_m"))
        assert 4.6 < gb < 5.0

    def test_moe_holds_all_weights_but_reads_active_only(self):
        model = models.MODELS["gpt-oss-120b"]
        assert model.is_moe
        assert model.decode_params_b == pytest.approx(5.1)
        # Memory follows the held size...
        held = planner.weight_gb(model, quants.resolve("q4_k_m"))
        assert held > 60
        # ...and speed follows the active size.
        active = planner.weight_gb(
            models.Model("x", "x", model.active_params_b, 1, 1, 1, 1, 1),
            quants.resolve("q4_k_m"),
        )
        assert active < 4


class TestFitVerdict:
    def test_fits_when_total_under_capacity(self):
        gpu = hardware.GPUS["rtx-3090-24g"]
        plan = planner.plan(gpu, models.MODELS["llama-3.1-8b"], "q4_k_m", 8192)
        assert plan.fits
        assert plan.headroom_gb > 0

    def test_does_not_fit_when_model_exceeds_card(self):
        gpu = hardware.GPUS["rtx-4060-8g"]
        plan = planner.plan(gpu, models.MODELS["llama-3.1-70b"], "q4_k_m", 8192)
        assert not plan.fits
        assert plan.headroom_gb < 0

    def test_headroom_reduces_capacity(self):
        """Memory already spoken for must come out of the budget."""
        gpu = hardware.GPUS["rtx-3090-24g"]
        clean = planner.plan(gpu, models.MODELS["llama-3.1-8b"], "q4_k_m", 8192)
        busy = planner.plan(
            gpu, models.MODELS["llama-3.1-8b"], "q4_k_m", 8192, headroom_gb=4.0
        )
        assert busy.capacity_gb == pytest.approx(clean.capacity_gb - 4.0)
        assert busy.total_gb == pytest.approx(clean.total_gb)

    def test_context_growth_eventually_does_not_fit(self):
        """The KV cache is what kills a plan as context rises."""
        gpu = hardware.GPUS["rtx-3060-12g"]
        small = planner.plan(gpu, models.MODELS["llama-3.1-8b"], "q8_0", 4096)
        huge = planner.plan(gpu, models.MODELS["llama-3.1-8b"], "q8_0", 131072)
        assert small.fits
        assert not huge.fits
        assert huge.kv_gb > small.kv_gb

    def test_low_headroom_is_flagged_not_silently_green(self):
        """A plan with almost no margin will fail on a card driving a display."""
        # Find the context at which an 8B Q5 barely fits a 12 GB card.
        gpu = hardware.GPUS["rtx-3060-12g"]
        tight = None
        for ctx in range(1024, 131073, 1024):
            candidate = planner.plan(gpu, models.MODELS["llama-3.1-8b"], "q6_k", ctx)
            if candidate.fits and candidate.headroom_gb < planner.LOW_HEADROOM_GB:
                tight = candidate
                break
        assert tight is not None, "expected some context to fit tightly"
        assert any("spare" in w for w in tight.warnings)


class TestQuantSelection:
    def test_picks_the_best_quality_that_fits(self):
        gpu = hardware.GPUS["rtx-3090-24g"]
        chosen = planner.largest_fitting_quant(gpu, models.MODELS["llama-3.1-8b"], 8192)
        assert chosen is not None
        assert chosen.fits
        # Nothing higher-quality should also fit.
        for name in quants.FIT_ORDER:
            if name == chosen.quant_name:
                break
            above = planner.plan(gpu, models.MODELS["llama-3.1-8b"], name, 8192)
            assert not above.fits, f"{name} fits but a lower-quality quant was chosen"

    def test_max_loss_rejects_a_lossy_quant(self):
        """'It fits' and 'it is good enough' are different questions."""
        gpu = hardware.GPUS["rtx-3060-12g"]
        chosen = planner.largest_fitting_quant(
            gpu, models.MODELS["llama-3.1-70b"], 8192, max_loss=0.5
        )
        if chosen is not None:
            assert chosen.quality_delta is not None
            assert chosen.quality_delta <= 0.5

    def test_max_loss_can_make_nothing_acceptable(self):
        gpu = hardware.GPUS["rtx-4060-8g"]
        chosen = planner.largest_fitting_quant(
            gpu, models.MODELS["llama-3.1-70b"], 8192, max_loss=0.0001
        )
        assert chosen is None

    def test_all_plans_covers_the_catalogue(self):
        gpu = hardware.GPUS["tesla-p40-24g"]
        plans = planner.all_plans(gpu, models.MODELS["llama-3.1-8b"], 8192)
        assert len(plans) >= 18
        assert plans[0].quant_name == "f16"

    def test_max_context_is_capped_by_the_model(self):
        gpu = hardware.GPUS["apple-m3-ultra-192g"]
        ctx = planner.max_context(gpu, models.MODELS["gemma-2-9b"], "q4_k_m")
        assert ctx <= models.MODELS["gemma-2-9b"].context_max

    def test_max_context_shrinks_as_the_quant_grows(self):
        gpu = hardware.GPUS["rtx-3060-12g"]
        big = planner.max_context(gpu, models.MODELS["llama-3.1-8b"], "q4_k_m")
        small = planner.max_context(gpu, models.MODELS["llama-3.1-8b"], "q8_0")
        assert small < big


class TestResidentFraction:
    def test_partial_offload_reports_a_fraction_and_warns(self):
        gpu = hardware.GPUS["rtx-4060-8g"]
        plan = planner.plan(gpu, models.MODELS["llama-3.1-70b"], "q2_k", 2048)
        assert 0.0 < plan.resident_fraction <= 1.0
        if plan.resident_fraction < planner.FULL_OFFLOAD_THRESHOLD:
            assert any("system RAM" in w for w in plan.warnings)

    def test_unified_memory_has_no_pcie_penalty(self):
        """On Apple Silicon there is no transfer to penalise, so no warning."""
        gpu = hardware.GPUS["apple-m4-max-64g"]
        plan = planner.plan(gpu, models.MODELS["llama-3.1-70b"], "q4_k_m", 4096)
        assert not any("over PCIe" in w for w in plan.warnings)


class TestSpeedEstimate:
    def test_measured_anchor_wins_and_is_labelled(self):
        gpu = hardware.GPUS["tesla-p40-24g"]
        plan = planner.plan(gpu, models.MODELS["llama-3.2-3b"], "q4_k_m", 8192)
        assert plan.measured is not None
        assert plan.measured.tok_s == pytest.approx(94.3)
        assert plan.tok_s_mid == pytest.approx(94.3)

    def test_ceiling_is_always_computed_even_with_a_measurement(self):
        """A measurement plus the bound it must obey beats either alone."""
        gpu = hardware.GPUS["tesla-p40-24g"]
        plan = planner.plan(gpu, models.MODELS["llama-3.2-3b"], "q4_k_m", 8192)
        assert plan.ceiling_tok_s > 0
        assert plan.measured is not None
        # The measurement must not exceed the bandwidth bound it is measured against.
        assert plan.measured.tok_s <= plan.ceiling_tok_s

    def test_estimate_respects_the_bandwidth_ceiling(self):
        gpu = hardware.GPUS["rtx-3090-24g"]
        plan = planner.plan(gpu, models.MODELS["llama-3.1-8b"], "q4_k_m", 8192)
        assert plan.measured is None
        assert plan.tok_s_mid <= plan.ceiling_tok_s
        assert plan.tok_s_low < plan.tok_s_mid < plan.tok_s_high

    def test_exceeding_the_ceiling_means_a_bad_anchor(self):
        """Every measured row must obey the physics it was measured under."""
        from localfit import perf

        for anchor in perf.ANCHORS:
            gpu = hardware.GPUS[anchor.gpu]
            model = models.MODELS[anchor.model]
            quant = quants.resolve(anchor.quant)
            bpt = model.decode_params_b * 1e9 * quant.bytes_per_weight
            ceiling = gpu.bandwidth_gbs * 1e9 / bpt
            assert anchor.tok_s <= ceiling * 1.05, (
                f"{anchor.gpu}/{anchor.model}/{anchor.quant} measured "
                f"{anchor.tok_s} tok/s but the ceiling is {ceiling:.1f}"
            )

    def test_pascal_iq_advisory_is_raised(self):
        gpu = hardware.GPUS["tesla-p40-24g"]
        plan = planner.plan(gpu, models.MODELS["llama-3.1-8b"], "iq4_xs", 8192)
        assert any("IQ" in w for w in plan.warnings)
        assert "K-quant" in " ".join(plan.warnings)

    def test_iq_advisory_not_raised_on_modern_architectures(self):
        gpu = hardware.GPUS["rtx-4090-24g"]
        plan = planner.plan(gpu, models.MODELS["llama-3.1-8b"], "iq4_xs", 8192)
        assert not any("INT8 dot-product" in w for w in plan.warnings)

    def test_moe_prefill_note_differs_from_dense(self):
        gpu = hardware.GPUS["rtx-3090-24g"]
        moe = planner.plan(gpu, models.MODELS["gpt-oss-120b"], "q4_k_m", 8192)
        dense = planner.plan(gpu, models.MODELS["llama-3.1-8b"], "q4_k_m", 8192)
        assert "expert" in moe.prefill_note
        assert "expert" not in dense.prefill_note


class TestOverhead:
    def test_overhead_grows_with_context_but_not_linearly_enough_to_ignore(self):
        small = planner.overhead_gb(2048)
        large = planner.overhead_gb(131072)
        assert small < large
        assert small > 0


class TestCatalogueIntegrity:
    def test_every_gpu_has_a_positive_bandwidth_and_usable_memory(self):
        """Every entry must declare memory a plan can use: VRAM, or system RAM
        for the CPU-only baselines. Zero of both would make every plan fail."""
        for gpu in hardware.GPUS.values():
            assert gpu.bandwidth_gbs > 0, gpu.key
            assert planner.usable_memory_gb(gpu) > 0, gpu.key

    def test_cpu_only_entries_use_system_ram(self):
        cpu = hardware.GPUS["cpu-ddr5-dual"]
        assert cpu.vram_gb == 0.0
        assert planner.usable_memory_gb(cpu) == cpu.host_ram_gb
        # And a CPU plan must therefore be able to fit something sensible.
        plan = planner.plan(cpu, models.MODELS["llama-3.1-8b"], "q4_k_m", 8192)
        assert plan.fits
        assert "CPU" in plan.gpu.name

    def test_every_model_has_coherent_geometry(self):
        for model in models.MODELS.values():
            assert model.heads % model.kv_heads == 0, model.key
            assert model.kv_heads <= model.heads, model.key
            assert model.kv_bytes_per_token() > 0, model.key

    def test_unified_memory_cards_are_flagged(self):
        assert hardware.GPUS["apple-m4-max-64g"].unified_memory
        assert not hardware.GPUS["rtx-3090-24g"].unified_memory

    def test_pascal_lacks_the_int8_instructions(self):
        """This flag drives the IQ-quant advisory, so it must be right."""
        assert not hardware.GPUS["tesla-p40-24g"].int8_dp4a
        assert hardware.GPUS["rtx-3090-24g"].int8_dp4a
