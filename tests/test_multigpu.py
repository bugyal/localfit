"""Tests for multi-GPU planning.

The behaviour under test is the one that generated this work order: allocation
happens in whole layers, so the split never balances and some VRAM is unusable no
matter what ratios you pass. These tests pin the arithmetic that explains it.
"""

from __future__ import annotations

import pytest

from localfit import hardware, models, multigpu, planner, quants


class TestLayerGranularity:
    def test_layer_size_is_a_whole_number_of_layers(self):
        """A layer's size is the per-layer weights plus its share of the cache.

        Allocation must be quantised to this unit, because that is what
        llama.cpp's --split-mode layer does.
        """
        model = models.MODELS["llama-3.1-8b"]
        quant = quants.resolve("q4_k_m")
        size = multigpu.layer_bytes(model, quant)
        expected = planner.weight_gb(model, quant) * 1e9 / model.layers
        assert size == pytest.approx(expected, rel=1e-6)

    def test_bigger_model_has_bigger_layers(self):
        small = multigpu.layer_bytes(
            models.MODELS["llama-3.2-3b"], quants.resolve("q4_k_m")
        )
        large = multigpu.layer_bytes(
            models.MODELS["llama-3.1-70b"], quants.resolve("q4_k_m")
        )
        assert large > small


class TestBalancedCase:
    def test_identical_cards_split_evenly(self):
        """Equal cards should end up equally *used*, not equally layered.

        The main card holds the whole KV cache and graph, so it takes fewer
        layers; that is correct, not imbalance.
        """
        gpus = [hardware.GPUS["rtx-3090-24g"]] * 2
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "q4_k_m", 8192)
        assert plan.cpu_layers == 0
        wastes = [g.vram_gb - s.gb_assigned for g, s in zip(gpus, plan.splits)]
        # Waste must be within one layer of each other, and the main card must
        # hold no more layers than the other.
        assert max(wastes) - min(wastes) < multigpu.layer_bytes(
            models.MODELS["llama-3.1-70b"], quants.resolve("q4_k_m")
        ) / 1e9
        main = [s for s in plan.splits if s.is_main][0]
        other = [s for s in plan.splits if not s.is_main][0]
        assert main.layers_assigned <= other.layers_assigned

    def test_identical_cards_minimise_the_worst_leftover(self):
        """The rebalance pass targets the largest leftover, which is the symptom."""
        gpus = [hardware.GPUS["rtx-3090-24g"]] * 2
        model = models.MODELS["llama-3.1-70b"]
        quant = quants.resolve("q4_k_m")
        plan = multigpu.plan_multi(gpus, model, "q4_k_m", 8192)
        assert plan.cpu_layers == 0
        # Waste can never be beaten down below what the KV cache forces: the main
        # card must hold the whole cache, so its leftover is bounded below by the
        # cache size modulo a layer. Balanced means the two cards are within a
        # layer of each other, not that each is under a layer.
        wastes = [g.vram_gb - s.gb_assigned for g, s in zip(gpus, plan.splits)]
        layer_gb = multigpu.layer_bytes(model, quant) / 1e9
        assert max(wastes) - min(wastes) < layer_gb
        assert plan.imbalance_gb < layer_gb
        # And no card may be over budget.
        for split in plan.splits:
            assert split.gb_assigned <= split.gpu.vram_gb + 1e-9

    def test_identical_cards_have_no_bandwidth_mismatch_warning(self):
        gpus = [hardware.GPUS["rtx-3090-24g"]] * 2
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "q4_k_m", 8192)
        assert not any("Mixed-bandwidth" in w for w in plan.warnings)


class TestImbalance:
    def test_unequal_cards_leave_vram_unused(self):
        """The evidence #1 symptom: unusable memory even when it all 'fits'."""
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        assert plan.unusable_gb > 0
        # Each card's leftover is under one layer size, which is the point: you
        # cannot do better than this by tuning ratios.
        for split, gpu in zip(plan.splits, gpus):
            assert gpu.vram_gb - split.gb_assigned < multigpu.layer_bytes(
                models.MODELS["llama-3.1-70b"], quants.resolve("iq3_m")
            ) / 1e9

    def test_allocation_never_exceeds_a_card_budget(self):
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        for split, gpu in zip(plan.splits, gpus):
            assert split.gb_assigned <= gpu.vram_gb + 1e-9, split.gpu.name

    def test_mixed_bandwidth_warning_is_raised(self):
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        assert any("Mixed-bandwidth" in w for w in plan.warnings)
        assert any("does not add bandwidth" in w for w in plan.warnings)

    def test_mixed_architecture_warning_is_raised(self):
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["tesla-p40-24g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "q4_k_m", 8192)
        assert any("architectures" in w for w in plan.warnings)


class TestRatioAdvice:
    def test_tuned_ratio_differs_from_the_naive_ratio(self):
        """The split that matches the layer breakdown is not the memory ratio.

        This is the actionable version of the diagnosis: it is why nudging the
        ratio by 0.002 swings allocation by gigabytes.
        """
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        assert plan.actual_split
        assert plan.actual_split != plan.tensor_split
        assert sum(plan.actual_split) == pytest.approx(1.0)

    def test_ratios_sum_to_one(self):
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        assert sum(plan.tensor_split) == pytest.approx(1.0, abs=1e-3)

    def test_warning_names_both_ratios(self):
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        combined = " ".join(plan.warnings)
        assert "Tensor-split takes ratios" in combined
        assert "Use the second one" in combined


class TestFlags:
    def test_ngl_matches_the_layers_that_actually_fit(self):
        """`-ngl 99` is the common advice and is wrong whenever layers spill."""
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "q8_0", 8192)
        assert plan.cpu_layers > 0
        flags = " ".join(plan.flags)
        assert f"-ngl {plan.gpu_layers}" in flags
        assert "-ngl 99" not in flags

    def test_ngl_is_full_when_everything_fits(self):
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        assert plan.cpu_layers == 0
        model = models.MODELS["llama-3.1-70b"]
        assert f"-ngl {model.layers}" in " ".join(plan.flags)

    def test_main_gpu_is_the_largest_card(self):
        gpus = [hardware.GPUS["rtx-3060-12g"], hardware.GPUS["rtx-3090-24g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        main = [s for s in plan.splits if s.is_main]
        assert len(main) == 1
        assert main[0].gpu.key == "rtx-3090-24g"

    def test_main_gpu_index_is_in_the_flags(self):
        gpus = [hardware.GPUS["rtx-3060-12g"], hardware.GPUS["rtx-3090-24g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        assert "--main-gpu 1" in " ".join(plan.flags)

    def test_flags_are_copy_pasteable(self):
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        joined = " ".join(plan.flags)
        for expected in ("--split-mode", "--tensor-split", "--main-gpu", "-ngl"):
            assert expected in joined
        # No ratio should render as a bare float repr.
        assert "0.666666666" not in joined


class TestKvPlacement:
    def test_kv_cache_lands_on_the_main_card(self):
        """It is why the main card holds less than its VRAM suggests."""
        gpus = [hardware.GPUS["rtx-3060-12g"], hardware.GPUS["rtx-3090-24g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        main = [s for s in plan.splits if s.is_main][0]
        other = [s for s in plan.splits if not s.is_main][0]
        kv = planner.kv_gb(models.MODELS["llama-3.1-70b"], 8192)
        # The main card's assignment includes the cache on top of its layers.
        layer = multigpu.layer_bytes(
            models.MODELS["llama-3.1-70b"], quants.resolve("iq3_m")
        ) / 1e9
        assert main.gb_assigned == pytest.approx(
            main.layers_assigned * layer + kv + planner.overhead_gb(8192), rel=1e-3
        )
        assert other.gb_assigned == pytest.approx(other.layers_assigned * layer, rel=1e-3)

    def test_kv_warning_names_the_main_card(self):
        gpus = [hardware.GPUS["rtx-3060-12g"], hardware.GPUS["rtx-3090-24g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        assert any("KV cache" in w for w in plan.warnings)
        assert any("NVIDIA RTX 3090" in w for w in plan.warnings)


class TestDraftModelGuidance:
    def test_speculative_decoding_pitfall_is_explained(self):
        """Evidence #1 tried --spec-draft-device to no effect; the report says why."""
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        combined = " ".join(plan.warnings)
        assert "spec-draft-device" in combined
        assert "not part of the tensor split" in combined


class TestVerifySplit:
    def test_clean_plan_has_no_problems(self):
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        assert multigpu.verify_split(plan) == []

    def test_spilled_layers_are_reported_as_a_problem(self):
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "q8_0", 8192)
        problems = multigpu.verify_split(plan)
        assert problems
        assert any("system RAM" in p for p in problems)

    def test_no_contradiction_between_verdict_and_problems(self):
        """A plan reporting FITS must not also list a problem."""
        cases = [
            (["rtx-3090-24g", "rtx-3060-12g"], "llama-3.1-70b", "iq3_m", 8192),
            (["rtx-3090-24g", "rtx-3060-12g"], "llama-3.1-70b", "q8_0", 8192),
            (["rtx-3090-24g", "rtx-3090-24g"], "llama-3.1-70b", "q4_k_m", 8192),
            (["tesla-p40-24g", "tesla-p40-24g"], "llama-3.1-8b", "q4_k_m", 32768),
        ]
        for keys, model_key, quant, ctx in cases:
            gpus = [hardware.GPUS[k] for k in keys]
            plan = multigpu.plan_multi(gpus, models.MODELS[model_key], quant, ctx)
            problems = multigpu.verify_split(plan)
            assert plan.fits == (not problems), (
                f"{keys} {model_key} {quant}: fits={plan.fits} but problems={problems}"
            )

    def test_tiny_card_cannot_hold_a_layer_at_all(self):
        """A degenerate case worth answering rather than crashing on."""
        gpus = [hardware.GPUS["gtx-1660-ti-6g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-70b"], "q4_k_m", 8192)
        assert not plan.fits
        assert plan.cpu_layers > 0


class TestPascalAcrossCards:
    def test_iq_advisory_appears_in_multi_gpu_too(self):
        gpus = [hardware.GPUS["tesla-p40-24g"], hardware.GPUS["tesla-p40-24g"]]
        plan = multigpu.plan_multi(gpus, models.MODELS["llama-3.1-8b"], "iq4_xs", 8192)
        assert any("IQ" in w for w in plan.warnings)


class TestSuggestTensorSplit:
    def test_ratios_proportional_to_free_memory(self):
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        ratios = multigpu.suggest_tensor_split(gpus)
        assert ratios[0] > ratios[1]
        assert ratios == pytest.approx([24 / 36, 12 / 36], rel=1e-3)

    def test_headroom_is_respected(self):
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        ratios = multigpu.suggest_tensor_split(gpus, headroom=[12.0, 0.0])
        # With half the 3090 spoken for, the split should be much closer to even.
        assert ratios[0] == pytest.approx(12 / 24, rel=1e-3)
