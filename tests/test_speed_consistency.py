"""Tests for the speed estimate's internal consistency.

A quantisation table is read as a ranking: smaller quant, faster. If the numbers
disagree with that, the reader stops trusting the whole table — including the
measured rows. These tests enforce the ordering and the provenance of the
correction factor that keeps estimates in the same world as the measurements.
"""

from __future__ import annotations

import pytest

from localfit import hardware, models, perf, planner, quants


def _fitting_plans(gpu_key: str, model_key: str = "llama-3.1-8b", context: int = 8192):
    gpu = hardware.GPUS[gpu_key]
    model = models.MODELS[model_key]
    plans = planner.all_plans(gpu, model, context)
    return [p for p in plans if p.fits and p.quant_name != "f16"], gpu, model


class TestMonotonicity:
    @pytest.mark.parametrize(
        "gpu_key",
        ["tesla-p40-24g", "rtx-3060-12g", "rtx-4060-8g", "rtx-4090-24g", "apple-m4-max-64g"],
    )
    def test_smaller_quant_is_not_slower(self, gpu_key):
        """A 4-bit model must not be predicted slower than the 6-bit one.

        Bytes per token is what decode time is proportional to, so any violation
        means a correction or a penalty has gone wrong.
        """
        fitting, _, _ = _fitting_plans(gpu_key)
        # Order by increasing quality (decreasing bpw) within a family, and check
        # speed rises as the quant gets smaller.
        ordered = sorted(
            [p for p in fitting if p.quant.family in ("k", "legacy")],
            key=lambda p: p.quant.bpw,
            reverse=True,
        )
        previous = None
        for plan in ordered:
            speed = plan.tok_s_mid
            assert speed > 0, f"{gpu_key} {plan.quant_name} has no estimate"
            if previous is not None:
                prev_name, prev_speed = previous
                assert speed >= prev_speed - 1e-6, (
                    f"{gpu_key}: {plan.quant_name} ({speed:.1f}) should be at least "
                    f"as fast as {prev_name} ({prev_speed:.1f})"
                )
            previous = (plan.quant_name, speed)

    def test_estimate_never_exceeds_the_bandwidth_ceiling(self):
        """No estimate may break the memory-bandwidth bound it is derived from."""
        for gpu_key in ("tesla-p40-24g", "rtx-3060-12g", "rtx-4060-8g", "rx-7900-xtx-24g"):
            gpu = hardware.GPUS[gpu_key]
            for plan in planner.all_plans(gpu, models.MODELS["llama-3.1-8b"], 8192):
                if plan.ceiling_tok_s > 0:
                    assert plan.tok_s_mid <= plan.ceiling_tok_s * 1.001, (
                        f"{gpu_key} {plan.quant_name}: {plan.tok_s_mid:.1f} > "
                        f"ceiling {plan.ceiling_tok_s:.1f}"
                    )


class TestCorrectionFactor:
    def test_no_measurement_means_no_correction(self):
        """A card with nothing measured must not have its estimates nudged."""
        factor, samples = perf.gpu_correction("rx-7900-xtx-24g")
        assert factor == 1.0
        assert samples == 0

    def test_correction_pulls_toward_the_measurement(self):
        """For a card with an anchor, the corrected estimate must be closer to it."""
        gpu = hardware.GPUS["rtx-3060-12g"]
        model = models.MODELS["llama-3.1-8b"]
        anchor = perf.find_anchor("rtx-3060-12g", "llama-3.1-8b", "q4_k_m")
        assert anchor is not None
        # With the anchor present the plan reports it directly, so compare the
        # correction's effect on a neighbouring quant instead.
        near = planner.plan(gpu, model, "q4_k_s", 8192)
        uncorrected_factor = perf.efficiency_for(gpu.arch, "k")
        raw = (
            uncorrected_factor
            * gpu.bandwidth_gbs
            * 1e9
            / (model.decode_params_b * 1e9 * quants.resolve("q4_k_s").bytes_per_weight)
        )
        # The corrected estimate must sit closer to the measured 52 tok/s.
        assert abs(near.tok_s_mid - anchor.tok_s) <= abs(raw - anchor.tok_s) + 1e-6

    def test_exotic_quant_family_does_not_distort_k_quants(self):
        """The P40 has an MXFP4 anchor at 0.22 efficiency; it must not drag K-quants."""
        factor_k, samples_k = perf.gpu_correction("tesla-p40-24g", "k")
        factor_mx, _ = perf.gpu_correction("tesla-p40-24g", "mxfp4")
        assert samples_k == 3
        assert factor_k > factor_mx
        # The K-quant correction should be close to 1, since the fitted factor was
        # derived from those very anchors.
        assert 0.85 < factor_k < 1.15

    def test_median_resists_a_single_bad_anchor(self):
        """One wild anchor must not move the correction much."""
        factor, samples = perf.gpu_correction("tesla-p40-24g", "k")
        assert samples == 3
        assert 0.85 < factor < 1.15

    def test_correction_is_disclosed_in_the_plan(self):
        gpu = hardware.GPUS["tesla-p40-24g"]
        plan = planner.plan(gpu, models.MODELS["llama-3.1-8b"], "q4_k_s", 8192)
        assert any("scaled by" in w for w in plan.warnings)

    def test_correction_not_disclosed_when_there_is_nothing_to_scale(self):
        gpu = hardware.GPUS["rx-7900-xtx-24g"]
        plan = planner.plan(gpu, models.MODELS["llama-3.1-8b"], "q4_k_s", 8192)
        assert not any("scaled by" in w for w in plan.warnings)


class TestAnchorHygiene:
    def test_every_anchor_has_a_named_source(self):
        """A measurement without a source is not a measurement."""
        for anchor in perf.ANCHORS:
            assert anchor.source, f"{anchor.gpu}/{anchor.model} has no source"
            assert anchor.engine, f"{anchor.gpu}/{anchor.model} has no engine"

    def test_every_anchor_references_a_real_catalogue_entry(self):
        for anchor in perf.ANCHORS:
            assert anchor.gpu in hardware.GPUS, anchor.gpu
            assert anchor.model in models.MODELS, anchor.model
            assert quants.resolve(anchor.quant)

    def test_implied_efficiency_is_physically_plausible(self):
        """Every anchor must sit in a believable share of its own ceiling.

        A figure implying >85% of theoretical bandwidth is almost certainly a
        mis-transcribed row rather than a real measurement, because no inference
        engine reaches that. This is the check that caught a duplicated 4060 row.
        """
        for anchor in perf.ANCHORS:
            gpu = hardware.GPUS[anchor.gpu]
            model = models.MODELS[anchor.model]
            quant = quants.resolve(anchor.quant)
            ceiling = gpu.bandwidth_gbs * 1e9 / (
                model.decode_params_b * 1e9 * quant.bytes_per_weight
            )
            implied = anchor.tok_s / ceiling
            assert 0.05 < implied < 0.85, (
                f"{anchor.gpu}/{anchor.model}/{anchor.quant} implies {implied:.0%} of "
                "bandwidth, which is not plausible"
            )

    def test_no_duplicate_rows(self):
        """The same configuration measured twice would silently double a weight."""
        seen = set()
        for anchor in perf.ANCHORS:
            key = (anchor.gpu, anchor.model, anchor.quant)
            assert key not in seen, f"duplicate anchor for {key}"
            seen.add(key)
