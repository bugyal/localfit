"""Tests for the calibration audit.

The point of these is that the package's honesty claims are enforced: a plan must
not present an unvalidated estimate as if it were calibrated, and the calibration
figures must be computed from held-out anchors rather than reported from a fit.
"""

from __future__ import annotations

import json

import pytest

from localfit import cli, hardware, models, perf, planner


class TestCalibrationMath:
    def test_every_architecture_is_reported_including_empty_ones(self):
        rows = perf.calibrate()
        reported = {r.arch for r in rows}
        expected = {g.arch for g in hardware.GPUS.values()}
        assert expected <= reported, "architectures must not be silently omitted"

    def test_architectures_with_matching_anchors_have_samples(self):
        by_arch = {r.arch: r for r in perf.calibrate()}
        assert by_arch["nvidia-pascal"].samples == 4
        assert by_arch["apple-silicon"].samples == 4

    def test_unmeasured_architecture_reports_zero_not_a_guess(self):
        by_arch = {r.arch: r for r in perf.calibrate()}
        assert by_arch["amd-rdna3"].samples == 0
        assert by_arch["amd-rdna3"].verdict == "no measurements for this architecture"

    def test_mean_error_is_the_mean_of_the_per_anchor_errors(self):
        rows = perf.calibrate("apple-silicon")
        assert rows[0].samples == 4
        # All four Apple anchors are predicted within 20% by the fitted factor.
        assert rows[0].mean_abs_error_pct < 20
        assert rows[0].worst_error_pct >= rows[0].mean_abs_error_pct

    def test_worst_case_is_named(self):
        rows = perf.calibrate("nvidia-pascal")
        assert rows[0].worst_error_pct > 0
        assert rows[0].worst_case

    def test_verdict_thresholds_are_applied(self):
        assert perf.Calibration("x", 0, 0.0, 0.0, "").verdict.startswith("no measurements")
        assert perf.Calibration("x", 1, 10.0, 12.0, "y").verdict == "well calibrated"
        assert perf.Calibration("x", 1, 25.0, 30.0, "y").verdict == "reasonable"
        assert "poor" in perf.Calibration("x", 1, 60.0, 90.0, "y").verdict

    def test_calibration_does_not_depend_on_a_model_import_cycle(self):
        """perf must be importable without the catalogues loaded first."""
        import importlib
        import sys

        for name in [n for n in sys.modules if n.startswith("localfit")]:
            del sys.modules[name]
        module = importlib.import_module("localfit.perf")
        assert module.EFFICIENCY
        assert module.calibrate("cpu")[0].samples == 0


class TestCalibrationCommand:
    def test_reports_a_table(self, capsys):
        assert cli.main(["calibration"]) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "| architecture |" in out
        assert "held out" in out

    def test_names_the_unvalidated_architectures(self, capsys):
        cli.main(["calibration"])
        out = capsys.readouterr().out
        assert "No measurements exist for" in out
        assert "amd-rdna3" in out

    def test_json_is_well_formed(self, capsys):
        cli.main(["calibration", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert any(row["arch"] == "nvidia-pascal" for row in data)
        for row in data:
            assert "verdict" in row and "samples" in row

    def test_single_arch_filter(self, capsys):
        cli.main(["calibration", "--arch", "apple-silicon", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert len(data) == 1
        assert data[0]["arch"] == "apple-silicon"


class TestHonestyInPlans:
    def test_unvalidated_architecture_says_so_in_the_plan(self):
        from localfit import report

        plan = planner.plan(
            hardware.GPUS["rx-7900-xtx-24g"], models.MODELS["qwen3-27b"], "q4_k_m", 8192
        )
        text = report.render_plan(plan)
        assert "no measured runs exist" in text
        assert "unvalidated" in text

    def test_calibrated_architecture_reports_its_track_record(self):
        from localfit import report

        # Apple M4 Max has anchors, but not for this model — so the plan must
        # estimate and disclose the track record behind that estimate.
        plan = planner.plan(
            hardware.GPUS["apple-m4-max-64g"], models.MODELS["qwen3-27b"], "q4_k_m", 8192
        )
        assert plan.measured is None
        text = report.render_plan(plan)
        assert "track record" in text
        assert "well calibrated" in text

    def test_measured_plans_do_not_carry_a_calibration_note(self):
        """A measurement needs no accuracy disclaimer."""
        from localfit import report

        plan = planner.plan(
            hardware.GPUS["tesla-p40-24g"], models.MODELS["llama-3.2-3b"], "q4_k_m", 8192
        )
        assert plan.measured is not None
        text = report.render_plan(plan)
        assert "track record" not in text


class TestMxfp4Advisory:
    def test_raised_on_architectures_without_the_fp4_path(self):
        for key in ("tesla-p40-24g", "rtx-3090-24g", "tesla-t4-16g"):
            plan = planner.plan(
                hardware.GPUS[key], models.MODELS["gpt-oss-120b"], "mxfp4", 8192
            )
            assert any("MXFP4" in w for w in plan.warnings), key

    def test_not_raised_on_blackwell(self):
        plan = planner.plan(
            hardware.GPUS["rtx-5090-32g"], models.MODELS["gpt-oss-120b"], "mxfp4", 8192
        )
        assert not any("MXFP4" in w for w in plan.warnings)

    def test_mxfp4_is_a_resolvable_quant_with_aliases(self):
        from localfit import quants

        assert quants.resolve("mxfp4").name == "mxfp4"
        assert quants.resolve("mx4").name == "mxfp4"
        assert quants.resolve("fp4").name == "mxfp4"
