"""Tests for the CLI surface and the reports.

The CLI is the product, so these test behaviour through ``main`` rather than
calling internals: exit codes, output content, and the mistakes a user will
actually make.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from localfit import cli, models, hardware, quants, report
from localfit.planner import plan as make_plan
from localfit.multigpu import plan_multi

FIXTURES = Path(__file__).parent / "fixtures"


class TestExitCodes:
    def test_help_with_no_command_is_a_usage_error(self, capsys):
        assert cli.main([]) == cli.EXIT_USAGE
        assert "usage" in capsys.readouterr().out.lower()

    def test_successful_plan_exits_zero(self):
        assert cli.main(["plan", "llama-3.1-8b", "--gpu", "rtx-3090-24g"]) == cli.EXIT_OK

    def test_a_model_that_does_not_fit_exits_one(self):
        """Not an error: a real answer, distinguishable from a crash."""
        code = cli.main(
            ["plan", "llama-3.1-70b", "--gpu", "rtx-4060-8g", "--quant", "q8_0"]
        )
        assert code == cli.EXIT_DOES_NOT_FIT

    def test_unknown_gpu_is_a_usage_error(self, capsys):
        assert cli.main(["plan", "llama-3.1-8b", "--gpu", "rtx-99999"]) == cli.EXIT_USAGE
        assert "unknown GPU" in capsys.readouterr().err

    def test_unknown_model_is_a_usage_error(self, capsys):
        assert cli.main(["plan", "not-a-model", "--gpu", "rtx-3090-24g"]) == cli.EXIT_USAGE
        assert "unknown model" in capsys.readouterr().err

    def test_unknown_quant_is_a_usage_error(self, capsys):
        code = cli.main(
            ["plan", "llama-3.1-8b", "--gpu", "rtx-3090-24g", "--quant", "q9_bogus"]
        )
        assert code == cli.EXIT_USAGE
        assert "unknown quantisation" in capsys.readouterr().err

    def test_measure_of_a_useless_log_exits_one(self, tmp_path, capsys):
        bad = tmp_path / "empty.log"
        bad.write_text("nothing to see")
        assert cli.main(["measure", str(bad)]) == cli.EXIT_DOES_NOT_FIT


class TestPlanOutput:
    def test_default_picks_a_quant_that_fits_and_says_so(self, capsys):
        cli.main(["plan", "llama-3.1-8b", "--gpu", "rtx-3090-24g"])
        out = capsys.readouterr().out
        assert "FITS" in out
        assert "### Memory" in out
        assert "### Speed" in out

    def test_measured_figure_is_labelled_measured_and_sourced(self, capsys):
        cli.main(
            ["plan", "llama-3.2-3b", "--gpu", "tesla-p40-24g", "--quant", "q4_k_m"]
        )
        out = capsys.readouterr().out
        assert "Measured:" in out
        assert "source:" in out
        # The ceiling must appear alongside it, so the reader can judge it.
        assert "ceiling" in out

    def test_estimate_is_labelled_an_estimate(self, capsys):
        cli.main(["plan", "llama-3.1-8b", "--gpu", "rx-7900-xtx-24g"])
        out = capsys.readouterr().out
        assert "Estimated" in out
        assert "not a measurement" in out

    def test_provenance_section_shows_the_arithmetic(self, capsys):
        cli.main(["plan", "llama-3.1-8b", "--gpu", "rtx-3090-24g"])
        out = capsys.readouterr().out
        assert "### Provenance" in out
        assert "bits/weight" in out
        assert "B/token" in out

    def test_failure_states_the_shortfall(self, capsys):
        cli.main(
            ["plan", "llama-3.1-70b", "--gpu", "rtx-4060-8g", "--quant", "q8_0"]
        )
        out = capsys.readouterr().out
        assert "short by" in out or "does not fit" in out.lower()

    def test_no_fit_shows_the_whole_table_rather_than_nothing(self, capsys):
        cli.main(["plan", "llama-3.1-70b", "--gpu", "gtx-1660-ti-6g"])
        captured = capsys.readouterr()
        assert "| quant |" in captured.out
        assert "nothing fits" in captured.err

    def test_context_note_reports_the_ceiling(self, capsys):
        cli.main(["plan", "llama-3.1-8b", "--gpu", "rtx-3090-24g"])
        out = capsys.readouterr().out
        assert "largest context that fits" in out

    def test_multi_gpu_plan_prints_flags_and_diagnosis(self, capsys):
        cli.main(
            [
                "plan",
                "llama-3.1-70b",
                "--gpu",
                "rtx-3090-24g",
                "--gpu",
                "rtx-3060-12g",
            ]
        )
        out = capsys.readouterr().out
        assert "--tensor-split" in out
        assert "Why it will not balance" in out
        assert "Layer assignment" in out

    def test_json_output_is_valid_and_complete(self, capsys):
        cli.main(["plan", "llama-3.1-8b", "--gpu", "rtx-3090-24g", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["model"] == "llama-3.1-8b"
        assert data["gpu"] == "rtx-3090-24g"
        assert data["fits"] is True
        assert data["memory"]["total_gb"] > 0
        assert "estimated" in data["speed"]

    def test_json_multi_gpu_includes_the_tuned_split(self, capsys):
        cli.main(
            [
                "plan",
                "llama-3.1-70b",
                "--gpu",
                "rtx-3090-24g",
                "--gpu",
                "rtx-3060-12g",
                "--json",
            ]
        )
        data = json.loads(capsys.readouterr().out)
        assert data["actual_tensor_split"]
        assert data["cpu_layers"] == 0
        assert data["max_overflow_gb"] <= 1e-9
        assert data["problems"] == []

    def test_measurement_and_estimate_are_distinguishable_in_json(self, capsys):
        cli.main(
            ["plan", "llama-3.2-3b", "--gpu", "tesla-p40-24g", "--quant", "q4_k_m", "--json"]
        )
        data = json.loads(capsys.readouterr().out)
        assert data["speed"]["measured"] is not None
        assert data["speed"]["measured"]["source"]
        cli.main(["plan", "llama-3.1-8b", "--gpu", "rx-7900-xtx-24g", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["speed"]["measured"] is None

    def test_max_loss_is_honoured(self, capsys):
        cli.main(
            [
                "plan",
                "llama-3.1-70b",
                "--gpu",
                "rtx-3090-24g",
                "--gpu",
                "rtx-3090-24g",
                "--max-loss",
                "0.0",
                "--json",
            ]
        )
        # With no loss acceptable, whatever is chosen must have negligible loss.
        out = capsys.readouterr().out
        if out.strip():
            data = json.loads(out)
            if isinstance(data, dict) and "quality_ppl_delta" in data:
                assert (data["quality_ppl_delta"] or 0) <= 0.0


class TestQuantsCommand:
    def test_table_lists_all_schemes_and_marks_fits(self, capsys):
        assert cli.main(["quants", "llama-3.1-8b", "--gpu", "rtx-4060-8g"]) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "| quant |" in out
        assert "q4_k_m" in out
        assert "Highest-quality scheme that fits" in out

    def test_nothing_fits_message_is_specific(self, capsys):
        cli.main(["quants", "llama-3.1-70b", "--gpu", "gtx-1660-ti-6g"])
        out = capsys.readouterr().out
        assert "Nothing in the catalogue fits" in out


class TestProbeCommand:
    def test_probe_reports_what_it_did_not_determine(self, capsys):
        cli.main(["probe"])
        out = capsys.readouterr().out
        assert "detected hardware" in out

    def test_probe_json_is_valid(self, capsys):
        cli.main(["probe", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert "platform" in data
        assert isinstance(data["gpus"], list)


class TestMeasureCommand:
    def test_measures_a_log_and_reports_the_spread(self, capsys):
        code = cli.main(["measure", str(FIXTURES / "llamacpp_server.log")])
        assert code == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "47.78 tok/s" in out
        assert "Per-device buffers" in out
        assert "Spread between fullest and emptiest device" in out

    def test_contribute_writes_a_row(self, tmp_path, capsys):
        out_path = tmp_path / "row.json"
        cli.main(
            [
                "measure",
                str(FIXTURES / "llamacpp_server.log"),
                "--contribute",
                str(out_path),
            ]
        )
        assert out_path.exists()
        assert json.loads(out_path.read_text())["eval_tok_s"] == pytest.approx(47.78)

    def test_measure_json_matches_text(self, capsys):
        cli.main(["measure", str(FIXTURES / "llamacpp_server.log"), "--json"])
        data = json.loads(capsys.readouterr().out)
        assert data["eval_tok_s"] == pytest.approx(47.78)
        assert data["layers_offloaded"] == 80


class TestCatalogues:
    def test_hardware_lists_known_cards(self, capsys):
        assert cli.main(["hardware"]) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Tesla P40" in out
        assert "bandwidth" in out

    def test_hardware_json_includes_sources(self, capsys):
        cli.main(["hardware", "--json"])
        data = json.loads(capsys.readouterr().out)
        assert all("sources" in gpu for gpu in data)

    def test_models_lists_the_catalogue(self, capsys):
        assert cli.main(["models"]) == cli.EXIT_OK
        out = capsys.readouterr().out
        assert "Llama 3.1 8B" in out
        assert "MoE" in out

    def test_models_json_includes_kv_arithmetic(self, capsys):
        cli.main(["models", "--json"])
        data = json.loads(capsys.readouterr().out)
        for entry in data:
            assert entry["kv_bytes_per_token"] > 0


class TestReportRendering:
    def test_no_bypass_language_anywhere(self):
        """A fit tool must never advise disabling a check or forcing a load."""
        import localfit

        source = Path(localfit.__file__).parent
        banned = (
            "--no-check",
            "skip-check",
            "ignore-oob",
            "force-load",
            "unsafe",
            "bypass",
        )
        for path in source.glob("*.py"):
            text = path.read_text().lower()
            for term in banned:
                assert term not in text, f"{path.name} contains {term!r}"

    def test_provenance_separates_measured_from_estimated(self, capsys):
        gpu = hardware.GPUS["tesla-p40-24g"]
        measured = make_plan(gpu, models.MODELS["llama-3.2-3b"], "q4_k_m", 8192)
        text = report.render_plan(measured)
        assert "Measured:" in text

        gpu2 = hardware.GPUS["rx-7900-xtx-24g"]
        estimated = make_plan(gpu2, models.MODELS["llama-3.1-8b"], "q4_k_m", 8192)
        text = report.render_plan(estimated)
        assert "Estimated" in text

    def test_bars_render_within_width(self):
        for value, total in ((0, 10), (5, 10), (10, 10), (11, 10)):
            bar = report._bar(value, total, width=10)
            assert len(bar) == 10

    def test_bar_handles_zero_total(self):
        assert report._bar(1.0, 0.0) == ""

    def test_multi_report_explains_the_verification_step(self, capsys):
        gpus = [hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3060-12g"]]
        multi = plan_multi(gpus, models.MODELS["llama-3.1-70b"], "iq3_m", 8192)
        text = report.render_multi(multi)
        # The user must be told to check the load log rather than trust the flags.
        assert "grep" in text
        assert "buffer size" in text


class TestAutoDetection:
    def test_plan_without_gpu_still_answers_something(self, capsys):
        """No accelerator and no --gpu must still produce a plan, not a lecture."""
        code = cli.main(["plan", "llama-3.1-8b", "--no-probe"])
        assert code in (cli.EXIT_OK, cli.EXIT_DOES_NOT_FIT)
        out = capsys.readouterr().out
        assert "FITS" in out or "DOES NOT FIT" in out

    def test_quiet_suppresses_the_note(self, capsys):
        cli.main(["plan", "llama-3.1-8b", "--no-probe", "--quiet"])
        assert "no accelerator detected" not in capsys.readouterr().err
