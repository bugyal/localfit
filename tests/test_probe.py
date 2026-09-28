"""Tests for hardware probing and log ingestion.

The probe is tested against injected command output rather than this machine's
hardware, because a test that only passes on the build host proves nothing about
the machines users have.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from localfit import probe

FIXTURES = Path(__file__).parent / "fixtures"


class TestNameMatching:
    def test_exact_names_map_to_catalogue_keys(self):
        assert probe.match_gpu_by_name("NVIDIA GeForce RTX 3090").key == "rtx-3090-24g"
        assert probe.match_gpu_by_name("Tesla P40").key == "tesla-p40-24g"
        assert probe.match_gpu_by_name("AMD Radeon RX 7900 XTX").key == "rx-7900-xtx-24g"

    def test_more_specific_pattern_wins(self):
        """'4060 Ti' must not be matched by the plain '4060' pattern."""
        assert probe.match_gpu_by_name("NVIDIA GeForce RTX 4060 Ti").key == "rtx-4060-ti-16g"
        assert probe.match_gpu_by_name("NVIDIA GeForce RTX 4060").key == "rtx-4060-8g"

    def test_unknown_card_with_vram_becomes_a_synthetic_entry(self):
        """An unlisted card that reports its memory still gets a useful plan."""
        gpu = probe.match_gpu_by_name("NVIDIA RTX 5090 Ti Super", vram_gb=36.0)
        assert gpu is not None
        assert "unlisted" in gpu.name
        assert gpu.vram_gb == 36.0
        assert "Not in the catalogue" in gpu.note

    def test_vram_mismatch_falls_back_to_synthetic(self):
        """A name that does not match its reported memory is not trusted."""
        gpu = probe.match_gpu_by_name("NVIDIA GeForce RTX 3090", vram_gb=12.0)
        # 12 GB is not a 3090, so the name is not believed.
        assert gpu.vram_gb == 12.0

    def test_unrecognised_name_without_vram_is_none(self):
        assert probe.match_gpu_by_name("Some Accelerator 9000") is None


class TestLogParsing:
    def test_reads_the_llamacpp_fixture(self):
        m = probe.parse_log_file(FIXTURES / "llamacpp_server.log")
        assert m.eval_tok_s == pytest.approx(47.78)
        assert m.prompt_tok_s == pytest.approx(562.48)
        assert m.layers_offloaded == 80
        assert m.layers_total == 81
        assert m.context == 8192
        assert m.tensor_split == "0.667,0.333"
        assert m.main_gpu == "0"
        assert m.kv_cache_mib == pytest.approx(546.0)
        assert m.model_path.endswith("llama-3.1-8b-Q4_K_M.gguf")

    def test_reads_per_device_buffers(self):
        m = probe.parse_log_file(FIXTURES / "llamacpp_server.log")
        assert m.per_device_mib == [("CUDA0", 3902.0), ("CUDA1", 2462.0)]

    def test_partial_offload_is_flagged(self):
        """CPU-bound layers mean the tok/s figure understates the card."""
        m = probe.parse_log_file(FIXTURES / "llamacpp_server.log")
        assert any("understates" in w for w in m.warnings)

    def test_device_imbalance_is_flagged(self):
        m = probe.parse_log_file(FIXTURES / "llamacpp_server.log")
        assert any("1440" in w or "imbalance" in w for w in m.warnings)

    def test_balanced_devices_are_not_flagged(self):
        text = (
            "llama_new_context_with_model: CUDA0 model buffer size =  3000.00 MiB\n"
            "llama_new_context_with_model: CUDA1 model buffer size =  3050.00 MiB\n"
            "llama_print_timings: eval time = 1000.00 ms /  100 runs "
            "( 10.00 ms per token, 100.00 tokens per second)\n"
        )
        m = probe.parse_log(text)
        assert m.eval_tok_s == pytest.approx(100.0)
        assert not any("imbalance" in w for w in m.warnings)

    def test_vllm_style_throughput_line(self):
        m = probe.parse_log("Throughput: 42.5 tokens/s\n")
        assert m.eval_tok_s == pytest.approx(42.5)

    def test_empty_log_says_so_rather_than_inventing_a_number(self):
        m = probe.parse_log("nothing useful here")
        assert m.eval_tok_s == 0.0
        assert not m.is_useful
        assert any("No tokens-per-second" in w for w in m.warnings)

    def test_is_useful_when_only_buffers_present(self):
        text = "llama_new_context_with_model: CUDA0 model buffer size =  3000.00 MiB\n"
        m = probe.parse_log(text)
        assert m.is_useful

    def test_single_device_is_not_an_imbalance(self):
        text = (
            "llama_new_context_with_model: CUDA0 model buffer size =  3000.00 MiB\n"
            "llama_print_timings: eval time = 1000.00 ms /  100 runs "
            "( 10.00 ms per token, 100.00 tokens per second)\n"
        )
        m = probe.parse_log(text)
        assert not any("imbalance" in w for w in m.warnings)


class TestContributionRow:
    def test_row_is_written_and_well_formed(self, tmp_path):
        m = probe.parse_log_file(FIXTURES / "llamacpp_server.log")
        out = tmp_path / "row.json"
        row = probe.contribute_row(m, out)
        assert out.exists()
        assert row["schema"] == "localfit.measurement/1"
        assert row["eval_tok_s"] == pytest.approx(47.78)
        # It must be valid JSON on disk, not just in memory.
        import json

        assert json.loads(out.read_text())["schema"] == "localfit.measurement/1"

    def test_nothing_is_uploaded_anywhere(self):
        """The tool must have no network path for contributed data."""
        import localfit

        source = Path(localfit.__file__).parent
        for path in source.glob("*.py"):
            text = path.read_text()
            for banned in ("urllib", "requests", "http.client", "socket", "urlopen"):
                assert banned not in text, f"{path.name} references {banned}"


class TestProbeResultShape:
    def test_to_dict_is_serialisable(self):
        import json

        result = probe.ProbeResult(platform="Linux", system_ram_gb=64.0)
        assert json.loads(json.dumps(result.to_dict()))["platform"] == "Linux"

    def test_summarise_handles_no_gpus(self):
        assert "no accelerator" in probe.summarise_probe(probe.ProbeResult())

    def test_summarise_counts_duplicates(self):
        from localfit import hardware

        result = probe.ProbeResult(
            gpus=[hardware.GPUS["rtx-3090-24g"], hardware.GPUS["rtx-3090-24g"]]
        )
        assert "2x" in probe.summarise_probe(result)


class TestProbeOnThisMachine:
    def test_probe_returns_a_coherent_result_wherever_it_runs(self):
        """Probing must never crash, whatever hardware it finds — or does not."""
        result = probe.probe()
        assert result.platform
        for gpu in result.gpus:
            assert gpu.bandwidth_gbs > 0
            assert gpu.vram_gb > 0
        # If nothing was found, it must be said rather than left blank.
        if not result.gpus:
            assert result.unknown

    def test_probe_does_not_raise_when_tools_are_missing(self, monkeypatch):
        monkeypatch.setattr(probe.shutil, "which", lambda _: None)
        result = probe.probe()
        assert isinstance(result, probe.ProbeResult)
