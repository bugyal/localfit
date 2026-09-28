"""Tests for the Apple Silicon probe path.

The Darwin branch had no coverage, which is exactly how the promised shared-pool
note went missing from the output while the module docstring claimed it. These
tests inject sysctl output rather than reading this machine, so they hold on any
host.

Every case below is a real Apple memory tier: the printed "GB" is GiB, so the
byte figure divided by 1e9 never equals the marketed number.
"""

from __future__ import annotations

import pytest

from localfit import probe

GIB = 1024**3


def _fake_sysctl(monkeypatch, memsize_bytes: int, chip: str):
    canned = {
        ("sysctl", "-n", "hw.memsize"): str(memsize_bytes),
        ("sysctl", "-n", "machdep.cpu.brand_string"): chip,
    }
    monkeypatch.setattr(probe, "_run", lambda cmd: canned.get(tuple(cmd), ""))
    # probe() imports `platform` locally, so patch the stdlib module itself.
    import platform as _platform

    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    return probe.probe()


class TestSharedPoolIsExplained:
    """Unified memory means 'system RAM' and the card's memory are one pool."""

    @pytest.mark.parametrize(
        "gib,chip",
        [
            (16, "Apple M4"),
            (24, "Apple M4 Pro"),
            (64, "Apple M4 Max"),
            (192, "Apple M3 Ultra"),
        ],
    )
    def test_unified_probe_says_the_figures_are_one_pool(self, monkeypatch, gib, chip):
        result = _fake_sysctl(monkeypatch, gib * GIB, chip)
        assert result.gpus, "Apple chip should map to a catalogue entry"
        assert result.gpus[0].unified_memory
        notes = " ".join(result.notes).lower()
        assert "same physical pool" in notes
        assert "not two budgets" in notes

    def test_the_note_appears_once_per_probe_not_per_card(self, monkeypatch):
        result = _fake_sysctl(monkeypatch, 16 * GIB, "Apple M4")
        hits = [n for n in result.notes if "same physical pool" in n]
        assert len(hits) == 1

    def test_the_pool_is_warned_as_whole_allocation_not_free_memory(self, monkeypatch):
        """A 16 GB machine cannot give 16 GB to a model; the note must say so."""
        result = _fake_sysctl(monkeypatch, 16 * GIB, "Apple M4")
        notes = " ".join(result.notes).lower()
        assert "not memory free for inference" in notes


class TestMemoryTierIsResolvedFromMeasurement:
    """The catalogue holds one entry per tier; the measured pool picks it."""

    def test_same_tier_is_left_alone_despite_the_gib_gap(self, monkeypatch):
        """A '64 GB' M4 Max measures 68.7 decimal GB — that is the same part.

        Comparing decimal GB to a GiB-labelled catalogue figure reads as a 4.7 GB
        mismatch and would fabricate a tier that does not exist.
        """
        result = _fake_sysctl(monkeypatch, 64 * GIB, "Apple M4 Max")
        gpu = result.gpus[0]
        assert gpu.key == "apple-m4-max-64g"
        assert gpu.vram_gb == 64.0
        assert "different tier" not in gpu.note

    def test_a_genuinely_different_tier_is_corrected(self, monkeypatch):
        """A 48 GB M4 Pro must not be planned against the 24 GB entry."""
        result = _fake_sysctl(monkeypatch, 48 * GIB, "Apple M4 Pro")
        gpu = result.gpus[0]
        # Stored in the catalogue's units (marketed GiB), not decimal GB: every
        # other row is the number on the box, and 44.7 would not match any Mac.
        assert gpu.vram_gb == 48.0
        assert gpu.key != "apple-m4-pro-24g"
        assert "48 GB" in gpu.name
        assert "different tier" in gpu.note
        # Correcting the size must not silently pass off the default tier's
        # bandwidth as measured for this one.
        assert "may not match this one" in gpu.note

    def test_the_corrected_entry_reports_more_usable_memory(self, monkeypatch):
        """The whole point: a 48 GB machine must plan against 48 GB, not 24.

        A plan that under-reports capacity refuses models that actually run.
        """
        from localfit.planner import usable_memory_gb

        result = _fake_sysctl(monkeypatch, 48 * GIB, "Apple M4 Pro")
        assert usable_memory_gb(result.gpus[0]) == 48.0

    def test_the_correction_is_reported_to_the_user(self, monkeypatch):
        result = _fake_sysctl(monkeypatch, 48 * GIB, "Apple M4 Pro")
        assert any("different tier" in n for n in result.notes)

    def test_matching_respects_specificity(self, monkeypatch):
        """'m4 pro' must beat a bare 'm4' — the table is order-sensitive."""
        pro = _fake_sysctl(monkeypatch, 24 * GIB, "Apple M4 Pro").gpus[0]
        plain = _fake_sysctl(monkeypatch, 16 * GIB, "Apple M4").gpus[0]
        assert pro.key != plain.key
        assert pro.name != plain.name

    def test_a_zero_ram_reading_does_not_invent_a_tier(self, monkeypatch):
        result = _fake_sysctl(monkeypatch, 0, "Apple M4")
        gpu = result.gpus[0]
        assert gpu.key == "apple-m4-16g"
        assert "different tier" not in gpu.note


class TestUnmappableAppleChip:
    def test_unknown_chip_is_asked_about_not_guessed(self, monkeypatch):
        result = _fake_sysctl(monkeypatch, 16 * GIB, "Apple M99")
        assert not result.gpus
        assert any("--gpu" in u for u in result.unknown)
