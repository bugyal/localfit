# localfit

Answers *"what will run on my hardware, and at what speed?"* before you download
anything.

`r/LocalLLaMA` answers this question by hand, every day, one per-GPU thread at a
time — 24 GB card, 64 GB of RAM, will this model fit at the context I need, why is
one of my two cards permanently idle. This tool gives the same answer
deterministically, with the arithmetic shown so you can check it.

```console
$ localfit plan llama-3.1-8b --gpu tesla-p40-24g --quant q4_k_m
# localfit — Llama 3.1 8B @ q4_k_m

## FITS

6.36 GB needed of 24.00 GB usable on NVIDIA Tesla P40 (+17.64 GB headroom).

### Memory

| component | size | of capacity |
|---|---|---|
| weights (q4_k_m) | 4.84 GB | `#####...................` |
| KV cache (8,192 ctx) | 1.07 GB | `#.......................` |
| driver + compute graph | 0.45 GB | `........................` |
| **total** | **6.36 GB** | |
| usable on card | 24.00 GB | |

### Speed

**Measured: 47.8 tok/s** (llama-3.1-8b, q4_k_m, ollama)
  source: tinycomputers.io P40 home lab, 4x P40
  note: fitted entirely on one card; no multi-GPU overhead

The bandwidth ceiling for this configuration is 71.4 tok/s, so the measurement is
using 67% of what the hardware allows.
```

## Install

```console
$ python3.11 -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

Requires Python 3.11 or newer. No runtime dependencies.

## Commands

| command | what it answers |
|---|---|
| `plan <model>` | the largest quantisation that fits, with speed |
| `quants <model>` | every quantisation side by side |
| `probe` | what hardware is on this machine |
| `measure <log>` | what your own run actually did, and contribute a row |
| `hardware` / `models` | the catalogues |
| `calibration` | how accurate the speed estimates have been |

### It finds your hardware

```console
$ localfit probe
# localfit — detected hardware

- platform: Darwin
- system RAM: 17.2 GB

| card | arch | memory | bandwidth |
|---|---|---|---|
| Apple M4 | apple-silicon | 16 GB | 120 GB/s |
```

Omit `--gpu` and `plan` probes for you. Three families are covered, each failing
differently: NVIDIA via `nvidia-smi`, AMD via `rocm-smi`, Apple Silicon via
`sysctl`. Anything it cannot identify is reported as an explicit question rather
than defaulted to a plausible-looking number, because a fit plan computed from
the wrong VRAM figure is worse than no plan.

### It answers the real question, not the literal one

`--quant` is optional. Without it you get the **highest-quality scheme that
fits**, which is what people are actually asking:

```console
$ localfit quants llama-3.1-8b --gpu rtx-4060-8g
# localfit — Llama 3.1 8B on NVIDIA RTX 4060 8GB @ 8,192 context

| quant | bpw | weights | +kv+graph | fits | tok/s | quality |
|---|---|---|---|---|---|---|
| q8_0 | 8.5 | 8.50 GB | 10.02 GB | no | ~15 | +0.002 ppl |
| q6_k | 6.5625 | 6.56 GB | 8.08 GB | no | ~25 | +0.005 ppl |
| q5_k_m | 5.6719 | 5.67 GB | 7.19 GB | yes | ~29 | +0.011 ppl |
| q5_k_s | 5.5156 | 5.52 GB | 7.04 GB | yes | ~30 | +0.022 ppl |
| q4_k_m | 4.8438 | 4.84 GB | 6.36 GB | yes | ~34 | +0.053 ppl |
| q4_k_s | 4.5781 | 4.58 GB | 6.10 GB | yes | ~36 | +0.099 ppl |

Highest-quality scheme that fits: **q5_k_m** (7.19 GB, 0.011 ppl penalty).
```

Note the ordering: smaller quant, faster. A table that broke that rule would
mean the estimate had gone wrong, so a test enforces it for every card.

`--max-loss 0.02` rejects anything lossier than that, because *"it fits"* and
*"it is good enough"* are different questions.

### Multi-GPU: why your two cards never balance

This is the complaint that produced the tool. You tune `--tensor-split`, one card
stays idle, a two-value nudge swings allocation by 2 GB, and it never balances
whatever you try. Worse, 1 GB+ per card stays unusable no matter what values you
pass — because allocation is quantised to whole layers, which is a property of
`--split-mode layer` rather than a bug in your flags:

```console
$ localfit plan llama-3.1-70b --gpu rtx-3090-24g --gpu rtx-3060-12g --context 8192
# localfit — Llama 3.1 70B @ iq3_m across 2 GPUs

## FITS

Weights total 32.30 GB; free across the cards 36.00 GB.

### Layer assignment

| device | card | layers | assigned | budget | unused |
|---|---|---|---|---|---|
| 0 | NVIDIA RTX 3090 (main) | 51 | 23.72 GB | 24.0 GB | 0.28 GB |
| 1 | NVIDIA RTX 3060 12GB | 29 | 11.71 GB | 12.0 GB | 0.29 GB |

### Flags

```
--split-mode layer
--tensor-split 0.637,0.362
--main-gpu 0
-ngl 80
```

### Why it will not balance

- Mixed-bandwidth pair: NVIDIA RTX 3090 has 2.6x the bandwidth of NVIDIA RTX 3060
  12GB. With --split-mode layer the slowest card gates every token, so total
  throughput lands near the slow card's rate rather than the sum. Splitting does
  not add bandwidth across unequal cards; it only adds capacity.
- Tensor-split takes ratios, not gigabytes. The memory-proportional guess is
  0.667,0.333; the ratio that matches the layer breakdown above is 0.637,0.362.
  Use the second one — the first is what leaves a card holding less than it could.
  They differ because layers are indivisible: the split is quantised to whole
  layers, so the ideal ratio is never the tidy memory ratio.
- --main-gpu 0 (NVIDIA RTX 3090) holds the 2.68 GB KV cache for this context,
  which is why its budget is smaller than its VRAM suggests.
- A draft model for speculative decoding is not part of the tensor split:
  --spec-draft-device pins it to one card, and its weights plus its own KV cache
  come out of that card's budget.
```

Three things that tool explains and hand-answers usually miss:

1. **Allocation is quantised to whole layers.** A 70B model has 0.53 GB layers, so
   "unusable VRAM" is unavoidable and its size is computable. Total waste across
   cards is fixed — it is `sum(free) − footprint`, a constant — so the only thing
   worth optimising is the *largest* leftover on a single card, which is what you
   actually see in a monitor. `localfit` rebalances for that specifically.
2. **The ratio that matches the layer breakdown is not the memory ratio.** Both
   are printed, so you can see why the tidy number is wrong.
3. **The KV cache sits on one card** (`--main-gpu`), which is why that card holds
   fewer layers even when the cards are identical.

`-ngl` is computed rather than fixed at 99. Asking for more layers than fit turns
a slow-but-working load into an OOM.

### Read your own logs back

```console
$ localfit measure llama-server.log
# localfit — measured run

- model: /models/llama-3.1-8b-Q4_K_M.gguf
- **generation: 47.78 tok/s** (measured)
- layers offloaded: 80/81
- tensor-split in effect: 0.667,0.333
- KV cache: 546 MiB

### Per-device buffers

| device | model buffer |
|---|---|
| CUDA0 | 3902 MiB |
| CUDA1 | 2462 MiB |

Spread between fullest and emptiest device: 1440 MiB.
```

`--contribute out.json` writes a row you can keep or post. Nothing is uploaded:
the package contains no network code at all, and a test asserts that.

## What is measured, and what is guessed

This is the part worth reading carefully, because the tool's weakest claim is its
speed estimate and it says so in the output rather than hiding it.

- **Measured** figures come from published benchmarks, quoted with their source
  and reproduced exactly. They are never smoothed.
- **Arithmetic** — whether it fits, the KV cache, file size — is exact, given the
  architecture figures. Hand-computable, and the report shows the working.
- **Estimated** speed applies an efficiency factor fitted against the anchors.
  Every plan states its own track record:

```console
$ localfit calibration
| architecture | samples | mean error | worst error | verdict |
|---|---|---|---|---|
| nvidia-pascal | 4 | 59% | 204% | poor — treat the range as a rough order of magnitude |
| apple-silicon | 4 | 12% | 17% | well calibrated |
| nvidia-turing | 3 | 11% | 18% | well calibrated |
| nvidia-ampere | 1 | 17% | 17% | reasonable |
| amd-rdna3 | 0 | - | - | no measurements for this architecture |
...

No measurements exist for: amd-gcn, amd-rdna2, amd-rdna3, amd-rdna4, cpu,
intel-xe, intel-xe2, nvidia-blackwell, nvidia-volta.
```

Nine of fourteen architectures have **no** measured anchor, and a plan for one of
them says so in plain words instead of printing a confident range. AMD RDNA3 is in
that list — so a 7900 XTX user is told their speed figure is unvalidated, which is
true, rather than being handed a number that looks authoritative.

Architectures with a single anchor are also weak evidence, however good the
number looks: one sample cannot separate a real fit from a coincidence. The
`samples` column is there so you can see that for yourself.

If you have one of those cards, `localfit measure` on your own log is exactly how
that column stops being empty.

## Design rules

- **A measurement outranks a model of the hardware.** Where a measured figure
  exists it is printed as measured, and the bandwidth ceiling is shown beside it
  so you can judge it. A test asserts every anchor obeys its own ceiling — it
  caught one published figure that does not, which was dropped rather than
  smoothed.
- **Estimates are labelled, always.** "Estimated", "midpoint", and the track
  record, in the same breath.
- **Unknowns are said out loud.** Unrecognised card, missing driver, absent log
  figure — each becomes an explicit question, never a silent default.
- **No bypasses.** This tool tells you the least-privilege way to make something
  run: a smaller quant, less context, another card. It does not advise disabling
  a check or forcing a load, and a test scans for that language.
- **`does not fit` exits 1.** A real answer, distinguishable from a crash (2).

## Development

```console
$ .venv/bin/python -m pytest tests/ -q
150 passed
```

The suite pins the arithmetic rather than the implementation: the KV formula is
tested against hand-computed values, every anchor is checked against its
bandwidth ceiling and for physical plausibility, quantisation tables are asserted
to be monotonic, and the fit verdict for a given card and model is asserted
directly.

## Layout

```
src/localfit/
  hardware.py    cards, with published bandwidth and capability flags
  models.py      architectures and the KV-cache arithmetic
  quants.py      real bits-per-weight and measured perplexity deltas
  perf.py        measured anchors, efficiency factors, calibration audit
  planner.py     memory arithmetic and the fit verdict
  multigpu.py    layer-granular allocation and the imbalance diagnosis
  probe.py       hardware detection and log ingestion
  report.py      rendering
  cli.py         argument parsing and exit codes
```

## Licence

Apache-2.0.
