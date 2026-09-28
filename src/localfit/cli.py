"""localfit CLI.

    localfit plan <model> --gpu <card> [--context N]     # fit plan
    localfit plan <model> --gpu a --gpu b                # multi-GPU
    localfit quants <model> --gpu <card>                 # compare quantisations
    localfit probe                                       # detect this machine
    localfit measure <logfile> [--contribute out.json]   # read your own run
    localfit hardware | localfit models                  # catalogues

Exit codes: 0 success, 1 the plan does not fit (a real answer, not an error),
2 usage error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, hardware, models, quants
from .multigpu import plan_multi, verify_split
from .planner import all_plans, largest_fitting_quant, max_context, plan
from .probe import contribute_row, parse_log_file, probe, summarise_probe
from .report import (
    render_all_plans,
    render_hardware_catalogue,
    render_measurement,
    render_model_catalogue,
    render_multi,
    render_plan,
    render_probe,
)

EXIT_OK = 0
EXIT_DOES_NOT_FIT = 1
EXIT_USAGE = 2


class UsageError(Exception):
    """A mistake in how the tool was invoked, reported without a traceback."""


def _resolve_gpu(key: str):
    try:
        return hardware.resolve(key)
    except KeyError:
        raise UsageError(
            f"unknown GPU {key!r}. Known keys: "
            + ", ".join(hardware.suggest_keys(key))
            + "  (see `localfit hardware`)"
        ) from None


def _resolve_model(key: str):
    try:
        return models.resolve(key)
    except KeyError:
        raise UsageError(
            f"unknown model {key!r}. Known keys: "
            + ", ".join(models.suggest_keys(key))
            + "  (see `localfit models`)"
        ) from None


def _resolve_quant(key: str) -> str:
    try:
        return quants.resolve(key).name
    except KeyError:
        raise UsageError(
            f"unknown quantisation {key!r}. Try one of: {quants.suggest_name(key)}"
        ) from None


def cmd_plan(args) -> int:
    model = _resolve_model(args.model)

    if not args.gpu:
        # No --gpu: probe, or fall back to the catalogue's CPU entry so the user
        # still gets an answer instead of a usage lecture.
        detected = probe() if not args.no_probe else None
        if detected and detected.gpus:
            gpus = detected.gpus
            if not args.quiet:
                print(f"# detected: {summarise_probe(detected)}", file=sys.stderr)
        else:
            gpus = [hardware.resolve("cpu-ddr5-dual")]
            if not args.quiet:
                print(
                    "# no accelerator detected; planning against dual-channel DDR5 "
                    "(pass --gpu to override)",
                    file=sys.stderr,
                )
    else:
        gpus = [_resolve_gpu(key) for key in args.gpu]

    context = args.context
    # Validate any user-supplied quantisation through the same path as everything
    # else, so a typo is a usage error rather than an uncaught KeyError.
    quant_name = _resolve_quant(args.quant) if args.quant else None

    if len(gpus) > 1:
        if quant_name is None:
            single = largest_fitting_quant(
                hardware.Gpu(
                    key="aggregate",
                    name="aggregate",
                    arch=gpus[0].arch,
                    vram_gb=sum(g.vram_gb for g in gpus),
                    bandwidth_gbs=max(g.bandwidth_gbs for g in gpus),
                    unified_memory=all(g.unified_memory for g in gpus),
                ),
                model,
                context,
            )
            quant_name = single.quant_name if single else "q4_k_m"
        multi = plan_multi(gpus, model, quant_name, context)
        if args.json:
            print(json.dumps(_multi_to_dict(multi), indent=2))
        else:
            print(render_multi(multi))
        problems = verify_split(multi)
        return EXIT_DOES_NOT_FIT if problems else EXIT_OK

    gpu = gpus[0]

    if quant_name:
        result = plan(gpu, model, quant_name, context, headroom_gb=args.headroom)
        if args.json:
            print(json.dumps(_plan_to_dict(result), indent=2))
        else:
            print(render_plan(result))
        return EXIT_OK if result.fits else EXIT_DOES_NOT_FIT

    # No quant named: find the best one that fits, which is the question people
    # actually ask. If none fits, still print the table so they can see how far off.
    best = largest_fitting_quant(
        gpu, model, context, headroom_gb=args.headroom, max_loss=args.max_loss
    )
    if best is not None:
        if args.json:
            print(json.dumps(_plan_to_dict(best), indent=2))
        else:
            print(render_plan(best))
            ceiling_ctx = max_context(gpu, model, best.quant_name, args.headroom)
            if ceiling_ctx and ceiling_ctx < model.context_max:
                print(
                    f"\nNote: at {best.quant_name}, the largest context that fits is "
                    f"{ceiling_ctx:,} tokens (model supports {model.context_max:,})."
                )
        return EXIT_OK

    if not args.quiet:
        print(
            f"# nothing fits at {context:,} context; showing the full range",
            file=sys.stderr,
        )
    plans = all_plans(gpu, model, context, headroom_gb=args.headroom)
    if args.json:
        print(json.dumps([_plan_to_dict(p) for p in plans], indent=2))
    else:
        print(render_all_plans(plans, model, gpu, context))
    return EXIT_DOES_NOT_FIT


def cmd_quants(args) -> int:
    model = _resolve_model(args.model)
    gpu = _resolve_gpu(args.gpu[0]) if args.gpu else _auto_gpu(args)
    plans = all_plans(gpu, model, args.context, headroom_gb=args.headroom)
    if args.json:
        print(json.dumps([_plan_to_dict(p) for p in plans], indent=2))
    else:
        print(render_all_plans(plans, model, gpu, args.context))
        ceiling_ctx = max_context(gpu, model, "q4_k_m", args.headroom)
        print(
            f"\nContext headroom at Q4_K_M: {ceiling_ctx:,} tokens fits "
            f"(of {model.context_max:,} supported)."
        )
    return EXIT_OK


def cmd_probe(args) -> int:
    result = probe()
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(render_probe(result))
    return EXIT_OK if result.gpus else EXIT_DOES_NOT_FIT


def cmd_measure(args) -> int:
    # A missing or unreadable log is a usage mistake, not a crash: without this
    # the FileNotFoundError escapes as a raw traceback, which reads as a bug in
    # the tool rather than a wrong path.
    path = Path(args.logfile)
    if not path.exists():
        raise UsageError(f"no such log file: {args.logfile}")
    if path.is_dir():
        raise UsageError(f"{args.logfile} is a directory, not a log file")
    try:
        measurement = parse_log_file(path)
    except OSError as exc:
        raise UsageError(f"could not read {args.logfile}: {exc.strerror or exc}") from exc
    if args.json:
        print(json.dumps(measurement.to_dict(), indent=2))
    else:
        print(render_measurement(measurement))
    if args.contribute:
        row = contribute_row(measurement, args.contribute)
        print(f"\nWrote contribution row to {args.contribute}", file=sys.stderr)
        print(f"  generation: {row['eval_tok_s']} tok/s", file=sys.stderr)
    return EXIT_OK if measurement.is_useful else EXIT_DOES_NOT_FIT


def cmd_calibration(args) -> int:
    """Report how well the speed estimator reproduces the measured anchors."""
    from . import perf

    rows = perf.calibrate(args.arch)
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "arch": c.arch,
                        "samples": c.samples,
                        "mean_abs_error_pct": round(c.mean_abs_error_pct, 1),
                        "worst_error_pct": round(c.worst_error_pct, 1),
                        "worst_case": c.worst_case,
                        "verdict": c.verdict,
                    }
                    for c in rows
                ],
                indent=2,
            )
        )
        return EXIT_OK
    print("# localfit — speed-estimate calibration")
    print()
    print(
        "The tok/s figures for configurations with no direct measurement are "
        "estimates. This is their track record against the measurements they were "
        "fitted on, each predicted with its own anchor held out."
    )
    print()
    print("| architecture | samples | mean error | worst error | verdict |")
    print("|---|---|---|---|---|")
    for c in sorted(rows, key=lambda r: -r.mean_abs_error_pct):
        mean = f"{c.mean_abs_error_pct:.0f}%" if c.samples else "-"
        worst = f"{c.worst_error_pct:.0f}%" if c.samples else "-"
        print(f"| {c.arch} | {c.samples} | {mean} | {worst} | {c.verdict} |")
    print()
    uncalibrated = [c.arch for c in rows if c.samples == 0]
    if uncalibrated:
        print(
            "No measurements exist for: "
            + ", ".join(uncalibrated)
            + ". Estimates for those architectures are not validated by anything."
        )
    return EXIT_OK


def cmd_hardware(args) -> int:
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "key": g.key,
                        "name": g.name,
                        "arch": g.arch,
                        "vram_gb": g.vram_gb,
                        "bandwidth_gbs": g.bandwidth_gbs,
                        "tensor_cores": g.tensor_cores,
                        "unified_memory": g.unified_memory,
                        "note": g.note,
                        "sources": list(g.sources),
                    }
                    for g in sorted(hardware.GPUS.values(), key=lambda g: -g.bandwidth_gbs)
                ],
                indent=2,
            )
        )
    else:
        print(render_hardware_catalogue())
    return EXIT_OK


def cmd_models(args) -> int:
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "key": m.key,
                        "name": m.name,
                        "params_b": m.params_b,
                        "active_params_b": m.active_params_b,
                        "layers": m.layers,
                        "kv_heads": m.kv_heads,
                        "head_dim": m.head_dim,
                        "context_max": m.context_max,
                        "kv_bytes_per_token": m.kv_bytes_per_token(),
                        "note": m.note,
                    }
                    for m in sorted(models.MODELS.values(), key=lambda m: m.params_b)
                ],
                indent=2,
            )
        )
    else:
        print(render_model_catalogue())
    return EXIT_OK


def _auto_gpu(args):
    detected = probe()
    if detected.gpus:
        if not args.quiet:
            print(f"# detected: {summarise_probe(detected)}", file=sys.stderr)
        return detected.gpus[0]
    return hardware.resolve("cpu-ddr5-dual")


def _plan_to_dict(result) -> dict:
    return {
        "model": result.model.key,
        "model_name": result.model.name,
        "gpu": result.gpu.key,
        "gpu_name": result.gpu.name,
        "quant": result.quant_name,
        "context": result.context,
        "fits": result.fits,
        "memory": {
            "weights_gb": round(result.weights_gb, 3),
            "kv_gb": round(result.kv_gb, 3),
            "overhead_gb": round(result.overhead_gb, 3),
            "total_gb": round(result.total_gb, 3),
            "capacity_gb": round(result.capacity_gb, 3),
            "headroom_gb": round(result.headroom_gb, 3),
        },
        "resident_fraction": round(result.resident_fraction, 3),
        "speed": {
            "measured": (
                {
                    "tok_s": result.measured.tok_s,
                    "engine": result.measured.engine,
                    "source": result.measured.source,
                }
                if result.measured
                else None
            ),
            "estimated": {
                "low": round(result.tok_s_low, 1),
                "mid": round(result.tok_s_mid, 1),
                "high": round(result.tok_s_high, 1),
            },
            "ceiling_tok_s": round(result.ceiling_tok_s, 1),
            "prefill_note": result.prefill_note,
        },
        "quality_ppl_delta": result.quality_delta,
        "warnings": result.warnings,
    }


def _multi_to_dict(result) -> dict:
    return {
        "model": result.model.key,
        "quant": result.quant_name,
        "context": result.context,
        "fits": result.fits,
        "tensor_split": [round(r, 4) for r in result.tensor_split],
        "actual_tensor_split": [round(r, 4) for r in result.actual_split],
        "cpu_layers": result.cpu_layers,
        "gpu_layers": result.gpu_layers,
        "unusable_gb": round(result.unusable_gb, 3),
        "max_overflow_gb": round(result.max_overflow_gb, 3),
        "flags": result.flags,
        "devices": [
            {
                "index": s.index,
                "gpu": s.gpu.key,
                "name": s.gpu.name,
                "layers": s.layers_assigned,
                "assigned_gb": round(s.gb_assigned, 3),
                "vram_gb": s.gpu.vram_gb,
                "unused_gb": round(s.gpu.vram_gb - s.gb_assigned, 3),
                "is_main": s.is_main,
                "note": s.note,
            }
            for s in result.splits
        ],
        "imbalance_gb": round(result.imbalance_gb, 3),
        "problems": verify_split(result),
        "warnings": result.warnings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="localfit",
        description=(
            "Will this model run on my hardware, and how fast? Fit plans, "
            "quantisation comparisons and multi-GPU diagnosis, before downloading."
        ),
    )
    parser.add_argument("--version", action="version", version=f"localfit {__version__}")
    sub = parser.add_subparsers(dest="command")

    def add_common(p, with_gpu=True):
        if with_gpu:
            p.add_argument(
                "--gpu",
                action="append",
                default=[],
                help="catalogue key or common name; repeat for multi-GPU "
                "(e.g. --gpu rtx-3090-24g --gpu tesla-p40-24g)",
            )
        p.add_argument(
            "--context", type=int, default=8192, help="context length in tokens (default: 8192)"
        )
        p.add_argument(
            "--headroom",
            type=float,
            default=0.0,
            help="GB already used on the card (desktop, display, another process)",
        )
        p.add_argument("--json", action="store_true", help="machine-readable output")
        p.add_argument("--quiet", action="store_true", help="suppress notes on stderr")

    p_plan = sub.add_parser("plan", help="fit plan for one model on your hardware")
    p_plan.add_argument("model", help="model key, e.g. llama-3.1-8b or '8b'")
    add_common(p_plan)
    p_plan.add_argument("--quant", help="force a quantisation instead of choosing the best that fits")
    p_plan.add_argument(
        "--max-loss",
        type=float,
        default=None,
        help="maximum acceptable perplexity penalty (e.g. 0.1 rejects anything lossier)",
    )
    p_plan.add_argument(
        "--no-probe", action="store_true", help="do not probe hardware; assume CPU-only"
    )
    p_plan.set_defaults(func=cmd_plan)

    p_quants = sub.add_parser("quants", help="compare every quantisation on one card")
    p_quants.add_argument("model")
    add_common(p_quants)
    p_quants.set_defaults(func=cmd_quants)

    p_probe = sub.add_parser("probe", help="detect the hardware on this machine")
    p_probe.add_argument("--json", action="store_true")
    p_probe.set_defaults(func=cmd_probe)

    p_measure = sub.add_parser(
        "measure", help="read your own llama.cpp/vLLM log into a contribution row"
    )
    p_measure.add_argument("logfile")
    p_measure.add_argument(
        "--contribute", metavar="OUT.json", help="write a contribution row to this path"
    )
    p_measure.add_argument("--json", action="store_true")
    p_measure.set_defaults(func=cmd_measure)

    for name, handler, helptext in (
        ("hardware", cmd_hardware, "list known GPUs"),
        ("models", cmd_models, "list known models"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument("--json", action="store_true")
        p.set_defaults(func=handler)

    p_cal = sub.add_parser(
        "calibration", help="how accurate the speed estimate has been against real measurements"
    )
    p_cal.add_argument("--arch", help="restrict to one architecture")
    p_cal.add_argument("--json", action="store_true")
    p_cal.set_defaults(func=cmd_calibration)

    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE
    try:
        return args.func(args)
    except UsageError as exc:
        print(f"localfit: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyError as exc:
        # A KeyError escaping a resolver means a catalogue lookup was missed
        # somewhere. Report it as a usage error rather than a traceback, but say
        # what was not found so it is fixable.
        print(f"localfit: no entry found for {exc}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
