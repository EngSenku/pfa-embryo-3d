#!/usr/bin/env python3
"""Command-line entry point for the HistoWeave pipeline.

Examples
--------
Plan only (no notebooks executed, no uploads)::

    python3 run_pipeline.py --spec specs/cs17_atlas.yaml --dry-run

Full run on Kaggle GPUs (needs an access token in KAGGLE_API_TOKEN, or the
legacy KAGGLE_USERNAME / KAGGLE_KEY pair, plus HF_TOKEN)::

    python3 run_pipeline.py --spec specs/cs17_atlas.yaml --engine kaggle

Same, but skip the stages that already succeeded in an earlier attempt::

    python3 run_pipeline.py --spec specs/cs17_atlas.yaml --engine kaggle --resume \
        --run-id run-1a2b3c4d

The spec is saved to ``runs/<run_id>/spec.yaml`` together with per-stage state
(``state.json``), prepared notebooks and (for Kaggle) the kernel logs, so every
run is reproducible.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.runner import RunError, Runner  # noqa: E402
from pipeline.spec import SpecError, build_spec  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the HistoWeave pipeline from a run spec.")
    ap.add_argument("--spec", required=True, help="path to a run spec (YAML or JSON)")
    ap.add_argument("--engine", choices=("kaggle", "local"),
                    help="override run.engine from the spec")
    ap.add_argument("--dry-run", action="store_true",
                    help="prepare notebooks and print the plan without executing anything")
    ap.add_argument("--resume", action="store_true",
                    help="skip stages that already finished in this run")
    ap.add_argument("--runs-dir", default=str(ROOT / "runs"),
                    help="where run outputs live (default: ./runs)")
    ap.add_argument("--run-id", help="reuse an existing run id (used together with --resume)")
    ap.add_argument("--quiet", action="store_true", help="only print stage results")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    def log(msg: str) -> None:
        print(msg, flush=True)

    try:
        spec = build_spec(args.spec, run_id=args.run_id)
    except SpecError as exc:
        print(f"invalid spec: {exc}", file=sys.stderr)
        return 2

    if args.engine:
        spec.raw.setdefault("run", {})["engine"] = args.engine

    if not args.quiet:
        log(f"run  {spec.run_id}  ({spec.title})")
        log(f"  specimen : {spec.specimen_id}  stage {spec.stage}  engine {spec.engine}")
        log(f"  input    : {spec.input.get('format', 'atlas_am_volume')}"
            f"  registration={spec.input.get('needed', 'auto')}")
        log(f"  models   : {spec.model_source}")
        log("  stages:")
        for stage in spec.plan():
            log(f"    - {stage.describe()}")

    runner = Runner(spec=spec, runs_dir=Path(args.runs_dir).expanduser(),
                    log=log, dry_run=args.dry_run, resume=args.resume)
    try:
        state = runner.run()
    except (RunError, SpecError) as exc:
        log(f"✗ {exc}")
        return 1

    if not args.quiet:
        log(f"\nstate: {state['status']}  ->  {runner.run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
