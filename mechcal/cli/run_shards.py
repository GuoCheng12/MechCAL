from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

from mechcal.public_ids import anonymized_case_id
from mechcal.runtime.ablations import ABLATION_MODES

METRICS = [
    "top1_primary",
    "top1_any",
    "recall_at_3",
    "ndcg_at_3",
    "es_ndcg_at_3",
    "support_at_top1",
    "claim_safety_at_3",
    "unsupported_strong_claim_rate_at_3",
    "credibility_gap_at_3",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or merge sharded MAS mechanism benchmark evaluations."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run cases split over shards.")
    _add_common_args(run_parser)
    run_parser.add_argument("--shards", type=int, default=8)
    run_parser.add_argument("--max-rounds", type=int, default=30)
    run_parser.add_argument(
        "--ablation-mode",
        choices=ABLATION_MODES,
        default="full_mechcal",
    )
    run_parser.add_argument("--case-id", action="append", default=[])
    run_parser.add_argument("--exclude-case", action="append", default=[])
    run_parser.add_argument("--limit", type=int, default=None)
    run_parser.add_argument("--start", action="store_true")
    run_parser.add_argument("--refresh", action="store_true")
    run_parser.add_argument("--disable-incremental-portfolio", action="store_true")
    run_parser.add_argument("--enable-incremental-portfolio", action="store_true")
    run_parser.add_argument(
        "--public-case-id-map",
        type=Path,
        default=None,
        help=(
            "Optional existing JSON map from internal case_id to anonymized "
            "public case id. Use this when rerunning a subset from a larger "
            "fixed case list."
        ),
    )

    merge_parser = subparsers.add_parser("merge", help="Merge shard summaries.")
    _add_common_args(merge_parser)

    args = parser.parse_args()
    if args.command == "run":
        run_shards(args)
    elif args.command == "merge":
        merge_shards(args.output_dir)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/sharded"),
    )


def run_shards(args: argparse.Namespace) -> None:
    if args.disable_incremental_portfolio and args.enable_incremental_portfolio:
        raise ValueError(
            "--enable-incremental-portfolio and "
            "--disable-incremental-portfolio are mutually exclusive."
        )
    case_ids = _selected_case_ids(
        args.case_dir,
        explicit_case_ids=args.case_id,
        excluded_case_ids=args.exclude_case,
        limit=args.limit,
    )
    shards = _split_case_ids(case_ids, args.shards)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    public_case_id_map = (
        _load_public_case_id_map(args.public_case_id_map)
        if args.public_case_id_map is not None
        else _public_case_id_map(case_ids)
    )
    _validate_public_case_id_map(case_ids, public_case_id_map, args.public_case_id_map)
    public_case_id_map_path = output_dir / "public_case_id_map.json"
    _write_json(
        public_case_id_map_path,
        {
            "case_count": len(case_ids),
            "public_case_id_map": public_case_id_map,
        },
    )
    commands = []
    for index, shard_case_ids in enumerate(shards):
        shard_dir = output_dir / f"shard_{index:02d}"
        commands.append(
            _eval_command(
                case_dir=args.case_dir,
                output_dir=shard_dir,
                case_ids=shard_case_ids,
                max_rounds=args.max_rounds,
                ablation_mode=args.ablation_mode,
                public_case_id_map_path=public_case_id_map_path,
                refresh=args.refresh,
                disable_incremental_portfolio=args.disable_incremental_portfolio,
                enable_incremental_portfolio=args.enable_incremental_portfolio,
            )
        )
    _write_manifest(
        output_dir,
        case_ids=case_ids,
        shards=shards,
        commands=commands,
        public_case_id_map=public_case_id_map,
        public_case_id_map_path=public_case_id_map_path,
        max_rounds=args.max_rounds,
        ablation_mode=args.ablation_mode,
    )
    if not args.start:
        print(f"Prepared {len(shards)} shard commands for {len(case_ids)} cases.")
        print(f"Manifest: {output_dir / 'shard_manifest.json'}")
        print("Dry run only. Re-run with --start to launch.")
        for command in commands:
            print(_shell_join(command))
        return

    started_at = time.monotonic()
    processes: list[tuple[int, subprocess.Popen[str], Path]] = []
    for index, command in enumerate(commands):
        shard_dir = output_dir / f"shard_{index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        log_path = shard_dir / "run.log"
        log_file = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(  # noqa: S603
            command,
            cwd=Path.cwd(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env={
                **os.environ,
                "MECHCAL_UPDATE_REDUCER_MODE": os.environ.get(
                    "MECHCAL_UPDATE_REDUCER_MODE",
                    "reducer",
                ),
            },
        )
        processes.append((index, process, log_path))
        print(f"started shard_{index:02d} pid={process.pid} log={log_path}")

    failures = []
    for index, process, log_path in processes:
        code = process.wait()
        if code != 0:
            failures.append((index, code, log_path))
        print(f"finished shard_{index:02d} exit={code} log={log_path}")
    elapsed = time.monotonic() - started_at
    if failures:
        for index, code, log_path in failures:
            print(f"FAILED shard_{index:02d} exit={code} log={log_path}", file=sys.stderr)
        raise SystemExit(1)
    merged = merge_shards(output_dir)
    merged["wall_seconds"] = elapsed
    _write_json(output_dir / "summary.json", merged)
    print(f"merged summary: {output_dir / 'summary.json'}")
    print(f"wall_seconds={elapsed:.1f}")


def merge_shards(output_dir: Path) -> dict[str, object]:
    output_dir = output_dir.expanduser().resolve()
    rows = []
    shard_summaries = []
    for path in sorted(output_dir.glob("shard_*/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        shard_summaries.append(
            {
                "path": str(path),
                "case_count": summary.get("case_count", 0),
                "averages": summary.get("averages", {}),
            }
        )
        rows.extend(summary.get("rows") or [])
    rows.sort(key=lambda item: str(item.get("case_id", "")))
    clean_rows = [row for row in rows if isinstance(row, dict) and _is_clean_row(row)]
    merged = {
        "case_count": len(rows),
        "scored_case_count": sum(
            1 for row in rows if row.get("evaluation_status", "scored") == "scored"
        ),
        "unscored_case_count": sum(
            1 for row in rows if row.get("evaluation_status", "scored") != "scored"
        ),
        "clean_case_count": len(clean_rows),
        "dirty_case_count": len(rows) - len(clean_rows),
        "averages": {
            metric: _mean(
                row.get("scores", {}).get(metric)
                for row in rows
                if isinstance(row, dict)
            )
            for metric in METRICS
        },
        "clean_averages": {
            metric: _mean(row.get("scores", {}).get(metric) for row in clean_rows)
            for metric in METRICS
        },
        "average_verifier_failure_count": _mean(
            row.get("verifier_failure_count")
            for row in rows
            if isinstance(row, dict)
        ),
        "average_planner_failure_count": _mean(
            row.get("planner_failure_count")
            for row in rows
            if isinstance(row, dict)
        ),
        "average_mechanism_critic_failure_count": _mean(
            row.get("mechanism_critic_failure_count")
            for row in rows
            if isinstance(row, dict)
        ),
        "shard_summaries": shard_summaries,
        "rows": rows,
    }
    _write_json(output_dir / "summary.json", merged)
    print(f"merged {len(rows)} rows from {len(shard_summaries)} shard summaries")
    return merged


def _selected_case_ids(
    case_dir: Path,
    *,
    explicit_case_ids: list[str],
    excluded_case_ids: list[str] | None = None,
    limit: int | None,
) -> list[str]:
    if explicit_case_ids:
        return list(dict.fromkeys(explicit_case_ids))
    case_ids = [path.stem for path in sorted(case_dir.expanduser().resolve().glob("*.json"))]
    excluded = set(excluded_case_ids or [])
    case_ids = [case_id for case_id in case_ids if case_id not in excluded]
    if limit is not None:
        case_ids = case_ids[:limit]
    return case_ids


def _split_case_ids(case_ids: list[str], shard_count: int) -> list[list[str]]:
    if shard_count <= 0:
        raise ValueError("--shards must be positive.")
    if not case_ids:
        return []
    shard_count = min(shard_count, len(case_ids))
    shards = [[] for _ in range(shard_count)]
    for index, case_id in enumerate(case_ids):
        shards[index % shard_count].append(case_id)
    return [shard for shard in shards if shard]


def _public_case_id_map(case_ids: list[str]) -> dict[str, str]:
    return {
        case_id: anonymized_case_id(index)
        for index, case_id in enumerate(case_ids, start=1)
    }


def _load_public_case_id_map(path: Path) -> dict[str, str]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("public_case_id_map"), dict):
        payload = payload["public_case_id_map"]
    if not isinstance(payload, dict):
        raise ValueError(f"Public case id map must be a JSON object: {path}")
    return {str(case_id): str(public_case_id) for case_id, public_case_id in payload.items()}


def _validate_public_case_id_map(
    case_ids: list[str],
    public_case_id_map: dict[str, str],
    map_path: Path | None,
) -> None:
    missing = [case_id for case_id in case_ids if case_id not in public_case_id_map]
    if missing:
        source = str(map_path) if map_path is not None else "generated map"
        raise ValueError(
            f"Public case id map {source} is missing case ids: " + ", ".join(missing)
        )
    empty_values = [case_id for case_id in case_ids if not public_case_id_map[case_id]]
    if empty_values:
        source = str(map_path) if map_path is not None else "generated map"
        raise ValueError(
            f"Public case id map {source} has empty ids for: " + ", ".join(empty_values)
        )


def _eval_command(
    *,
    case_dir: Path,
    output_dir: Path,
    case_ids: list[str],
    max_rounds: int,
    ablation_mode: str,
    public_case_id_map_path: Path,
    refresh: bool,
    disable_incremental_portfolio: bool,
    enable_incremental_portfolio: bool,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "mechcal.cli.evaluate",
        "--case-dir",
        str(case_dir),
        "--output-dir",
        str(output_dir),
        "--max-rounds",
        str(max_rounds),
        "--ablation-mode",
        ablation_mode,
        "--public-case-id-map",
        str(public_case_id_map_path),
    ]
    for case_id in case_ids:
        command.extend(["--case-id", case_id])
    if refresh:
        command.append("--refresh")
    if disable_incremental_portfolio:
        command.append("--disable-incremental-portfolio")
    if enable_incremental_portfolio:
        command.append("--enable-incremental-portfolio")
    return command


def _write_manifest(
    output_dir: Path,
    *,
    case_ids: list[str],
    shards: list[list[str]],
    commands: list[list[str]],
    public_case_id_map: dict[str, str],
    public_case_id_map_path: Path,
    max_rounds: int,
    ablation_mode: str,
) -> None:
    _write_json(
        output_dir / "shard_manifest.json",
        {
            "case_count": len(case_ids),
            "shard_count": len(shards),
            "max_rounds": max_rounds,
            "ablation_mode": ablation_mode,
            "public_case_id_map_path": str(public_case_id_map_path),
            "public_case_id_map": public_case_id_map,
            "shards": [
                {
                    "shard_id": f"shard_{index:02d}",
                    "case_count": len(shard_case_ids),
                    "case_ids": shard_case_ids,
                    "command": command,
                }
                for index, (shard_case_ids, command) in enumerate(
                    zip(shards, commands, strict=True)
                )
            ],
        },
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _is_clean_row(row: dict[str, object]) -> bool:
    return (
        row.get("status") == "finalized"
        and row.get("evaluation_status", "scored") == "scored"
        and row.get("planner_failure_count") == 0
        and row.get("verifier_failure_count") == 0
        and row.get("mechanism_critic_failure_count", 0) == 0
    )


def _mean(values) -> float | None:  # noqa: ANN001
    numeric = [float(value) for value in values if isinstance(value, int | float)]
    return statistics.fmean(numeric) if numeric else None


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(item) for item in command)


if __name__ == "__main__":
    main()
