#!/usr/bin/env python3
"""Resubmit failed toy MALA SLURM jobs from existing launch manifests.

The launcher writes one ``launch_manifest.csv`` per submitted job family.  This
helper reads those manifests, checks each corresponding ``*.out`` file for the
``wrote results to`` success marker, and resubmits only rows that do not have a
successful output.  It reuses the original sbatch script so the command and run
directory template stay identical, while moving logs into a fresh directory.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable


SUCCESS_MARKER = "wrote results to"


def out_path_for_manifest_row(log_dir: Path, row: dict[str, str]) -> Path:
    return log_dir / f"{row['job_name']}_{row['job_id']}.out"


def row_has_success(log_dir: Path, row: dict[str, str]) -> bool:
    out_path = out_path_for_manifest_row(log_dir, row)
    if not out_path.exists():
        return False
    return SUCCESS_MARKER in out_path.read_text(errors="replace")


def patch_sbatch_text(text: str, new_log_dir: Path, exclude: str | None) -> str:
    lines = text.splitlines()
    patched: list[str] = []
    saw_exclude = False
    inserted_exclude = False

    for line in lines:
        if line.startswith("#SBATCH --output="):
            patched.append(f"#SBATCH --output={new_log_dir}/%x_%j.out")
            continue
        if line.startswith("#SBATCH --error="):
            patched.append(f"#SBATCH --error={new_log_dir}/%x_%j.err")
            continue
        if line.startswith("#SBATCH --exclude="):
            saw_exclude = True
            if exclude:
                patched.append(f"#SBATCH --exclude={exclude}")
            continue

        patched.append(line)

        if exclude and not saw_exclude and not inserted_exclude and line.startswith("#SBATCH --nodes="):
            patched.append(f"#SBATCH --exclude={exclude}")
            inserted_exclude = True

    if exclude and not saw_exclude and not inserted_exclude:
        for idx, line in enumerate(patched):
            if line.startswith("#SBATCH "):
                continue
            patched.insert(idx, f"#SBATCH --exclude={exclude}")
            break

    return "\n".join(patched) + "\n"


def read_manifest(log_dir: Path) -> list[dict[str, str]]:
    manifest = log_dir / "launch_manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    with manifest.open(newline="") as f:
        return list(csv.DictReader(f))


def iter_failed_rows(source_log_dirs: Iterable[Path]) -> Iterable[tuple[Path, dict[str, str]]]:
    for log_dir in source_log_dirs:
        for row in read_manifest(log_dir):
            if row.get("status") == "dry_run":
                continue
            if not row_has_success(log_dir, row):
                yield log_dir, row


def default_log_dir() -> Path:
    user = os.environ.get("USER", "zzhong")
    scratch = os.environ.get("ZSCRATCH", f"/n/netscratch/kdbrantley_lab/Lab/{user}")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(scratch) / "logs" / "toy_mala" / f"{stamp}_rerun_failed"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source-log-dir",
        action="append",
        required=True,
        help="Original log directory containing launch_manifest.csv. Repeat for multiple families.",
    )
    p.add_argument("--new-log-dir", default=None, help="Fresh log directory for rerun sbatch/out/err files.")
    p.add_argument("--exclude", default=None, help="Node list to pass to #SBATCH --exclude, e.g. holygpu8a19102.")
    p.add_argument("--dry-run", action="store_true", help="Print failed rows without submitting.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    source_log_dirs = [Path(p).expanduser() for p in args.source_log_dir]
    new_log_dir = Path(args.new_log_dir).expanduser() if args.new_log_dir else default_log_dir()
    failed = list(iter_failed_rows(source_log_dirs))

    print(f"found failed/no-result jobs: {len(failed)}")
    print(f"new log dir: {new_log_dir}")
    if args.exclude:
        print(f"excluding node(s): {args.exclude}")

    if args.dry_run:
        for source_log_dir, row in failed:
            print(f"{source_log_dir.name}: {row['job_name']} old_job_id={row['job_id']}")
        return

    new_log_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str]] = []
    submitted = 0

    for source_log_dir, row in failed:
        old_script = Path(row["sbatch_script"])
        if not old_script.exists():
            raise FileNotFoundError(f"Missing old sbatch script: {old_script}")

        old_text = old_script.read_text()
        new_text = patch_sbatch_text(old_text, new_log_dir, args.exclude)
        source_tag = source_log_dir.name
        new_script = new_log_dir / f"{source_tag}_{old_script.name}"
        new_script.write_text(new_text)

        result = subprocess.run(["sbatch", str(new_script)], text=True, capture_output=True)
        if result.returncode == 0:
            stdout = result.stdout.strip()
            new_job_id = stdout.split()[-1] if stdout.startswith("Submitted batch job ") else ""
            status = "submitted"
            submitted += 1
            print(f"{row['job_name']} old={row['job_id']}: {stdout}")
        else:
            new_job_id = ""
            status = "failed"
            print(f"{row['job_name']} old={row['job_id']}: FAILED")
            print(result.stderr.strip())

        manifest_rows.append(
            {
                "source_log_dir": str(source_log_dir),
                "old_job_name": row["job_name"],
                "old_job_id": row["job_id"],
                "new_job_id": new_job_id,
                "new_sbatch_script": str(new_script),
                "status": status,
            }
        )

    manifest_path = new_log_dir / "rerun_manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_log_dir",
                "old_job_name",
                "old_job_id",
                "new_job_id",
                "new_sbatch_script",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Submitted {submitted}/{len(failed)} rerun job(s).")
    print(f"Rerun manifest: {manifest_path}")


if __name__ == "__main__":
    main()
