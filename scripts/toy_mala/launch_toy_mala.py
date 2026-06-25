#!/usr/bin/env python3
"""Small SLURM launcher for toy / analysis commands.

Unlike scripts/launch.py, this launcher does not require a MuJoCo --env flag in
the command. It keeps the same Kempner-friendly environment setup and supports
simple Cartesian CLI ablations for smoke tests and toy sweeps.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import re
import shlex
import subprocess
from datetime import datetime
from pathlib import Path


SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --time={time}
#SBATCH --output={log_dir}/%x_%j.out
#SBATCH --error={log_dir}/%x_%j.err
{requeue_directives}

set -euo pipefail

module load python/3.10.13-fasrc01
source ~/.venvs/general/bin/activate

cd {project_dir}

export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:/lib64:${{LD_LIBRARY_PATH:-}}"
export CPATH="$HOME/.local/glew/glew-2.1.0/include:${{CPATH:-}}"
export PYTHONPATH="{project_dir}:${{PYTHONPATH:-}}"
export ZSCRATCH="${{ZSCRATCH:-/n/netscratch/kdbrantley_lab/Lab/$USER}}"
export TOY_MALA_RUN_DIR="${{ZSCRATCH}}/runs/toy_mala/{job_name}_${{SLURM_JOB_ID}}"

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Command: {cmd}"
echo "Toy MALA run dir: $TOY_MALA_RUN_DIR"
echo "Started: $(date)"
nvidia-smi || true

{cmd}

echo "Finished: $(date)"
"""


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cmd", required=True, help="Base command to submit.")
    p.add_argument("--ablate", action="append", nargs="+", metavar=("FLAG", "VALUES"),
                   help="Cartesian ablation: --ablate beta 0 1 2.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--job-name", default="toy_mala")
    p.add_argument("--log-dir", default=None)
    p.add_argument("--partition", "-p", default=None)
    p.add_argument("--account", default="kempner_kdbrantley_lab")
    p.add_argument("--requeue", action="store_true")
    p.add_argument("--num-gpus", type=int, default=1)
    p.add_argument("--cpus", "-c", type=int, default=2)
    p.add_argument("--mem", default="16G")
    p.add_argument("--time", "-t", default="00:30:00")
    return p.parse_args()


def normalize_cmd(cmd: str) -> str:
    return " ".join(cmd.split())


def remove_flag(cmd: str, flag: str) -> str:
    tokens = shlex.split(cmd)
    out = []
    i = 0
    target = f"--{flag}"
    while i < len(tokens):
        if tokens[i] == target:
            if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(shlex.quote(t) for t in out)


def set_flag(cmd: str, flag: str, value: str) -> str:
    cmd = remove_flag(cmd, flag)
    if value == "True":
        return f"{cmd} --{flag}"
    if value == "False":
        return cmd
    return f"{cmd} --{flag} {shlex.quote(value)}"


def infer_desc(assignments):
    if not assignments:
        return "base"
    return ",".join(f"{k}={v}" for k, v in assignments)


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")[:80] or "job"


def build_jobs(base_cmd: str, ablations):
    if not ablations:
        return [(base_cmd, "base")]
    flags = [a[0] for a in ablations]
    values = [a[1:] for a in ablations]
    jobs = []
    for combo in itertools.product(*values):
        cmd = base_cmd
        assignments = []
        for flag, value in zip(flags, combo):
            cmd = set_flag(cmd, flag, value)
            assignments.append((flag, value))
        jobs.append((cmd, infer_desc(assignments)))
    return jobs


def main():
    args = parse_args()
    project_dir = Path(__file__).resolve().parents[2]
    partition = args.partition or ("kempner_requeue" if args.requeue else "kempner_h100")
    if args.log_dir:
        log_dir = Path(args.log_dir).expanduser()
    else:
        user = os.environ.get("USER", "zzhong")
        scratch = os.environ.get("ZSCRATCH", f"/n/netscratch/kdbrantley_lab/Lab/{user}")
        log_dir = Path(scratch) / "logs" / "toy_mala" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.dry_run:
        log_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = normalize_cmd(args.cmd)
    jobs = build_jobs(base_cmd, args.ablate)
    requeue_directives = "#SBATCH --requeue\n#SBATCH --signal=B:SIGTERM@120" if args.requeue else ""
    manifest_rows = []

    submitted = 0
    for idx, (cmd, desc) in enumerate(jobs):
        job_name = safe_name(f"{args.job_name}_{idx}_{desc}" if len(jobs) > 1 else args.job_name)
        cmd_for_script = cmd.replace("@RUN_DIR@", "$TOY_MALA_RUN_DIR")
        script = SBATCH_TEMPLATE.format(
            job_name=job_name,
            account=args.account,
            partition=partition,
            cpus=args.cpus,
            mem=args.mem,
            num_gpus=args.num_gpus,
            time=args.time,
            log_dir=log_dir,
            project_dir=project_dir,
            cmd=cmd_for_script,
            requeue_directives=requeue_directives,
        )
        script_path = log_dir / f"{job_name}.sbatch"
        if args.dry_run:
            print(f"[{idx + 1}/{len(jobs)}] {desc}")
            print(f"  job_name: {job_name}")
            print(f"  command: {cmd}")
            print()
            manifest_rows.append({
                "index": idx,
                "desc": desc,
                "job_name": job_name,
                "job_id": "",
                "command": cmd_for_script,
                "output_dir_pattern": f"$ZSCRATCH/runs/toy_mala/{job_name}_${{SLURM_JOB_ID}}",
                "sbatch_script": str(script_path),
                "status": "dry_run",
            })
            continue
        script_path.write_text(script)
        result = subprocess.run(["sbatch", str(script_path)], text=True, capture_output=True)
        if result.returncode == 0:
            stdout = result.stdout.strip()
            job_id = stdout.split()[-1] if stdout.startswith("Submitted batch job ") else ""
            print(f"[{idx + 1}/{len(jobs)}] {desc}: {stdout}")
            submitted += 1
            status = "submitted"
        else:
            print(f"[{idx + 1}/{len(jobs)}] {desc}: FAILED")
            print(result.stderr.strip())
            job_id = ""
            status = "failed"
        manifest_rows.append({
            "index": idx,
            "desc": desc,
            "job_name": job_name,
            "job_id": job_id,
            "command": cmd_for_script,
            "output_dir_pattern": f"$ZSCRATCH/runs/toy_mala/{job_name}_${{SLURM_JOB_ID}}",
            "sbatch_script": str(script_path),
            "status": status,
        })

    if args.dry_run:
        print(f"Dry run generated {len(jobs)} job(s). Logs would go to: {log_dir}")
    else:
        manifest_path = log_dir / "launch_manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "index", "desc", "job_name", "job_id", "command",
                    "output_dir_pattern", "sbatch_script", "status",
                ],
            )
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Submitted {submitted}/{len(jobs)} job(s). Logs: {log_dir}")
        print(f"Launch manifest: {manifest_path}")


if __name__ == "__main__":
    main()
