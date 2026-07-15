#!/usr/bin/env python3
"""
Sequential MOBO Controller for SAI SC-SPECT.

Multi-objective BO loop: propose one config, evaluate on HPC, collect metrics, repeat.
Objectives: FWHM (min), ASCI (max), sensitivity (max), MPXI (min).
Uses mobo_agent (ModelListGP + qLogNEHVI).

Usage:
  python run_mobo_loop.py                   # run with defaults
  python run_mobo_loop.py --max_iters 50    # custom iteration count
"""
import os
import sys
import time
import subprocess
import argparse
import pandas as pd

from mobo_agent import get_next_candidate

from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel

# =========================
# PATHS (overridable via env; defaults to the current 3-spebt repo layout
# on CCR: /vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt/)
# =========================
CODE_DIR = os.environ.get(
    "MOBO_CODE_DIR",
    "/vscratch/grp-rutaoyao/Omer/spebt/spebt/spebt",
)
RESULTS_DIR = os.environ.get(
    "MOBO_RESULTS_DIR",
    os.path.join(CODE_DIR, "optimization", "results"),
)
MANIFEST_FILE = os.path.join(RESULTS_DIR, "mobo_manifest.csv")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results_summary_mobo.csv")
SLURM_SCRIPT = os.path.join(CODE_DIR, "optimization", "run_sai_pipeline.sh")
LOG_DIR = os.path.join(RESULTS_DIR, "slurm_logs")

OBJ_COLUMNS = ["fwhm_mean", "asci_pct", "sensitivity_mean", "mpxi_mean"]

console = Console()


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "out"), exist_ok=True)
    os.makedirs(os.path.join(LOG_DIR, "err"), exist_ok=True)


def ensure_manifest_header():
    if not os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "w") as f:
            f.write("idx,config_name,aperture_diam_mm,n_apertures,"
                    "n_det_ring1,n_det_ring2,work_dir,job_id,status\n")


def get_next_manifest_index():
    if not os.path.exists(MANIFEST_FILE):
        return 0
    with open(MANIFEST_FILE, "r") as f:
        lines = [ln for ln in f.readlines() if ln.strip()]
    return max(0, len(lines) - 1)


def append_manifest_row(idx, config_name, diam, n_ap, n_det1, n_det2, work_dir, job_id, status="submitted"):
    with open(MANIFEST_FILE, "a") as f:
        f.write(f"{idx},{config_name},{diam:.6f},{n_ap},"
                f"{n_det1},{n_det2},{work_dir},{job_id},{status}\n")


def patch_manifest_status(idx, job_id, status):
    with open(MANIFEST_FILE, "r") as f:
        lines = f.readlines()
    last_i = max(i for i, ln in enumerate(lines) if ln.strip())
    parts = lines[last_i].rstrip("\n").split(",")
    if len(parts) >= 9:
        parts[7] = str(job_id)
        parts[8] = status
        lines[last_i] = ",".join(parts) + "\n"
        with open(MANIFEST_FILE, "w") as f:
            f.writelines(lines)


def is_job_running(job_id: str) -> bool:
    try:
        r = subprocess.run(["squeue", "--job", str(job_id)], capture_output=True, text=True)
        return str(job_id) in r.stdout
    except Exception:
        return True


def assert_initial_data():
    if not os.path.exists(RESULTS_CSV):
        console.print("[bold red]ERROR:[/bold red] Results CSV not found.")
        console.print(f"  Expected: {RESULTS_CSV}")
        console.print("  Run the initial LHS sweep first (with MPXI), then re-run.")
        sys.exit(1)
    df = pd.read_csv(RESULTS_CSV)
    df = df.dropna(subset=OBJ_COLUMNS)
    n = len(df)
    if n < 3:
        console.print(f"[bold red]ERROR:[/bold red] Need >= 3 feasible points for MOBO, got {n}.")
        sys.exit(1)
    console.print(f"[green]Loaded {n} feasible data points with all 4 objectives.[/green]")
    return n


def print_status(idx, config_name):
    """Print Pareto front summary."""
    try:
        df = pd.read_csv(RESULTS_CSV).dropna(subset=OBJ_COLUMNS)
        t = Table(title=f"MOBO Status (iter {idx})")
        t.add_column("Metric", style="cyan")
        t.add_column("Best", style="green")
        t.add_column("Worst", style="red")
        t.add_column("Mean", style="yellow")
        for col, label, direction in [
            ("fwhm_mean", "FWHM (mm)", "min"),
            ("asci_pct", "ASCI (%)", "max"),
            ("sensitivity_mean", "Sensitivity", "max"),
            ("mpxi_mean", "MPXI", "min"),
        ]:
            vals = df[col]
            best = vals.min() if direction == "min" else vals.max()
            worst = vals.max() if direction == "min" else vals.min()
            fmt = ".4f" if col != "sensitivity_mean" else ".4e"
            t.add_row(label, f"{best:{fmt}}", f"{worst:{fmt}}", f"{vals.mean():{fmt}}")
        t.add_row("Total configs", str(len(df)), "", "")
        console.print(t)
    except Exception as e:
        console.print(f"[yellow]Could not print status: {e}[/yellow]")


def main():
    parser = argparse.ArgumentParser(description="Sequential MOBO loop for SAI SC-SPECT")
    parser.add_argument("--max_iters", type=int, default=50,
                        help="Total MOBO iterations (default: 50)")
    args = parser.parse_args()

    TOTAL_ITERATIONS = args.max_iters

    console.print(Panel.fit(
        "[bold green]SAI SC-SPECT MOBO Controller[/bold green]\n"
        "Design: (aperture_diam, n_apertures, n_det_ring1, n_det_ring2)\n"
        "4 objectives: FWHM (min), ASCI (max), Sensitivity (max), MPXI (min)\n"
        "ModelListGP + qLogNEHVI | Sequential q=1",
        subtitle=f"Max iterations: {TOTAL_ITERATIONS}"
    ))

    ensure_dirs()
    ensure_manifest_header()
    assert_initial_data()

    start_idx = get_next_manifest_index()
    if start_idx >= TOTAL_ITERATIONS:
        console.print(f"[green]Done.[/green] Already have {start_idx} entries.")
        return

    n_to_run = TOTAL_ITERATIONS - start_idx

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[green]MOBO Loop", total=n_to_run)

        for _ in range(n_to_run):
            idx = get_next_manifest_index()
            console.print(f"\n[bold yellow]=== MOBO Iteration {idx} ===[/bold yellow]")

            # 1. Ask MOBO agent for next candidate
            console.log("Calling mobo_agent.get_next_candidate()...")
            try:
                diam, n_ap, n_det1, n_det2 = get_next_candidate(RESULTS_CSV)
            except Exception as e:
                console.print(f"[bold red]MOBO Agent Failed:[/bold red] {e}")
                break

            console.log(f"[cyan]Candidate:[/cyan] d={diam:.4f}mm  n={n_ap}  "
                        f"nd1={n_det1}  nd2={n_det2}")

            # 2. Create work directory
            config_name = f"mobo_{idx:04d}_ap{diam:.4f}_nap{n_ap}_nd1_{n_det1}_nd2_{n_det2}"
            work_dir = os.path.join(RESULTS_DIR, config_name)
            os.makedirs(work_dir, exist_ok=True)

            # 3. Append to manifest
            append_manifest_row(idx, config_name, diam, n_ap, n_det1, n_det2, work_dir, "", "pending")

            # 4. Submit SLURM job
            env_vars = (
                f"ALL,"
                f"WORK_DIR={work_dir},"
                f"APERTURE_DIAM={diam},"
                f"N_APERTURES={n_ap},"
                f"N_DET_RING1={n_det1},"
                f"N_DET_RING2={n_det2},"
                f"A_MM=0.2,B_MM=0.2,"
                f"CODE_DIR={CODE_DIR},"
                f"RESULTS_CSV={RESULTS_CSV},"
                f"CONFIG_NAME={config_name}"
            )

            sbatch_cmd = [
                "sbatch",
                "--parsable",
                f"--output={LOG_DIR}/out/{config_name}_%j.out",
                f"--error={LOG_DIR}/err/{config_name}_%j.err",
                "--export", env_vars,
                SLURM_SCRIPT,
            ]

            try:
                job_id = subprocess.check_output(sbatch_cmd, text=True).strip().split(";")[0]
                console.log(f"[bold green]Submitted job:[/bold green] {job_id}")
            except Exception as e:
                console.print(f"[bold red]SLURM Submission Failed:[/bold red] {e}")
                patch_manifest_status(idx, "", "failed")
                break

            patch_manifest_status(idx, job_id, "running")

            # 5. Wait for job to finish
            console.log(f"Waiting for job {job_id}...")
            wait_start = time.time()
            last_print = 0

            while is_job_running(job_id):
                time.sleep(60)
                elapsed_min = int((time.time() - wait_start) / 60)
                if elapsed_min >= last_print + 10:
                    console.log(f"[yellow]Job {job_id} still running... ({elapsed_min} min)[/yellow]")
                    last_print = elapsed_min

            elapsed_total = int((time.time() - wait_start) / 60)
            console.log(f"Job {job_id} finished in {elapsed_total} min.")
            patch_manifest_status(idx, job_id, "completed")

            # 6. Print status
            print_status(idx, config_name)
            progress.advance(task)

    console.print("\n[bold green]MOBO Loop finished.[/bold green]")
    print_status(get_next_manifest_index() - 1, "final")


if __name__ == "__main__":
    main()
