"""Master orchestrator: runs every pipeline stage in order, each as an isolated module."""
import argparse
import os
import subprocess
import sys

from mri.config import ROOT

# (module, description) in execution order.
PIPELINE_STEPS = [
    ("mri.data.build_db", "Hydrate SQLite database from data/raw CSVs"),
    ("mri.detection.rules", "V1 rule engine (weak labels)"),
    ("mri.features.engineering", "Feature engineering"),
    ("mri.models.train", "Train supervised + anomaly models"),
    ("mri.models.score", "Score entities"),
    ("mri.models.explain", "Generate explanations"),
    ("mri.detection.graph", "V3 graph / entity link analysis"),
    ("mri.detection.temporal", "V4 temporal burst detection"),
    ("mri.fusion.fuse", "V4.1 risk fusion"),
    ("mri.evaluation", "Measure against ground truth (precision@k, lift, per-pattern recall)"),
    ("mri.governance.ops", "V6 governance, drift and registry"),
]
WORKBENCH = ROOT / "src" / "mri" / "app" / "workbench.py"


def log(msg: str) -> None:
    print(f"[pipeline] {msg}")


def _env() -> dict:
    # Make `mri` importable in child processes even without `pip install -e .`.
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_step(module: str, description: str) -> None:
    log(f"Running: {module} - {description}")
    # Fixed argv list, no shell: module names come from PIPELINE_STEPS only.
    result = subprocess.run(
        [sys.executable, "-m", module], cwd=ROOT, env=_env(), text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {module}")
    log(f"Completed: {module}\n")


def launch_workbench() -> None:
    log("Launching Streamlit workbench")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(WORKBENCH)],
        cwd=ROOT, env=_env(), text=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MRI pipeline.")
    parser.add_argument("--no-ui", action="store_true", help="skip the workbench prompt (CI / scripted runs)")
    args = parser.parse_args()

    log("Starting pipeline...")
    log("-" * 60)
    for module, description in PIPELINE_STEPS:
        run_step(module, description)
    log("=" * 60)
    log("PIPELINE COMPLETED SUCCESSFULLY")

    if args.no_ui:
        return
    if input("Launch the analyst workbench now? (y/n): ").strip().lower() == "y":
        launch_workbench()
    else:
        log("To launch manually: streamlit run src/mri/app/workbench.py")


if __name__ == "__main__":
    main()
