import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def log(msg: str):
    print(f"[pipeline-run] {msg}")


def run_step(script_name: str):
    script_path = BASE_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Missing script: {script_path}")

    log(f"Running: {script_name}")
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script_name}")

    log(f"Completed: {script_name}\n")


def launch_streamlit(app_name: str):
    app_path = BASE_DIR / app_name

    if not app_path.exists():
        raise FileNotFoundError(f"Missing Streamlit app: {app_path}")

    log(f"Launching Streamlit UI: {app_name}")

    # This starts Streamlit and keeps it running
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        cwd=BASE_DIR,
        text=True
    )


def main():
    pipeline_steps = [
        "risk_scoring.py",
        "feature_eng.py",
        "train_model.py",
        "score_entities.py",
        "explain_entities.py",
        "intelligence_graph.py",
        "temporal_intelligence.py",
        "risk_fusionv4.py",
        "operation_v6.py",
    ]

    log("Starting incremental pipeline...")
    log("-" * 60)

    for step in pipeline_steps:
        run_step(step)

    log("=" * 60)
    log("INCREMENTAL PIPELINE COMPLETED SUCCESSFULLY")

    launch_ui = input("Do you want to launch the V5 workbench now? (y/n): ").strip().lower()

    if launch_ui == "y":
        launch_streamlit("v5_investigation_workbench.py")
    else:
        log("UI launch skipped.")
        log("To launch manually, run:")
        log("python -m streamlit run v5_investigation_workbench.py")


if __name__ == "__main__":
    main()