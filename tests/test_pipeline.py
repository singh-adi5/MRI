"""End-to-end smoke test: the full pipeline must run on the committed synthetic data."""
import subprocess
import sys

import pandas as pd

from mri.config import OUTPUT_DIR, ROOT


def test_pipeline_runs_and_scores_are_valid():
    result = subprocess.run(
        [sys.executable, "-m", "mri.pipeline", "--no-ui"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]

    fused = pd.read_csv(OUTPUT_DIR / "final_fused_risk_v4.csv")
    assert not fused.empty
    assert fused["fused_risk_score"].between(0, 100).all()
    assert set(fused["fused_risk_band"].unique()) <= {"Critical", "High", "Medium", "Low"}
