#!/usr/bin/env python3
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from england_amr.config import load_config
from england_amr.synthetic import generate_demo_data
from england_amr.validation import run_validation
from england_amr.preprocess import run_preprocessing
from england_amr.analysis import run_analysis
from england_amr.modeling import run_training
from england_amr.scenario import run_scenario
from england_amr.visualization import run_visualizations


def main() -> None:
    cfg = load_config(SCRIPT_DIR / "demo.yaml")
    generate_demo_data(cfg, n_areas=40)
    validation = run_validation(cfg)
    if not validation["passed"].all():
        raise RuntimeError("Data validation failed.")
    run_preprocessing(cfg)
    run_analysis(cfg)
    run_training(cfg)
    run_scenario(cfg)
    run_visualizations(cfg)
    print("Pipeline completed successfully.")
    print(f"Figures: {PROJECT_ROOT / 'figures'}")
    print(f"Tables:  {PROJECT_ROOT / 'output' / 'tables'}")


if __name__ == "__main__":
    main()
