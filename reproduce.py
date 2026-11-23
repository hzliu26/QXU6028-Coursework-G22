"""Regenerate the regular analysis and the dated, post-hoc blind comparison."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib
import numpy as np
import pandas as pd

from analysis import analyse
from figures import make_figures
from supplements_l12 import write_l2_gate
from validate import validate_results


ROOT = Path(__file__).resolve().parent
FIGURES = ("Figure_1_Carrier_counting", "Figure_2_DoS_inference",
           "Figure_3_Spatial_reconstruction", "Figure_4_Validation", "Figure_S1_Repeat_checks",
           "L2_gate_cone_wings_and_neutrality")


def revision():
    if (ROOT / ".git").exists():
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    return (ROOT / "SOURCE_REVISION.txt").read_text(encoding="utf-8").strip()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(data, output, commit):
    result, diagnostics, arrays = analyse(data, commit)
    level1, level2, level3 = arrays
    dataset = result["dataset_set"]
    print(f"Analysed {dataset} data ({data.name})", flush=True)
    diagnostics["environment"] = dict(python=platform.python_version(), numpy=np.__version__,
                                      pandas=pd.__version__, matplotlib=matplotlib.__version__)
    output.mkdir(exist_ok=True, parents=True)
    filename = "results.json" if dataset == "regular" else "blind_results.json"
    save_json(output / filename, result)
    save_json(output / "diagnostics.json", diagnostics)
    (output / "pipeline_commit.txt").write_text(commit + "\n", encoding="utf-8")
    tables = output / "tables"
    tables.mkdir(exist_ok=True)
    level1["sweep"].to_csv(tables / "L1_sweep.csv", index=False)
    level2["trajectory"].to_csv(tables / "L2_trajectory.csv", index=False)
    pd.DataFrame({"candidate_gap_kT": level2["gaps"], "pair_rms": level2["residual"]}).to_csv(
        tables / "L2_model_residuals.csv", index=False)
    level3["sensitivity"].to_csv(tables / "L3_regularisation.csv", index=False)
    pd.DataFrame(result["L3"]["features"]).to_csv(tables / "L3_features.csv", index=False)
    np.savetxt(tables / "L3_map.csv", np.array(result["L3"]["map"]).reshape(10, 10), delimiter=",")
    np.savetxt(tables / "L2_bootstrap_gaps_kT.csv", level2["boot_gap"], delimiter=",")
    np.savetxt(tables / "L3_noise_maxima.csv", level3["noise_maxima"], delimiter=",")
    make_figures(output, result, diagnostics, arrays)
    write_l2_gate(output, diagnostics, arrays)
    checks = validate_results(data, result, diagnostics, output, FIGURES)
    save_json(output / "checks.json", checks)
    print(f"{dataset} | L1 C={result['L1']['correction_factor']:.6f} | "
          f"L2 {result['L2']['model']}, Eg/kT={result['L2']['Eg_kT']:g} | "
          f"L3 peak={100*diagnostics['L3']['maximum_fraction']:.5f}%, {result['L3']['confidence']} | "
          f"checks passed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT, help="Root directory for regenerated outputs")
    parser.add_argument("--regular-only", action="store_true", help="Skip the separate post-hoc blind reanalysis")
    args = parser.parse_args()
    if not __debug__:
        sys.exit("Run without -O to enable the physics and integrity assertions.")
    commit = revision()
    print(f"Reproduction started {datetime.now(timezone.utc).isoformat()} | commit {commit}", flush=True)
    out = args.output.resolve()
    run(ROOT / "data" / "regular_G22", out, commit)
    subprocess.run([sys.executable, str(ROOT / "checkpoint_l3.py"), str(out),
                    str(out / "checkpoints/retrospective/G22_L3_Checkpoint.zip"),
                    "--data", str(ROOT / "data/regular_G22")], check=True)
    if not args.regular_only:
        run(ROOT / "data" / "blind_G22", out / "posthoc_blind_reanalysis", commit)
    print("Completed " + datetime.now(timezone.utc).isoformat(), flush=True)


if __name__ == "__main__":
    main()
