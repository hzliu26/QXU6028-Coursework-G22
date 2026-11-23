"""Build a retrospective G22 regular-data checkpoint from a completed analysis.

Usage: python checkpoint_l3.py RUN_DIRECTORY OUTPUT.zip [--data REGULAR_INPUT_DIRECTORY]
The run directory must contain the regular results.json and its diagnostics and tables.
"""
import argparse
import json
import re
import tempfile
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def figures(directory, raw, results, diagnostics, checks):
    field = np.asarray(results["map"]).reshape(10, 10) * 100
    features = results["features"]
    limit = float(np.max(np.abs(field)))
    fig, ax = plt.subplots(figsize=(8, 6.5), layout="constrained")
    display = ax.imshow(field, origin="lower", extent=(-3, 3, -3, 3), cmap="RdBu_r",
                        vmin=-limit, vmax=limit, interpolation="nearest")
    labels = []
    for index, feature in enumerate(features, 1):
        ax.plot(feature["x"], feature["y"], "o", ms=14, mec="black", mfc="white")
        ax.text(feature["x"], feature["y"], str(index), va="center", ha="center", fontsize=8)
        labels.append(f'{index}   ({feature["x"]:+.1f}, {feature["y"]:+.1f})   {100*feature["amplitude"]:+.3f}%')
    ax.set(xlabel="x (probe pitches)", ylabel="y (probe pitches)", title="G22 regular wafer")
    ax.text(0, -.14, "Feature   (x, y)   deviation\n" + "\n".join(labels),
            transform=ax.transAxes, va="top", family="monospace", fontsize=9)
    fig.colorbar(display, ax=ax, shrink=.85, label="Conductivity deviation from mean (%)")
    fig.savefig(directory / "L3_map.pdf")
    plt.close(fig)

    A = np.load(raw / "L3_A.npy", allow_pickle=False)
    repeats = np.loadtxt(raw / "L3_d.csv", delimiter=",")
    grid = np.arange(100).reshape(10, 10)
    eye = np.eye(100)
    gradient = np.concatenate((eye[grid[:, 1:].ravel()] - eye[grid[:, :-1].ravel()],
                               eye[grid[1:, :].ravel()] - eye[grid[:-1, :].ravel()]))
    # A constrained saddle-point solve independently checks the submitted projected solve.
    ones = np.ones((100, 1))
    normal = A.T @ A + diagnostics["eta_selected"] * (gradient.T @ gradient) + 1e-9 * eye
    kkt = np.block([[normal, ones], [ones.T, np.zeros((1, 1))]])
    inverse = np.linalg.solve(kkt, np.vstack((A.T, np.zeros((1, A.shape[0])))))[:-1]
    np.testing.assert_allclose(inverse @ repeats.mean(axis=0), field.ravel()/100, atol=1e-10)
    sigma = repeats.std(axis=0, ddof=1)
    samples = np.random.default_rng(diagnostics["seed"]).normal(size=(diagnostics["noise_runs"], A.shape[0]))
    maxima = np.max(np.abs((samples * sigma) @ inverse.T), axis=1)
    np.testing.assert_allclose(np.percentile(maxima, 95), results["detection_limit_fraction"], rtol=1e-8)

    fig, (tradeoff, profile) = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    tradeoff.loglog(checks["roughness_norm"], checks["residual_norm"], "o-", color="#276388")
    for check in checks:
        offset = (-12, -13) if check["eta_factor"] == .01 else (1, 6)
        tradeoff.annotate(f'{check["eta_factor"]:g}', (check["roughness_norm"], check["residual_norm"]),
                          xytext=offset, textcoords="offset points", fontsize=9)
    i = np.flatnonzero(np.isclose(checks["eta_factor"], diagnostics["eta_factor_selected"]))[0]
    tradeoff.scatter(checks["roughness_norm"][i], checks["residual_norm"][i], s=75,
                     facecolors="none", edgecolors="#a3301c", label="Selected smoothing", zorder=4)
    tradeoff.axhline(diagnostics["mean_scan_noise_norm_estimate"], ls=":", color="#444",
                     label="Mean-scan noise norm")
    tradeoff.margins(x=.18, y=.17)
    tradeoff.set(xlabel="Spatial roughness norm", ylabel="Data residual norm",
                 title="A. Smoothing factors relative to reference")
    tradeoff.legend(fontsize=8)
    grid_xy = np.column_stack(np.unravel_index(np.arange(100), (10, 10))) * .6
    for pos, recorded_width in zip(diagnostics["resolution_positions_row_column"],
                                   diagnostics["resolution_values_probe_pitch"]):
        row, col = pos
        point = eye[row*10+col] - .01
        response = inverse @ (A @ point)
        weights = np.maximum(response, 0)
        radii_squared = ((grid_xy - grid_xy[row*10+col])**2).sum(axis=1)
        width = 2.355 * np.sqrt(np.dot(weights, radii_squared)/(2*weights.sum()))
        np.testing.assert_allclose(width, recorded_width, rtol=1e-8)
        cut = response.reshape(10, 10)[row]
        profile.plot((np.arange(10)-col)*.6, cut/cut[col], ".-", label=f"({row}, {col}): {width:.3f} pitches")
    profile.axhline(.5, color="#666", ls=":", lw=.9)
    profile.set(xlabel="Offset along injected row (probe pitches)", ylabel="Response / value at injection",
                title="B. Response to a single pixel")
    profile.legend(fontsize=8, title="(row, column) and 2-D width", title_fontsize=8)
    fig.savefig(directory / "L3_method.pdf")
    plt.close(fig)


def note(result, diag):
    l3 = result["L3"]
    amp = diag["maximum_fraction"]
    delta = l3["detection_limit_fraction"]
    def pct(value):
        return f"{100*value:.4f}%"
    rows = [f'{i}. ({f["x"]:+.1f}, {f["y"]:+.1f})   {100*f["amplitude"]:+.4f}%'
            for i, f in enumerate(l3["features"], 1)]
    return f"""G22 | L3 regular-data checkpoint
Retrospective reconstruction from the regular analysis. The commit in commit_hash.txt reproduces these results. This package makes no claim about the original checkpoint upload date or the pre-blind freeze.

Question
Does the maximum relative Hall-conductivity deviation exceed the 3.0% wafer specification?

Inverse model and smoothing
The 128 × 100 matrix has numerical rank {diag["numerical_rank"]} and maximum absolute row sum {diag["max_absolute_row_sum"]:.2e}. A uniform offset is invisible. We imposed a zero-mean map and penalised horizontal and vertical first differences. Three scans estimate the noise. At the guide's reference smoothing, the residual norm {diag["eta0_residual_norm"]:.3g} exceeds three times the mean-scan noise norm {diag["mean_scan_noise_norm_estimate"]:.3g}. Of five tested factors, {diag["eta_factor_selected"]:g} gives the closest residual to that norm on a log scale. At the selected eta = {diag["eta_selected"]:.4g}, the residual is {diag["residual_norm"]:.3g}. Larger smoothing suppresses peak amplitudes.

Spatial findings
Map labels give reconstructed extrema at (x,y) in probe pitches. A connected region above the larger of the detection limit and 35% of its sign-specific peak contributes one extremum.
{chr(10).join(rows)}

Resolution and decision
Point injections at the centre and off centre gave equivalent two-dimensional positive-lobe widths of {diag["resolution_values_probe_pitch"][0]:.3f} and {diag["resolution_values_probe_pitch"][1]:.3f} probe pitches. The larger width sets the reported resolution. For {diag["noise_runs"]} independent Gaussian noise vectors (seed {diag["seed"]}), each channel used its single-scan sample standard deviation. The 95th percentile of the largest absolute reconstructed noise value was {delta:.6g} of reference conductivity. The measured peak was {pct(amp)}. The guide's envelope was {pct(max(0,amp-delta))} to {pct(amp+delta)}, giving a {l3["confidence"]} at {pct(diag["threshold_fraction"])}. This noise test excludes smoothing bias, correlated drift and features narrower than the tested resolution.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--data", type=Path)
    args = parser.parse_args()
    run = args.run
    raw = args.data or run / "data" / "regular_G22"
    result = json.loads((run / "results.json").read_text(encoding="utf-8-sig"))
    diag = json.loads((run / "diagnostics.json").read_text(encoding="utf-8-sig"))["L3"]
    if (result["group"], result["dataset_set"]) != ("G22", "regular"):
        raise ValueError("G22 regular data required")
    commit = result["pipeline_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or (run / "pipeline_commit.txt").read_text().strip() != commit:
        raise ValueError("A matching analysis commit is required")
    checks = np.genfromtxt(run / "tables" / "L3_regularisation.csv", delimiter=",", names=True)
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        figures(directory, raw, result["L3"], diag, checks)
        (directory / "brief_note.txt").write_text(note(result, diag), encoding="utf-8")
        (directory / "commit_hash.txt").write_text(commit + "\n", encoding="ascii")
        (directory / "results.json").write_bytes((run / "results.json").read_bytes())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in sorted(directory.iterdir()):
                archive.write(file, file.name)


if __name__ == "__main__":
    main()
