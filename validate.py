"""Independent published-model and machine-output checks for both G22 data sets."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import gradient_matrix


def validate_results(data, result, diagnostics, output, figure_names):
    blind = result["dataset_set"] == "blind"
    template = json.loads((data / ("blind_results.json" if blind else "results_template.json")).read_text())
    assert result.keys() == template.keys()
    assert all(result[key].keys() == template[key].keys() for key in ("L1", "L2", "L3"))
    assert (result["group"], result["dataset_set"]) == (template["group"], template["dataset_set"])
    params = json.loads((data / "L1_params.json").read_text())
    lower = params["EF_op_eV"]
    kt = 8.617333e-5 * params["T"]
    independent_counts = []
    for start in [0.0, lower]:
        energy = np.linspace(start, 2.5, 40001)
        occupied = 1 / (1 + np.exp(np.clip((energy - lower) / kt, -60, 60)))
        independent_counts.append(3.816e15 * np.trapezoid(params["c"] * energy * occupied, energy))
    nb, nf = independent_counts
    ls = result["L1"]
    np.testing.assert_allclose([ls["correction_factor"], ls["n_fermi_op_cm2"], ls["mobility_corrected_cm2_Vs"]],
                               [nb/nf, nf, 1/(1.602176634e-19*nf*params["rho_ohm_sq"])], rtol=1e-6, atol=0)
    ids = pd.read_csv(data / "L2_data.csv")["measurement_index"]
    traj = pd.read_csv(output / "tables/L2_trajectory.csv")
    assert (ids.value_counts() == 3).all()
    assert traj["measurement_index"].tolist() == ids.unique().tolist()
    np.testing.assert_allclose(traj["EF_eV"], result["L2"]["EF_trajectory_eV"], rtol=0, atol=1e-12)
    assert (traj["EF_p16_eV"] <= traj["EF_p84_eV"]).all()
    pair_rms = np.sqrt(np.mean((traj.RH_normalised-traj.RH_fitted)**2 + (traj.Rs_normalised-traj.Rs_fitted)**2))
    assert abs(pair_rms - diagnostics["L2"]["selected_curve_pair_RMS"]) < 1e-12
    d2 = diagnostics['L2']
    hard_gap = d2["r_best_gap"] < .85*d2["r_cone"] and d2["best_gap_candidate_kT"] >= 1
    assert (result["L2"]["model"] == "hard_gap") == hard_gap
    assert result["L2"]["Eg_kT"] == (d2["best_gap_candidate_kT"] if hard_gap else 0)
    draws = np.loadtxt(output / "tables/L2_bootstrap_gaps_kT.csv")
    assert len(draws) == 200
    np.testing.assert_allclose(np.percentile(draws, [16,84]), result["L2"]["Eg_uncertainty_kT"])
    assert abs(np.mean(draws > 0) - d2["hard_gap_selection_fraction"]) < 1e-12
    d3 = diagnostics["L3"]
    spatial = result["L3"]
    m = np.array(spatial["map"])
    assert m.shape == (100,) and np.isfinite(m).all() and abs(m.mean()) < 1e-10
    A = np.load(data / "L3_A.npy")
    L = gradient_matrix()
    H = A.T @ A + d3["eta_selected"] * L.T @ L + 1e-9 * np.eye(100)
    K = np.block([[H, np.ones((100,1))], [np.ones((1,100)), np.zeros((1,1))]])
    B = np.linalg.solve(K, np.vstack([A.T, np.zeros((1,128))]))[:100]
    scans = np.loadtxt(data / "L3_d.csv", delimiter=",")
    error = float(np.max(np.abs(B @ scans.mean(axis=0) - m)))
    assert error < 1e-8
    noise = np.random.default_rng(d3["seed"]).normal(size=(500,128)) * scans.std(axis=0, ddof=1)
    delta = np.percentile(np.max(np.abs(noise @ B.T), axis=1), 95)
    assert abs(spatial["detection_limit_fraction"] - delta) < 1e-9
    peak = float(np.max(np.abs(m)))
    tau = d3["threshold_fraction"]
    expected = "clear pass" if peak+delta < tau else ("clear fail" if peak-delta > tau else "borderline")
    assert spatial["pass"] == bool(peak <= tau) and spatial["confidence"] == expected
    for f in spatial["features"]:
        assert m.reshape(10,10)[f["pixel_row"],f["pixel_column"]] == f["amplitude"]
        assert (f["amplitude"] > 0) == (f["sign"] == "positive")
    for name in figure_names:
        for ext in ("png","svg","pdf"):
            assert (output / "figures" / (name + "." + ext)).stat().st_size > 1000
    return dict(status="passed", group=result["group"], dataset=result["dataset_set"], revision=result["pipeline_commit"],
                schema_matches=True, published_L1_quadrature_rtol=1e-6, measurement_indices=len(traj),
                L2_pair_rms=float(pair_rms), L2_resamples=len(draws), L3_independent_KKT_max_error=error,
                L3_independent_noise_limit=float(delta), max_fraction=peak, spatial_decision=expected,
                figure_files_checked=len(figure_names)*3)
