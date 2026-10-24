import awkward as ak
import matplotlib.pyplot as plt
import numpy as np

from mltau.tools.evaluation import kinematics as k
from mltau.tools.general import reinitialize_p4


def plot_performance(
    predictions: dict,
    true: np.ndarray,
    x_min: float = 0.0,
    x_max: float = 1.5,
    nbins: int = 15,
    xlabel: str = "",
    ylabel: str = "Density",
    output_path: str | None = None,
):  # Regression

    bins = np.linspace(x_min, x_max, nbins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    true_hist = k.to_bh(true, bins)
    pred_hists = {}
    for name, pred in predictions.items():
        pred_hists[name] = k.to_bh(pred, bins)

    true_hist[:] = true_hist.values() / (true_hist.values().sum() * np.diff(bins))
    for name, pred_hist in pred_hists.items():
        pred_hist[:] = pred_hist.values() / (pred_hist.values().sum() * np.diff(bins))

    fig, (ax, ax_diff) = plt.subplots(
        2,
        1,
        figsize=(6, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
    )

    ax.errorbar(
        bin_centers,
        true_hist,
        fmt="o",
        color="k",
        markersize=5,
        label="Truth",
    )
    line_styles = ["solid", "dashed", "dashdot", "dotted"]
    for index, (name, pred_hist) in enumerate(pred_hists.items()):
        eb = ax.errorbar(
            bin_centers,
            pred_hist,
            fmt="o",
            linestyle=line_styles[index % len(line_styles)],
            markersize=5,
            label=name,
        )
        color = eb.lines[0].get_color()
        ax_diff.errorbar(
            bin_centers,
            true_hist - pred_hist,
            marker="o",
            linestyle="none",
            markersize=4,
            color=color,
        )
    ax_diff.axhline(0, linewidth=1, color="k")

    ax.set_xlim(x_min, x_max)
    ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax_diff.set_xlabel("x")
    ax_diff.set_ylabel("Diff.")
    ax_diff.set_xlim(x_min, x_max)

    plt.show()


def calculate_x_dm0(E_pi, E_tau: float = 45.6):
    """x = E_pi/E_tau is used for polarization studies shape of the x distribution carries information about the tau polarization.
    For ultrarelativistic tau decays this can be approximated as x = (1 + cos(theta*))/2
    For FCCee Z->tautau we can take the E_tau to be 45.6 GeV.
    """
    x = E_pi / E_tau
    return x


def hps_x_dm0(hps_data: ak.Array):
    # Filter out the true and pred one-prong decays
    gen_mask = hps_data.gen_jet_tau_decaymode == 0
    pred_mask = hps_data.tau_decaymode == 0
    # As we are targeting the visible tau energy, then we can use directly the predicted tau_p4s and gen_jet_tau_p4
    # as the E_pi to calculate the x_dm0
    pred_op_energy = reinitialize_p4(hps_data.tau_p4s).energy[pred_mask]
    true_op_energy = reinitialize_p4(hps_data.gen_jet_tau_p4).energy[gen_mask]
    true_x_dm0 = calculate_x_dm0(true_op_energy)
    pred_x_dm0 = calculate_x_dm0(pred_op_energy)
    return pred_x_dm0, true_x_dm0


# TODO: Need to evaluate also the pT and the dR between pred tau and true vis tau for HPS and ParTauDETR like in the following:
#
# from mltau.tools.evaluation import kinematics as k
# hps_ke = k.KinematicsEvaluator(
#     predicted_p4=hps_data.tau_p4s,
#     true_p4=hps_data.gen_jet_tau_p4,
#     cfg=cfg,
#     algorithm="HPS",
#     sample_name='z'
# )
# kme = k.KinematicsMultiEvaluator(output_dir="/home/laurits/HPS_kin_eval", cfg=cfg, sample="z")
# kme.combine_results([hps_ke])
