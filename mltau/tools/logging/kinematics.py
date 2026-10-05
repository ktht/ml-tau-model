import warnings
import logging
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from mltau.tools.evaluation import kinematics as k
from mltau.tools.general import reinitialize_p4

warnings.filterwarnings("ignore", message=".*sumw are zero.*", category=RuntimeWarning)
warnings.filterwarnings(
    "ignore",
    message=".*divide by zero encountered in scalar divide.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*invalid value encountered in multiply.*",
    category=RuntimeWarning,
)
warnings.filterwarnings(
    "ignore",
    message=".*invalid value encountered in divide.*",
    category=RuntimeWarning,
)
logging.getLogger("matplotlib.mathtext").setLevel(logging.WARNING)


def _log_single_variable(
    pred: np.array,
    truth: np.array,
    var_name: str,
    var_cfg,
    cfg: DictConfig,
    tb_logger,
    current_epoch: int,
    reco_pred: np.ndarray | None = None,
    mode: str = "ratio",
):
    """Log response, resolution, 2D resolution, and bin distribution plots for one variable."""
    evaluator = k.RegressionEvaluator(
        prediction=pred,
        truth=truth,
        bin_edges=var_cfg.bin_edges["all"],
        algorithm="all",
        sample_name="all",
        mode=mode,
        variable=var_name,
    )
    reco_evaluator = None
    if reco_pred is not None:
        reco_evaluator = k.RegressionEvaluator(
            prediction=reco_pred,
            truth=truth,
            bin_edges=var_cfg.bin_edges["all"],
            algorithm="all",
            sample_name="reco",
            mode=mode,
            variable=var_name,
        )

    response_lineplot = k.LinePlot(
        cfg=cfg,
        xlabel=var_cfg.response_plot.xlabel,
        ylabel=var_cfg.response_plot.ylabel,
        xscale=var_cfg.response_plot.xscale,
        yscale=var_cfg.response_plot.yscale,
        ymin=var_cfg.response_plot.ylim[0],
        ymax=var_cfg.response_plot.ylim[1],
        nticks=var_cfg.response_plot.nticks,
    )
    response_lineplot.add_line(
        evaluator.bin_centers, evaluator.responses, evaluator.algorithm, label=""
    )
    if reco_evaluator is not None:
        response_lineplot.ax.plot(
            reco_evaluator.bin_centers,
            reco_evaluator.responses,
            label="Reco jet",
            color="tab:orange",
            ls="--",
            lw=3,
            marker="o",
            ms=8,
        )
        response_lineplot.ax.legend()
    tb_logger.add_figure(
        f"kinematics/{var_name}/responses", response_lineplot.fig, current_epoch
    )
    plt.close(response_lineplot.fig)

    resolution_lineplot = k.LinePlot(
        cfg=cfg,
        xlabel=var_cfg.resolution_plot.xlabel,
        ylabel=var_cfg.resolution_plot.ylabel,
        xscale=var_cfg.resolution_plot.xscale,
        yscale=var_cfg.resolution_plot.yscale,
        ymin=var_cfg.resolution_plot.ylim[0],
        ymax=var_cfg.resolution_plot.ylim[1],
        nticks=var_cfg.resolution_plot.nticks,
    )
    resolution_lineplot.add_line(
        evaluator.bin_centers, evaluator.resolutions, evaluator.algorithm, label=""
    )
    if reco_evaluator is not None:
        resolution_lineplot.ax.plot(
            reco_evaluator.bin_centers,
            reco_evaluator.resolutions,
            label="Reco jet",
            color="tab:orange",
            ls="--",
            lw=3,
            marker="o",
            ms=8,
        )
        resolution_lineplot.ax.legend()
    tb_logger.add_figure(
        f"kinematics/{var_name}/resolutions", resolution_lineplot.fig, current_epoch
    )
    plt.close(resolution_lineplot.fig)

    resolution_2d_plot = k.Resolution2DPlot(
        var_cfg.bin_edges["all"], evaluator, xlabel=var_cfg.response_plot.xlabel
    )
    tb_logger.add_figure(
        f"kinematics/{var_name}/resolutions_2d", resolution_2d_plot.fig, current_epoch
    )
    plt.close(resolution_2d_plot.fig)

    bin_distributions = k.RangeContentPlot(
        var_cfg.bin_edges["all"], xlabel=var_cfg.response_plot.xlabel, mode=mode
    )
    bin_distributions.add_line(evaluator)
    tb_logger.add_figure(
        f"kinematics/{var_name}/bin_distributions", bin_distributions.fig, current_epoch
    )
    plt.close(bin_distributions.fig)

    tb_logger.add_scalar(
        f"kinematics/{var_name}/resolution", evaluator.resolution, current_epoch
    )
    tb_logger.add_scalar(
        f"kinematics/{var_name}/response", evaluator.response, current_epoch
    )


def log_all_kinematics_metrics(
    targets: np.array,
    predictions: np.array,
    reco_jet_p4s: np.array,
    gen_jet_tau_p4s,
    cfg: DictConfig,
    tb_logger,
    current_epoch: int,
    dataset="train",
):
    signal_mask = targets["is_tau"] == 1
    signal_predictions = predictions["kinematics"][signal_mask]
    signal_targets = targets["kinematics"][signal_mask]
    reco = reinitialize_p4(reco_jet_p4s)[signal_mask]

    # --- pt ---
    pred_pt = np.exp(signal_predictions[:, 0]) * reco.pt
    true_pt = np.exp(signal_targets[:, 0]) * reco.pt
    _log_single_variable(
        pred_pt,
        true_pt,
        "pt",
        cfg.metrics.kinematics.pt,
        cfg,
        tb_logger,
        current_epoch,
        reco_pred=np.array(reco.pt),
        mode="ratio",
    )

    # --- eta (direct: index 1 is delta_eta = gen_eta - reco_eta) ---
    pred_eta = np.array(signal_predictions[:, 1]) + np.array(reco.eta)
    true_eta = np.array(signal_targets[:, 1]) + np.array(reco.eta)
    _log_single_variable(
        pred_eta,
        true_eta,
        "eta",
        cfg.metrics.kinematics.eta,
        cfg,
        tb_logger,
        current_epoch,
        reco_pred=np.array(reco.eta),
        mode="diff",
    )

    # --- theta (derived from eta; radians → degrees) ---
    pred_theta_rad = 2 * np.arctan(np.exp(-pred_eta))
    true_theta_rad = 2 * np.arctan(np.exp(-true_eta))
    pred_theta_deg = np.rad2deg(pred_theta_rad)
    true_theta_deg = np.rad2deg(true_theta_rad)
    reco_theta_deg = np.rad2deg(np.array(reco.theta))
    _log_single_variable(
        pred_theta_deg,
        true_theta_deg,
        "theta",
        cfg.metrics.kinematics.theta,
        cfg,
        tb_logger,
        current_epoch,
        reco_pred=reco_theta_deg,
        mode="diff",
    )

    # --- phi (indices 2,3 are sin(Δφ) and cos(Δφ), where Δφ = phi_gen - phi_reco)
    pred_phi = np.array(reco.phi) + np.arctan2(
        signal_predictions[:, 2], signal_predictions[:, 3]
    )
    true_phi = np.array(reco.phi) + np.arctan2(
        signal_targets[:, 2], signal_targets[:, 3]
    )

    pred_phi_deg = np.rad2deg(pred_phi)
    true_phi_deg = np.rad2deg(true_phi)
    _log_single_variable(
        pred_phi_deg,
        true_phi_deg,
        "phi",
        cfg.metrics.kinematics.phi,
        cfg,
        tb_logger,
        current_epoch,
        reco_pred=np.rad2deg(np.array(reco.phi)),
        mode="diff",
    )

    # --- m_vis (index 4 is log(m_gen / m_reco)) ---
    pred_m = np.exp(signal_predictions[:, 4]) * np.array(reco.mass)
    true_m = np.exp(signal_targets[:, 4]) * np.array(reco.mass)
    _log_single_variable(
        pred_m,
        true_m,
        "m_vis",
        cfg.metrics.kinematics.m_vis,
        cfg,
        tb_logger,
        current_epoch,
        reco_pred=np.array(reco.mass),
        mode="ratio",
    )

    # --- energy ---
    pred_energy = np.sqrt((pred_pt * np.cosh(pred_eta)) ** 2 + pred_m**2)
    true_energy = np.sqrt((true_pt * np.cosh(true_eta)) ** 2 + true_m**2)
    _log_single_variable(
        pred_energy,
        true_energy,
        "energy",
        cfg.metrics.kinematics.energy,
        cfg,
        tb_logger,
        current_epoch,
        reco_pred=np.array(reco.energy),
        mode="ratio",
    )

    # --- deltaR (predicted tau vs gen_tau, binned by true pT) ---
    gen_tau = reinitialize_p4(gen_jet_tau_p4s)[signal_mask]
    dphi_dr = np.array(pred_phi) - np.array(gen_tau.phi)
    dphi_dr = np.arctan2(np.sin(dphi_dr), np.cos(dphi_dr))
    deltaR = np.sqrt((np.array(pred_eta) - np.array(gen_tau.eta)) ** 2 + dphi_dr**2)

    dr_cfg = cfg.metrics.kinematics.deltaR
    deltaR_evaluator = k.DeltaREvaluator(
        deltaR=deltaR,
        pt_truth=np.array(true_pt),
        bin_edges=cfg.metrics.kinematics.pt.bin_edges["all"],
        algorithm="all",
    )
    reco_dphi_dr = np.array(reco.phi) - np.array(gen_tau.phi)
    reco_dphi_dr = np.arctan2(np.sin(reco_dphi_dr), np.cos(reco_dphi_dr))
    reco_deltaR = np.sqrt(
        (np.array(reco.eta) - np.array(gen_tau.eta)) ** 2 + reco_dphi_dr**2
    )
    reco_deltaR_evaluator = k.DeltaREvaluator(
        deltaR=reco_deltaR,
        pt_truth=np.array(true_pt),
        bin_edges=cfg.metrics.kinematics.pt.bin_edges["all"],
        algorithm="all",
    )
    median_lineplot = k.LinePlot(
        cfg=cfg,
        xlabel=dr_cfg.median_plot.xlabel,
        ylabel=dr_cfg.median_plot.ylabel,
        xscale=dr_cfg.median_plot.xscale,
        yscale=dr_cfg.median_plot.yscale,
        ymin=dr_cfg.median_plot.ylim[0],
        ymax=dr_cfg.median_plot.ylim[1],
        nticks=dr_cfg.median_plot.nticks,
    )
    median_lineplot.add_line(
        deltaR_evaluator.bin_centers,
        deltaR_evaluator.medians,
        deltaR_evaluator.algorithm,
        label="",
    )
    median_lineplot.ax.plot(
        reco_deltaR_evaluator.bin_centers,
        reco_deltaR_evaluator.medians,
        label="Reco jet",
        color="tab:orange",
        ls="--",
        lw=3,
        marker="o",
        ms=8,
    )
    median_lineplot.ax.legend()
    tb_logger.add_figure(
        "kinematics/deltaR/median_vs_pt_plot", median_lineplot.fig, current_epoch
    )
    plt.close(median_lineplot.fig)

    content_plot = k.DeltaRContentPlot(
        bin_edges=cfg.metrics.kinematics.pt.bin_edges["all"],
        xlabel=dr_cfg.median_plot.xlabel,
        xlim=tuple(dr_cfg.content_xlim),
    )
    content_plot.add_line(deltaR_evaluator)
    tb_logger.add_figure(
        "kinematics/deltaR/bin_distributions", content_plot.fig, current_epoch
    )
    plt.close(content_plot.fig)

    tb_logger.add_scalar(
        "kinematics/deltaR/median", deltaR_evaluator.median, current_epoch
    )
    tb_logger.add_scalar(
        "kinematics/deltaR/mean", float(np.mean(deltaR)), current_epoch
    )
    tb_logger.add_scalar(
        "kinematics/energy/mean_diff",
        float(np.mean(np.array(pred_energy) - np.array(true_energy))),
        current_epoch,
    )
