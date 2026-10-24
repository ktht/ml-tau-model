import os
import json
import numpy as np
import mplhep as hep
import awkward as ak
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mltau.tools.io.general import NpEncoder
from mltau.tools.general import reinitialize_p4
from mltau.tools import features as f
from mltau.tools.evaluation.general import calculate_bin_centers, to_bh

hep.style.use(hep.styles.CMS)
plt.rcParams["mathtext.fontset"] = "stix"


DIFF_FUNCTIONS = {
    "phi": f.signedDeltaPhi_deg,
    "eta": f.signedDeltaEta,
    "theta": f.signedDeltaTheta,
}


def plot_regression_confusion_matrix(
    y_true: np.array,
    y_pred: np.array,
    left_bin_edge: float = 0.0,
    right_bin_edge: float = 1.0,
    n_bins: int = 24,
    figsize: tuple = (12, 12),
    cmap: str = "Greys",
    y_label: str = "Predicted",
    x_label: str = "Truth",
    title: str = "Confusion matrix",
    show_projections: bool = True,
):
    """Plots the confusion matrix for the regression task. Although confusion
    matrix is in principle meant for classification task, the problem can be
    solved by binning the predictions and truth values.

    Args:
        y_true : np.array
            The array containing the truth values with shape (n,)
        y_pred : np.array
            The array containing the predicted values with shape (n,)
        left_bin_edge : float
            [default: 0.0] The smallest value
        right_bin_edge : float
            [default: 1.0] The largest value
        n_bins : int
            [default: 24] The number of bins the values will be divided into
            linearly. The number of bin edges will be n_bin_edges = n_bins + 1
        figsize : tuple
            [default: (12, 12)] The size of the figure that will be created
        cmap : str
            [default: "Greys"] Name of the colormap to be used for the
            confusion matrix
        y_label : str
            [default: "Predicted"] The label for the y-axis
        x_label : str
            [default: "Truth"] The label for the x-axis
        title : str
            [default: "Confusion matrix"] The title for the plot
        show_projections : bool
            [default: True] Whether to draw marginal density projections on the
            top (gen/truth) and right (reco/predicted) sides of the matrix

    """
    bin_edges = np.linspace(left_bin_edge, right_bin_edge, num=n_bins + 1)
    fig, ax = plt.subplots(figsize=figsize)
    bin_counts = np.histogram2d(y_true, y_pred, bins=[bin_edges, bin_edges])[0]
    total = bin_counts.sum()
    bin_density = bin_counts / total if total > 0 else bin_counts
    non_zero = bin_density[bin_density > 0]
    vmin = non_zero.min() if len(non_zero) > 0 else 1e-5
    vmax = bin_density.max() if bin_density.max() > 0 else 1.0
    vmin = max(vmin, vmax * 1e-4)
    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    im = ax.pcolor(bin_edges, bin_edges, bin_density.T, cmap=cmap, norm=norm)
    ax.set_aspect("equal")
    ax.set_ylabel(f"{y_label}")
    ax.set_xlabel(f"{x_label}")

    divider = make_axes_locatable(ax)

    if show_projections:
        ax_top = divider.append_axes("top", size="25%", pad=0.05, sharex=ax)
        ax_right = divider.append_axes("right", size="25%", pad=0.05, sharey=ax)
        ax_cbar = divider.append_axes("right", size="5%", pad=0.15)

        # Gen (truth) distribution on top x-axis
        ax_top.hist(
            y_true,
            bins=bin_edges,
            density=True,
            color="k",
            histtype="step",
            alpha=0.7,
        )
        ax_top.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        ax_top.set_ylabel("")
        if title is not None:
            ax_top.set_title(
                title,
                fontsize=18,
                loc="center",
                fontweight="bold",
                style="italic",
                family="monospace",
            )

        # Reco (predicted) distribution on right y-axis
        ax_right.hist(
            y_pred,
            bins=bin_edges,
            density=True,
            orientation="horizontal",
            color="k",
            histtype="step",
            alpha=0.7,
        )
        ax_right.tick_params(
            left=False, bottom=False, labelleft=False, labelbottom=False
        )
        ax_right.set_xlabel("")
    else:
        ax_cbar = divider.append_axes("right", size="5%", pad=0.15)
        if title is not None:
            ax.set_title(
                title,
                fontsize=18,
                loc="center",
                fontweight="bold",
                style="italic",
                family="monospace",
            )

    fig.colorbar(im, cax=ax_cbar)
    return fig, ax


def IQR(ratios: np.array) -> np.array:
    return np.quantile(ratios, 0.75) - np.quantile(ratios, 0.25)


class RegressionEvaluator:
    def __init__(
        self,
        prediction: np.array,
        truth: np.array,
        bin_edges: np.array,
        algorithm: str,
        sample_name: str = "",
        mode: str = "ratio",
        variable: str = "phi",
    ):
        self.prediction = np.array(prediction)
        self.truth = np.array(truth)
        self.gen_tau_pt = self.truth
        self.algorithm = algorithm
        self.mode = mode

        if self.mode == "ratio":
            self.ratios = self.prediction / self.truth
        else:
            diff_function = DIFF_FUNCTIONS[variable]
            self.ratios = diff_function(self.prediction, self.truth)

        self.resolution_function = IQR
        self.sample = sample_name
        self.response_function = np.median
        self.bin_edges = np.array(bin_edges)
        self.bin_centers = calculate_bin_centers(self.bin_edges)[0]
        self.resolutions, self.responses, self.binned_ratios = self._get_binned_values(
            self.ratios, self.truth
        )
        self.resolution, self.response = self._get_overall_resoluton_response()

    def _get_binned_values(self, ratios, gen_tau_vis_pt):
        binned_gen_tau_pt = np.digitize(
            np.array(gen_tau_vis_pt), bins=self.bin_edges
        )  # Biggest idx is overflow
        binned_ratios = [
            ratios[binned_gen_tau_pt == bin_idx]
            for bin_idx in range(1, len(self.bin_edges))
        ]
        resolutions = np.array(
            [
                self.resolution_function(r) if len(r) > 0 else np.nan
                for r in binned_ratios
            ]
        )
        responses = np.array(
            [self.response_function(r) if len(r) > 0 else np.nan for r in binned_ratios]
        )
        if self.mode == "ratio":
            return resolutions / responses, responses, binned_ratios
        else:
            return resolutions, responses, binned_ratios

    def _get_overall_resoluton_response(self):
        response = self.response_function(self.ratios)
        resolution = self.resolution_function(self.ratios)
        if self.mode == "ratio":
            return resolution / response, response
        else:
            return resolution, response

    def print_results(self):
        print("----------------------------")
        print(f"-------- {self.algorithm}--------")
        print(f"Resolution: {self.resolution} \t Response: {self.response}")
        print("----------------------------")


class DeltaREvaluator:
    def __init__(
        self,
        deltaR: np.array,
        pt_truth: np.array,
        bin_edges: np.array,
        algorithm: str,
    ):
        self.deltaR = np.array(deltaR)
        self.pt_truth = np.array(pt_truth)
        self.algorithm = algorithm
        self.bin_edges = np.array(bin_edges)
        self.bin_centers = calculate_bin_centers(self.bin_edges)[0]
        self.medians, self.binned_deltaRs = self._get_binned_values()
        self.median = np.median(self.deltaR)

    def _get_binned_values(self):
        binned_pt = np.digitize(self.pt_truth, bins=self.bin_edges)
        binned_deltaRs = [
            self.deltaR[binned_pt == bin_idx]
            for bin_idx in range(1, len(self.bin_edges))
        ]
        medians = np.array(
            [np.median(b) if len(b) > 0 else np.nan for b in binned_deltaRs]
        )
        return medians, binned_deltaRs


class RangeContentPlot:
    def __init__(self, bin_edges: np.array, xlabel: str, mode: str = "ratio"):
        self.bin_edges = np.array(bin_edges)
        self.xlabel = xlabel
        self.mode = mode
        self.fig, self.axes = self.plot()

    def plot(self):
        fig, rows = plt.subplots(nrows=3, ncols=4, sharex="col", figsize=(16, 9))
        axes = rows.flatten()
        for i, ax in enumerate(axes):
            if i == (len(self.bin_edges) - 1):
                break
            ax.set_title(
                f"{self.xlabel}"
                + r" $\in$"
                + f"$[{self.bin_edges[i]}, {self.bin_edges[i + 1]}]$",
                fontsize=12,
            )
            if self.mode == "ratio":
                ax.set_xlim(0.5, 1.5)
                ax.set_xlabel("$q$", fontsize=12)
            else:
                ax.set_xlim(-5.0, 5.0)  # Center on 0 for differences
                ax.set_xlabel(r"$\Delta$", fontsize=12)
        return fig, axes

    def add_line(self, evaluator):
        self.fig.suptitle(evaluator.algorithm, fontsize=16, fontweight="bold")
        if self.mode == "ratio":
            bins = np.linspace(0.5, 1.5, 101)
        else:
            bins = np.linspace(-5.0, 5.0, 101)
        for ax, data in zip(self.axes, evaluator.binned_ratios):
            if len(data) > 0:
                hep.histplot(
                    to_bh(data, bins=bins),
                    ax=ax,
                    density=True,
                    label=evaluator.algorithm,
                    yerr=None,
                )
            ax.text(
                0.05,
                0.95,
                (
                    f"IQR = {IQR(data) / np.median(data):.3f}"
                    if self.mode == "ratio" and len(data) > 0
                    else f"IQR = {IQR(data):.3f}" if len(data) > 0 else "N/A"
                ),
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
            )

    def save(self, output_path):
        self.fig.savefig(output_path, bbox_inches="tight", format="pdf")
        plt.close("all")


class DeltaRContentPlot:
    def __init__(self, bin_edges: np.array, xlabel: str, xlim: tuple = (0.0, 0.5)):
        self.bin_edges = np.array(bin_edges)
        self.xlabel = xlabel
        self.xlim = xlim
        self.fig, self.axes = self.plot()

    def plot(self):
        fig, rows = plt.subplots(nrows=3, ncols=4, sharex="col", figsize=(16, 9))
        axes = rows.flatten()
        for i, ax in enumerate(axes):
            if i == (len(self.bin_edges) - 1):
                break
            ax.set_title(
                f"{self.xlabel}"
                + r" $\in$"
                + f"$[{self.bin_edges[i]}, {self.bin_edges[i + 1]}]$",
                fontsize=12,
            )
            ax.set_xlim(*self.xlim)
            ax.set_xlabel(r"$\Delta R$", fontsize=12)
        return fig, axes

    def add_line(self, evaluator: str = "DeltaREvaluator"):
        bins = np.linspace(self.xlim[0], self.xlim[1], 101)
        for ax, data in zip(self.axes, evaluator.binned_deltaRs):
            if len(data) > 0:
                hep.histplot(
                    to_bh(data, bins=bins),
                    ax=ax,
                    density=True,
                    label=evaluator.algorithm,
                    yerr=None,
                )
            ax.text(
                0.05,
                0.95,
                (
                    f"median = {np.median(data):.3f}"
                    if len(data) > 0
                    else "median = N/A"
                ),
                transform=ax.transAxes,
                fontsize=8,
                va="top",
                ha="left",
            )

    def save(self, output_path):
        self.fig.savefig(output_path, bbox_inches="tight", format="pdf")
        plt.close("all")


class LinePlot:
    def __init__(
        self,
        cfg: DictConfig,
        xlabel: str,
        ylabel: str,
        xscale: str = "linear",
        yscale: str = "linear",
        ymin: float = 0,
        ymax: float = 1,
        nticks: int = 7,
        axhline_loc=None,
    ):
        self.cfg = cfg
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.xscale = xscale
        self.yscale = yscale
        self.nticks = nticks
        self.ymin, self.ymax = ymin, ymax
        self._y_values = []
        self.axhline_loc = axhline_loc
        self.fig, self.ax = self.plot()

    def add_line(self, x_values, y_values, algorithm, label=""):
        if label == "":
            label = self.cfg.metrics.ALGORITHM_PLOT_STYLES[algorithm].name
        self.ax.plot(
            x_values,
            y_values,
            label=label,
            marker=self.cfg.metrics.ALGORITHM_PLOT_STYLES[algorithm].marker,
            color=self.cfg.metrics.ALGORITHM_PLOT_STYLES[algorithm].color,
            ls=self.cfg.metrics.ALGORITHM_PLOT_STYLES[algorithm].ls,
            lw=self.cfg.metrics.ALGORITHM_PLOT_STYLES[algorithm].lw,
            ms=10,
        )
        finite_y = np.asarray(y_values, dtype=float)
        finite_y = finite_y[np.isfinite(finite_y)]
        if finite_y.size > 0:
            self._y_values.append(finite_y)
            self._autoscale_y()
        self.ax.legend()

    def _autoscale_y(self):
        if self.yscale != "linear" or not self._y_values:
            return

        y = np.concatenate(self._y_values)
        y_min = float(np.min(y))
        y_max = float(np.max(y))

        if np.isclose(y_min, y_max):
            pad = max(abs(y_min) * 0.1, 1e-3)
        else:
            pad = max((y_max - y_min) * 0.15, 1e-3)

        current_lower, current_upper = self.ax.get_ylim()
        lower = min(current_lower, y_min - pad)
        upper = max(current_upper, y_max + pad)

        # Keep positive-only metrics anchored at zero when appropriate.
        if y_min >= 0 and current_lower >= 0 and lower < 0:
            lower = 0.0

        self.ax.set_ylim((lower, upper))
        self.ax.yaxis.set_ticks(np.linspace(lower, upper, self.nticks))
        self._set_y_formatter(lower, upper)

    def _set_y_formatter(self, lower, upper, axis=None):
        if axis is None:
            axis = self.ax
        span = abs(upper - lower)
        magnitude = max(abs(lower), abs(upper))
        if span < 0.01 or magnitude < 0.01:
            axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.4f"))
        elif span < 0.1 or magnitude < 0.1:
            axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.3f"))
        else:
            axis.yaxis.set_major_formatter(ticker.FormatStrFormatter("%0.2f"))

    def plot(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        ax.set_yscale(self.yscale)
        ax.set_xscale(self.xscale)
        ax.set_ylim((self.ymin, self.ymax))
        if self.axhline_loc is not None:
            ax.axhline(self.axhline_loc, ls="--", color="k")
        # ax.grid()
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.linspace(start, end, self.nticks))
        self._set_y_formatter(start, end, axis=ax)
        return fig, ax

    def save(self, output_path: str):
        self._autoscale_y()
        self.fig.savefig(output_path, bbox_inches="tight", format="pdf")
        plt.close("all")


class Resolution2DPlot:
    def __init__(self, bin_edges: np.array, evaluator, xlabel: str):
        self.bin_edges = np.array(bin_edges)
        self.evaluator = evaluator
        self.xlabel = xlabel
        self.fig, self.ax = self.plot()

    def plot(self):
        fig, ax = plot_regression_confusion_matrix(
            y_true=self.evaluator.truth,
            y_pred=self.evaluator.prediction,
            left_bin_edge=self.bin_edges[0],
            right_bin_edge=self.bin_edges[-1],
            n_bins=50,
            figsize=(8, 9),
            cmap="Greys",
            y_label=f"Predicted {self.xlabel}",
            x_label=f"True {self.xlabel}",
            title=self.evaluator.algorithm,
        )
        return fig, ax

    def save(self, output_path: str):
        self.fig.savefig(output_path, bbox_inches="tight", format="pdf")
        plt.close("all")


class RegressionMultiEvaluator:
    def __init__(
        self,
        output_dir: str,
        cfg: DictConfig,
        sample: str,
        var_cfg,
        axhline_loc: float = 0.0,
    ):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.cfg = cfg
        self.sample = sample
        self.var_cfg = var_cfg
        self.response_lineplot = LinePlot(
            cfg=self.cfg,
            xlabel=var_cfg.response_plot.xlabel,
            ylabel=var_cfg.response_plot.ylabel,
            xscale=var_cfg.response_plot.xscale,
            yscale=var_cfg.response_plot.yscale,
            ymin=var_cfg.response_plot.ylim[0],
            ymax=var_cfg.response_plot.ylim[1],
            nticks=var_cfg.response_plot.nticks,
            axhline_loc=axhline_loc,
        )
        self.resolution_lineplot = LinePlot(
            cfg=self.cfg,
            xlabel=var_cfg.resolution_plot.xlabel,
            ylabel=var_cfg.resolution_plot.ylabel,
            xscale=var_cfg.resolution_plot.xscale,
            yscale=var_cfg.resolution_plot.yscale,
            ymin=var_cfg.resolution_plot.ylim[0],
            ymax=var_cfg.resolution_plot.ylim[1],
            nticks=var_cfg.resolution_plot.nticks,
        )
        self.bin_distributions_plots = {}
        self.resolution_2d_plots = {}
        self.resolution_performance_info = {}

    def combine_results(self, evaluators: list, plot_pred_vs_true_2d: bool = True):
        for evaluator in evaluators:
            self.response_lineplot.add_line(
                evaluator.bin_centers,
                evaluator.responses,
                evaluator.algorithm,
                label="",
            )
            self.resolution_lineplot.add_line(
                evaluator.bin_centers,
                evaluator.resolutions,
                evaluator.algorithm,
                label="",
            )
            if plot_pred_vs_true_2d:
                self.resolution_2d_plots[evaluator.algorithm] = Resolution2DPlot(
                    evaluator.bin_edges, evaluator, xlabel=self.var_cfg.response_plot.xlabel
                )
            self.bin_distributions_plots[evaluator.algorithm] = RangeContentPlot(
                evaluator.bin_edges,
                xlabel=self.var_cfg.response_plot.xlabel,
                mode=evaluator.mode,
            )
            self.bin_distributions_plots[evaluator.algorithm].add_line(evaluator)
            if evaluator.sample not in self.resolution_performance_info.keys():
                self.resolution_performance_info[evaluator.sample] = {}
            self.resolution_performance_info[evaluator.sample][evaluator.algorithm] = {
                "resolution": evaluator.resolution,
                "response": evaluator.response,
            }

    def save(self):
        responses_output_path = os.path.join(self.output_dir, "responses.pdf")
        self.response_lineplot.save(responses_output_path)
        resolutions_output_path = os.path.join(self.output_dir, "resolutions.pdf")
        self.resolution_lineplot.save(resolutions_output_path)
        for algorithm, res_2d_plot in self.resolution_2d_plots.items():
            res_2d_plot_output_path = os.path.join(
                self.output_dir, f"{algorithm}_{self.sample}_2D_resolution.pdf"
            )
            res_2d_plot.save(res_2d_plot_output_path)
        for algorithm, bin_plot in self.bin_distributions_plots.items():
            bin_distributions_plot_output_path = os.path.join(
                self.output_dir, f"{algorithm}_{self.sample}_bin_contents.pdf"
            )
            bin_plot.save(bin_distributions_plot_output_path)
        resolution_performance_info_path = os.path.join(
            self.output_dir, "performance_info.json"
        )
        with open(resolution_performance_info_path, "wt") as out_file:
            json.dump(
                self.resolution_performance_info, out_file, indent=4, cls=NpEncoder
            )


class KinematicsEvaluator:
    def __init__(
        self,
        predicted_p4: ak.Array,
        true_p4: ak.Array,
        cfg: DictConfig,
        algorithm: str,
        sample_name: str = "z",
    ):
        self.predicted_p4 = reinitialize_p4(predicted_p4)
        self.true_p4 = reinitialize_p4(true_p4)
        self.cfg = cfg
        self.algorithm = algorithm
        self.sample_name = sample_name
        self.var_evaluators = self._fill_evaluators()

    def _fill_evaluators(self):
        deltaR = f.deltaR_thetaPhi(
            theta1=self.predicted_p4.theta,
            phi1=self.predicted_p4.phi,
            theta2=self.true_p4.theta,
            phi2=self.true_p4.phi,
        )
        evaluators = {
            "pt": RegressionEvaluator(
                prediction=self.predicted_p4.pt,
                truth=self.true_p4.pt,
                bin_edges=self.cfg.metrics.kinematics.pt.bin_edges[self.sample_name],
                algorithm=self.algorithm,
                sample_name=self.sample_name,
                variable="pt",
                mode="ratio",
            ),
            "eta": RegressionEvaluator(
                prediction=self.predicted_p4.eta,
                truth=self.true_p4.eta,
                bin_edges=self.cfg.metrics.kinematics.eta.bin_edges[self.sample_name],
                algorithm=self.algorithm,
                sample_name=self.sample_name,
                variable="eta",
                mode="diff",
            ),
            "theta": RegressionEvaluator(
                prediction=np.rad2deg(self.predicted_p4.theta),
                truth=np.rad2deg(self.true_p4.theta),
                bin_edges=self.cfg.metrics.kinematics.theta.bin_edges[self.sample_name],
                algorithm=self.algorithm,
                sample_name=self.sample_name,
                variable="theta",
                mode="diff",
            ),
            "phi": RegressionEvaluator(
                prediction=np.rad2deg(np.asarray(self.predicted_p4.phi)),
                truth=np.rad2deg(np.asarray(self.true_p4.phi)),
                bin_edges=self.cfg.metrics.kinematics.phi.bin_edges[self.sample_name],
                algorithm=self.algorithm,
                sample_name=self.sample_name,
                variable="phi",
                mode="diff",
            ),
            "m_vis": RegressionEvaluator(
                prediction=self.predicted_p4.mass,
                truth=self.true_p4.mass,
                bin_edges=self.cfg.metrics.kinematics.m_vis.bin_edges[self.sample_name],
                algorithm=self.algorithm,
                sample_name=self.sample_name,
                variable="m_vis",
                mode="ratio",
            ),
            "energy": RegressionEvaluator(
                prediction=self.predicted_p4.energy,
                truth=self.true_p4.energy,
                bin_edges=self.cfg.metrics.kinematics.energy.bin_edges[
                    self.sample_name
                ],
                algorithm=self.algorithm,
                sample_name=self.sample_name,
                variable="energy",
                mode="ratio",
            ),
            "deltaR": DeltaREvaluator(
                deltaR=deltaR,
                pt_truth=np.array(self.true_p4.pt),
                bin_edges=self.cfg.metrics.kinematics.pt.bin_edges[self.sample_name],
                algorithm=self.algorithm,
            ),
        }
        return evaluators


class KinematicsMultiEvaluator:
    def __init__(self, output_dir: str, cfg: DictConfig, sample: str):
        self.output_dir = output_dir
        self.cfg = cfg
        self.sample = sample
        self.cfg = cfg
        self.variables = {
            "pt": "ratio",
            "eta": "diff",
            "phi": "diff",
            "theta": "diff",
            "m_vis": "ratio",
            "energy": "ratio",
        }
        self.multi_evaluators = {}
        self._init_plots()

    def _init_plots(self):
        plots = {}
        for variable, mode in self.variables.items():
            var_cfg = self.cfg.metrics.kinematics[variable]
            vme_output_dir = os.path.join(self.output_dir, variable)
            os.makedirs(vme_output_dir, exist_ok=True)
            axhline_loc = 1.0 if mode == "ratio" else 0.0
            self.multi_evaluators[variable] = RegressionMultiEvaluator(
                vme_output_dir, self.cfg, self.sample, var_cfg, axhline_loc=axhline_loc
            )
        dr_cfg = self.cfg.metrics.kinematics.deltaR
        self.multi_evaluators["deltaR"] = LinePlot(
            cfg=self.cfg,
            xlabel=dr_cfg.median_plot.xlabel,
            ylabel=dr_cfg.median_plot.ylabel,
            xscale=dr_cfg.median_plot.xscale,
            yscale=dr_cfg.median_plot.yscale,
            ymin=dr_cfg.median_plot.ylim[0],
            ymax=dr_cfg.median_plot.ylim[1],
            nticks=dr_cfg.median_plot.nticks,
        )
        return plots

    def combine_results(self, evaluators: list, plot_pred_vs_true_2d: bool = True):
        for variable in self.variables:
            var_evaluators = [
                evaluator.var_evaluators[variable] for evaluator in evaluators
            ]
            self.multi_evaluators[variable].combine_results(
                var_evaluators, plot_pred_vs_true_2d=plot_pred_vs_true_2d
            )
        for evaluator in evaluators:
            var_evaluator = evaluator.var_evaluators["deltaR"]
            self.multi_evaluators["deltaR"].add_line(
                var_evaluator.bin_centers,
                var_evaluator.medians,
                var_evaluator.algorithm,
                label=evaluator.algorithm,
            )

    def save(self):
        for variable in self.variables:
            self.multi_evaluators[variable].save()
        deltaR_output_dir = os.path.join(self.output_dir, "deltaR")
        os.makedirs(deltaR_output_dir, exist_ok=True)
        deltaR_output_path = os.path.join(deltaR_output_dir, "median_plot.pdf")
        self.multi_evaluators["deltaR"].save(deltaR_output_path)
