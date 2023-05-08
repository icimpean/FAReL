from enum import Enum
from typing import List, Dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import RegularPolygon, Circle
from matplotlib.path import Path
from matplotlib.projections import register_projection, PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from sklearn.metrics import ConfusionMatrixDisplay

from fairness import SensitiveAttribute
from fairness.fairness_framework import FairnessFramework
from fairness.group import GroupNotion
from fairness.group.group_fairness import GroupFairness
from fairness.history import History
from fairness.individual import IndividualNotion
from dash import dcc
import plotly.express as px
import plotly.graph_objects as go


class FairnessVisualisationDash(object):
    def __init__(self, actions):
        self.actions = actions
        self.action_order = [a.value for a in self.actions]
        self.labels = [a.name for a in self.actions]

    def _value_name(self, value):
        """Get the name to display for a value"""
        # None => 'other'
        if value is None:
            name = "other"
        # Enumeration => name
        elif isinstance(value, Enum):
            name = value.name
        # List => concat names
        elif isinstance(value, list):
            name = "{" + ", ".join([self._value_name(v) for v in value]) + "}"
        # Otherwise, the original value
        else:
            name = value
        # Return the name
        return name

    def _names(self, sensitive_value, other_value):
        return self._value_name(sensitive_value), self._value_name(other_value)

    def _print_probabilities_group(self, name, f_0, prob_0, f_1, prob_1, cm_formula, threshold, exact, approx, y_labels=None):
        text = f"{name}: {f_0} == {f_1}\n"
        text += f"{cm_formula} should be equal for groups\n"

        if y_labels is None:
            y_labels = self.labels

        # Exact fairness
        def _print(p_0, p_1, l):
            nonlocal text
            l = "" if l is None else f" for {l}"
            text += f"\tFound{l}: {p_0} {'!' if p_0 != p_1 else '='}= {p_1}\n"

        if isinstance(prob_0, tuple) and isinstance(prob_1, tuple):
            for p0, p1, label in zip(prob_0, prob_1, y_labels):
                _print(p0, p1, label)
        else:
            _print(prob_0, prob_1, None)
        prefix = 'Fair' if exact else "Unfair"
        text += f"\t{prefix} under exact constraint\n"

        # Approximate fairness
        if threshold is not None:
            text += f"\tWith threshold: | {f_0} - {f_1} | ≤ threshold\n"

            def _thr(p_0, p_1, l):
                nonlocal text
                l = "" if l is None else f"For {l}: "
                w_thresh = abs(p_0 - p_1)
                sign = "≤" if w_thresh <= threshold else "≰"
                text += f"\t                  {l}{w_thresh} {sign} {threshold}\n"

            if isinstance(prob_0, tuple) and isinstance(prob_1, tuple):
                for p0, p1, label in zip(prob_0, prob_1, y_labels):
                    _thr(p0, p1, label)
            else:
                _thr(prob_0, prob_1, None)

            prefix = 'Fair' if approx else "Unfair"
            text += f"\t                  {prefix} under approximate constraint\n"

        print(text)
        return text

    def print_group_notion(self, notion: GroupNotion, feature, sensitive_value, other_value, threshold, exact, approx,
                           prob_sensitive, prob_other):
        s_name, o_name = self._names(sensitive_value, other_value)

        true_1 = self.labels[1]
        true_0 = self.labels[0]
        pred_1 = self.labels[1]
        pred_0 = self.labels[0]

        notion_map = {
            GroupNotion.StatisticalParity: ("Statistical parity",
                                            f"P(y_pred = {true_1} | {feature.name} = {s_name})",
                                            f"P(y_pred = {true_1} | {feature.name} = {o_name})",
                                            f"(TP + FP) / (TP + FP + FN + TN)"),
            GroupNotion.EqualOpportunity: ("Equal Opportunity",
                                           f"P(y_pred = {pred_1} | y_true = {true_1}, {feature.name} = {s_name})",
                                           f"P(y_pred = {pred_1} | y_true = {true_1}, {feature.name} = {o_name})",
                                           f"TPR = TP / (TP + FN)"),
            GroupNotion.PredictiveEquality: ("Predictive Equality",
                                             f"P(y_pred = {pred_1} | y_true = {true_0}, {feature.name} = {s_name})",
                                             f"P(y_pred = {pred_1} | y_true = {true_0}, {feature.name} = {o_name})",
                                             f"FPR = FP / (FP + TN)"),
            GroupNotion.EqualizedOdds: ("Equalized Odds",
                                        f"P(y_pred = {pred_1} | y_true = y, {feature.name} = {s_name})",
                                        f"P(y_pred = {pred_1} | y_true = y, {feature.name} = {o_name})",
                                        f"TPR = TP / (TP + FN) and FPR = FP / (FP + TN)"),
            GroupNotion.OverallAccuracyEquality: ("Overall Accuracy Equality",
                                                  f"P(y_pred = y_true | {feature.name} = {s_name})",
                                                  f"P(y_pred = y_true | {feature.name} = {o_name})",
                                                  f"(TP + TN) / (TP + FN + FP + TN)"),
            GroupNotion.PredictiveParity: ("Predictive Parity",
                                           f"P(y_true = {true_1} | y_pred = {pred_1}, {feature.name} = {s_name})",
                                           f"P(y_true = {true_1} | y_pred = {pred_1}, {feature.name} = {o_name})",
                                           f"PPV = TP / (TP + FP)"),
            GroupNotion.ConditionalUseAccuracyEquality: ("Conditional Use Accuracy Equality",
                                                         f"P(y_true = y | y_pred = y, {feature.name} = {s_name})",
                                                         f"P(y_true = y | y_pred = y, {feature.name} = {o_name})",
                                                         f"PPV = TP / (TP + FP) and NPV = TN / (TN + FN)"),
            GroupNotion.TreatmentEquality: ("Treatment Equality",
                                            f"FN / FP ({feature.name} = {s_name})",
                                            f"FN / FP ({feature.name} = {o_name})",
                                            f"FN / FP"),
        }
        notion_name, text_sensitive, text_other, text_cm_formula = notion_map[notion]

        self._print_probabilities_group(notion_name,
                                        text_sensitive, prob_sensitive,
                                        text_other, prob_other,
                                        text_cm_formula,
                                        threshold, exact, approx)

    def _print_probabilities_individual(self, name, formula, description, threshold, exact, approx, diff,
                                        u_ind, u_pairs):
        text = f"{name}: {formula}\n"
        text += description + "\n"

        # Exact fairness
        def _print(ui, up):
            nonlocal text
            if (len(ui) == 0) and (len(up) == 0):
                len_u = "no"
                text_u = "individuals or pairs"
                lu = None
            elif len(ui) == 0:
                # Pairs
                len_u = len(up)
                text_u = "pairs"
                lu = up
            elif len(up) == 0:
                # Individuals
                len_u = len(ui)
                text_u = "individuals"
                lu = ui
            text += f"\tFound {len_u} unsatisfied {text_u}{':    ' + str(lu) if lu is not None else ''}\n"

        prefix = 'Fair' if exact else "Unfair"
        text += f"\t{prefix} under exact constraint\n"

        # Approximate fairness
        if threshold is not None:
            text += f"\tWith threshold:\n"
            sign = "≤" if diff <= threshold else "≰"
            text += f"\t                  {diff} {sign} {threshold}\n"

            prefix = 'Fair' if approx else "Unfair"
            text += f"\t                  {prefix} under approximate constraint\n"

        _print(u_ind, u_pairs)
        return text

    def print_individual_notion(self, notion: IndividualNotion, threshold, exact, approx, diff, u_ind, u_pairs):
        notion_map = {
            IndividualNotion.IndividualFairness: ("Individual Fairness (Fairness through awareness)",
                                                  f"D(M(v_i), M(v_j)) ≤ d(v_i, v_j)",
                                                  f"For any pair of individuals i and j, the distance between their "
                                                  f"distributions is smaller than the distance between the individuals"),
            IndividualNotion.WeaklyMeritocratic: ("Weakly Meritocratic (Approximate-action fairness)",
                                                  f"Q(s, a) > Q(s, a') + alpha ==> P(s, a, h) >= P(s, a', h)",
                                                  f"Prevent substantially worse actions from being chosen"),
        }
        notion_name, text_formula, text_description = notion_map[notion]

        return self._print_probabilities_individual(notion_name, text_formula, text_description, threshold, exact, approx,
                                                    diff, u_ind, u_pairs)

    def get_fairness_radar_plot(self, fairness_framework: FairnessFramework, requested_notions, threshold,
                                sensitive_attributes):
        labels = []
        differences = []
        exacts = []
        approxs = []
        # Calculate the requested fairness notions and return them
        for notion in requested_notions:
            if isinstance(notion, GroupNotion):
                for attribute in sensitive_attributes:
                    (exact, approx), diff, (prob_sensitive, prob_other) = \
                        fairness_framework.get_group_notion(notion, attribute, fairness_framework.threshold)

                    label = notion.name #+ " " + str(attribute)
                    labels.append(label)
                    differences.append(-diff)
                    exacts.append(exact)
                    approxs.append(approx)
            else:
                # print(notion)
                (exact, approx), diff, (_, _, _) = \
                    fairness_framework.get_individual_notion(notion, fairness_framework.get_individual,
                                                             fairness_framework.threshold,
                                                             fairness_framework.similarity_metric,
                                                             fairness_framework.alpha)
                label = notion.name
                labels.append(label)
                differences.append(-diff)
                exacts.append(exact)
                approxs.append(approx)
        # self.plot_radar(labels, (differences, exacts, approxs), -threshold)
        # self.bar_plot(labels, (differences, exacts, approxs), -threshold)
        return self.bar_plot2(labels, (differences, exacts, approxs), -threshold)

    def get_histogram_plot(self, history: History, feature, sensitive_value, other_value):
        # TODO, highlight sensitive features
        hist_counts, hist_bins = history.histograms[feature]
        values = history.feature_values[feature]
        uniques = len(np.unique(values))
        if uniques < 5:
            bins = uniques
        else:
            bins = "fd"
        # print("bins:", bins)
        # hist_counts, hist_bins = np.histogram(values, bins=bins, density=True)
        # hist_counts = hist_counts * np.diff(hist_bins)
        # x_hist_bins = [(edge + hist_bins[i + 1]) / 2.0 for i, edge in enumerate(hist_bins[0:-1])]
        # print(feature, hist_counts, hist_bins, x_hist_bins)
        # plt.bar(x_hist_bins, hist_counts, width=0.8)

        plt.hist(values, bins=bins)

        # plt.xlim(hist_bins[0], hist_bins[-1])
        plt.title(f"Feature {feature}")
        plt.show()

    def plot_radar(self, labels, data, threshold):
        differences, exact, approx = data

        theta = radar_factory(len(labels), frame='polygon')
        fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
        ax.set_title("Fairness notions", weight='bold', size='medium', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')

        r = np.arange(-1, 0, 0.1)
        # r = np.linspace(0, -1, 10, endpoint=False)
        ax.set_rgrids(r)
        m = min(np.nan_to_num(np.array([threshold, *differences]))) // 1
        steps = abs(m) // 10
        print(min(np.nan_to_num(np.array([threshold, *differences]))), m, steps, threshold, differences)
        print(r)

        ax.set_ylim(m, 0)

        ax.plot(theta, [threshold] * len(labels), color="black")

        col_exact = "green"
        col_approx = "orange"
        col_unfair = "red"
        alpha = 1.0

        # for i, (diff, e, a) in enumerate(zip(differences, exact, approx)):
        #     d = np.full(len(labels), min(differences))
        #     d[i] = diff
        #     col = col_exact if e else (col_approx if a else col_unfair)
        #     ax.plot(theta, d, color=col)
        #     ax.fill(theta, d, closed=True, facecolor=col, alpha=alpha)

        color = "red"
        ax.plot(theta, differences, color=color)
        ax.fill(theta, differences, closed=False, facecolor=color, alpha=0.25)

        ax.set_varlabels(labels)
        plt.show()

    def bar_plot(self, labels, data, threshold):
        differences, exact, approx = data
        # col_exact = "green"
        # col_approx = "orange"
        # col_unfair = "red"
        col_exact = "grey"
        col_approx = "tab:blue"
        col_unfair = "orange"

        for i, (label, diff, e, a) in enumerate(zip(labels, differences, exact, approx)):
            col = col_exact if e else (col_approx if a else col_unfair)
            plt.bar(i, diff, color=col)
            plt.text(i + 0.0, threshold - 0.005, label, rotation=90,
                     verticalalignment='bottom', horizontalalignment='center',
                     # path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                     color='black', fontsize=12)
        plt.hlines(threshold, -2, len(labels) + 1, color="black")
        # plt.xticks(range(len(labels)), labels, rotation=75)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        plt.xlim(-0.5, len(labels) - 0.5)
        plt.ylim(0, max(-1000, min(np.nan_to_num(differences))) - 0.01)
        # TODO
        plt.ylim(0, -0.16 - 0.01)
        plt.xlabel("Fairness Notions", fontsize=14)
        plt.ylabel("Unfairness", fontsize=14)
        # print(list(zip(labels, differences)))
        # print(0, min(np.nan_to_num(differences)) - 0.01)
        plt.tight_layout()

        plt.show()

    def bar_plot2(self, labels, data, threshold):
        differences, exact, approx = data
        # col_exact = "green"
        # col_approx = "orange"
        # col_unfair = "red"
        col_exact = "grey"
        col_approx = "blue"
        col_unfair = "orange"

        # x = []
        # y = []
        # colors = []

        fig = go.Figure()

        for i, (label, diff, e, a) in enumerate(zip(labels, differences, exact, approx)):
            col = col_exact if e else (col_approx if a else col_unfair)
            fig.add_bar(x=[-diff], y=[label], marker_color=col, orientation='h', name=label, showlegend=False)

        fig.add_scatter(x=[-threshold, -threshold], y=[labels[0], labels[-1]], mode='lines', line_color='black',
                        name=f'Threshold={-threshold}', marker_size=2, showlegend=False)

        # fig.show()
        return fig

    def confusion_matrix(self, confusion_matrix, sensitive_feature=None, sensitive_value=None, other_value=None,
                         is_sensitive_value=True, print_cm=True, plot_cm=False):
        name = "Confusion matrix"
        if None not in [sensitive_feature, sensitive_value, other_value]:
            name += f" {sensitive_feature.name} = {sensitive_value if is_sensitive_value else other_value}"

        if print_cm:
            print(name)
            print("(tn, fp, fn, tp):")
            print(confusion_matrix.ravel())
        if plot_cm:
            # ConfusionMatrixDisplay(confusion_matrix, display_labels=[str(a) for a in self.action_order]) \
            #     .plot(cmap="plasma")
            # plt.title(name)
            # plt.show()

            fig = px.imshow(confusion_matrix, text_auto=True,
                            labels=dict(x="Predicted label", y="True label", color="Percentage"),
                            x=[str(a) for a in self.action_order], y=[str(a) for a in self.action_order])
            fig.update_xaxes(side="bottom")
            fig.update_layout(title=name)
            # fig.show()
            return fig

    def fairness_notion_reward(self, fairness_framework: FairnessFramework, notion, threshold,
                               sensitive_attribute=None):
        exact = fairness_framework.exact_fairness[notion]
        approx = fairness_framework.approx_fairness[notion]
        fair_reward = fairness_framework.reward_fairness[notion]
        if sensitive_attribute:
            exact = exact[str(sensitive_attribute)]
            approx = approx[str(sensitive_attribute)]
            fair_reward = fair_reward[str(sensitive_attribute)]
        x = range(len(exact))
        exact = np.cumsum(exact) / len(exact)
        approx = np.cumsum(approx) / len(approx)

        # plt.title(f"{notion.name}")
        # # plt.ylim(-1, 1.05)
        # # TODO
        # # plt.ylim(min(-1, np.min(np.nan_to_num(fair_reward))), 1.05)
        # plt.plot(x, exact, label="exact", color="blue")
        # plt.plot(x, approx, label="approx", color="grey")
        # plt.plot(x, fair_reward, label="fairness", color="red")
        # if threshold:
        #     plt.hlines(-threshold, 0, len(approx), label="threshold", color="black")
        # plt.legend()
        # plt.show()

        lw = 10
        fig = go.Figure()
        fig.add_scatter(x=list(x), y=fair_reward, mode='lines', line_color='red', name=f'fairness', marker_size=lw,
                        showlegend=True)
        fig.add_scatter(x=list(x), y=[-threshold] * len(fair_reward), mode='lines', line_color='black', name=f'threshold',
                        marker_size=lw, showlegend=True)
        fig.update_layout(title=f"{notion.name}")

        # fig.show()
        return fig


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta
