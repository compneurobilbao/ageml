"""Implement the data visualizer.

Used in the AgeML project to enable the plotting of modelling results.

Classes:
--------
Visualizer - manages the visualization of data and results.
"""

import matplotlib.pyplot as plt
import math
import numpy as np
import os

from sklearn.linear_model import LinearRegression

from .utils import insert_newlines, create_directory


class Visualizer:

    """Manages the visualization of data and results.

    This class uses matplotlib to plot results.

    Parameters
    -----------
    out_dir: path to output directory where to save results

    Public methods:
    ---------------

    age_distribution(self, Y): Plot age distribution.

    features_vs_age(self, X, Y, features_name): Plots correlation between features and age.

    true_vs_pred_age(self, y_true, y_pred): Plot true age vs predicted age.

    age_bias_correction(self, y_true, y_pred, y_corrected): Plot before and after age bias correction procedure.

    deltas_by_groups(self, deltas, labels): Plot box plot for deltas in each group.
    """

    def __init__(self, out_dir):
        """Initialise variables."""

        # Setup
        self.set_directory(out_dir)

        # Make diectory for saving the file
        self.path_for_fig = os.path.join(self.dir, "figures")
        create_directory(self.path_for_fig)

        # Set color map
        self.cmap = plt.get_cmap("viridis")

    def set_directory(self, path):
        """Set directory to store results."""
        self.dir = path

    def age_distribution(self, Ys, labels=None, name: str = ""):
        """Plot age distribution.

        Parameters
        ----------
        Ys: 2D-Array with list of ages; shape=(m, n).
        labels: # TODO
        name: # TODO"""

        # Plot age distribution
        for Y in Ys:
            plt.hist(Y, bins=20, alpha=1 / len(Ys))
        if labels is not None:
            plt.legend(labels)
        plt.xlabel("Age (years)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(self.path_for_fig, "age_distribution_%s.svg" % name))
        plt.close()

    def features_vs_age(self, X, Y, corr, order, markers, feature_names, idxs: list = None, labels: list = None):
        """Plot correlation between features and age.

        Parameters
        ----------
        X: 2D-Array with features; shape=(n, m)
        Y: 1D-Array with age; shape=n
        corr: 1D-Array with correlation coefficients; shape=m
        order: 1D-Array with order of features; shape=m
        markers: list of markers for significant features; shape=m
        feature_names: list of names of features, shape=m
        idxs: list of list of indexes of features in each group; list of lists.
              outer list shape=(n_dfs), inner list shape=(n_points_in_df)
        labels: list of labels for each group; shape=n_dfs"""
        # Get the unique color set. Get color for each point in the data
        if idxs is not None:
            color_set = [self.cmap(unique_color) for unique_color in np.linspace(0, 1, len(idxs))]
            color_list = [(color_set[group_index]) for group_index, group_elements in enumerate(idxs) for _ in group_elements]
        else:
            idxs = [np.arange(len(Y)).tolist()]
            color_set = [self.cmap(0)]
            color_list = [self.cmap(0) for _ in range(len(Y))]
        if labels is None:
            labels = ['population']
        # Show results
        nplots = len(feature_names)
        plt.figure(figsize=(14, 3 * math.ceil(nplots / 4)))
        for i, o in enumerate(order):
            plt.subplot(math.ceil(nplots / 4), 4, i + 1)
            ax = plt.gca()  # Get current axis
            for i in range(len(color_set)):  # Remap indexes to be consecutive
                color = [color_list[n] for n in idxs[i]]  # Take colors from list
                ax.scatter(Y[idxs[i]], X[idxs[i], o],
                           s=15, c=color, label=labels[i])
            ax.set_ylabel(insert_newlines(feature_names[o], 4))
            ax.set_xlabel("age (years)")
            ax.set_title("%s Corr:%.2f" % (markers[o], corr[o]))
            ax.legend(labels)
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_for_fig, "features_vs_age.svg"))
        plt.close()

    def true_vs_pred_age(self, y_true, y_pred):
        """Plot true age vs predicted age.

        Parameters
        ----------
        y_true: 1D-Array with true age; shape=n
        y_pred: 1D-Array with predicted age; shape=n."""

        # Find min and max age range to fit in graph
        age_range = np.arange(np.min([y_true, y_pred]), np.max([y_true, y_pred]))

        # Plot true vs predicted age
        plt.scatter(y_true, y_pred)
        plt.plot(age_range, age_range, color="k", linestyle="dashed")
        plt.xlabel("True Age")
        plt.ylabel("Predicted Age")
        plt.savefig(os.path.join(self.path_for_fig, "true_vs_pred_age.svg"))
        plt.close()

    def age_bias_correction(self, y_true, y_pred, y_corrected):
        """Plot before and after age bias correction procedure.

        Parameters
        ----------
        y_true: 1D-Array with true age; shape=n
        y_pred: 1D-Array with predicted age before age bias correction; shape=n.
        y_corrected: 1D-Array with predicted age after age bias correction; shape=n"""

        # Find min and max age range to fit in graph
        age_range = np.arange(
            np.min([y_true, y_pred, y_corrected]), np.max([y_true, y_pred, y_corrected])
        )

        # Before age-bias correction
        LR_age_bias = LinearRegression(fit_intercept=True)
        LR_age_bias.fit(y_true.reshape(-1, 1), y_pred)
        plt.subplot(1, 2, 1)
        plt.plot(age_range, age_range, color="k", linestyle="dashed")
        plt.plot(age_range, LR_age_bias.predict(age_range.reshape(-1, 1)), color="r")
        plt.scatter(y_true, y_pred)
        plt.title("Before age-bias correction")
        plt.ylabel("Predicted Age")
        plt.xlabel("True Age")

        # After age-bias correction
        LR_age_bias.fit(y_true.reshape(-1, 1), y_corrected)
        plt.subplot(1, 2, 2)
        plt.plot(age_range, age_range, color="k", linestyle="dashed")
        plt.plot(age_range, LR_age_bias.predict(age_range.reshape(-1, 1)), color="r")
        plt.scatter(y_true, y_corrected)
        plt.title("After age-bias correction")
        plt.ylabel("Predicted Age")
        plt.xlabel("True Age")
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_for_fig, "age_bias_correction.svg"))
        plt.close()

    def factors_vs_deltas(self, corrs, groups, labels, markers):
        """Plot bar graph for correlation between factors and deltas.
        
        Parameters
        ----------
        corr: 2D-Array with correlation coefficients; shape=(n, m)
        labels: list of labels for each factor; shape=m,
        markers: list of list of significance markers; shape=(n, m)"""

        # Plot bar graph
        fig, axs = plt.subplots(nrows=len(corrs), ncols=1)

        def bargraph(ax, labels, corrs, markers, group):
            """Plot bar graph."""
            # Order from highest to lowest correlation
            corr, labels, marker = zip(*sorted(zip(corrs, labels, markers), reverse=True))
            # Create a bar graph
            bars = ax.bar(labels, corr)
            # Add significant markers
            for j, m in enumerate(marker):
                bar = bars[j]
                height = bar.get_height()
                if height > 0:
                    position = 'center'
                else:
                    position = 'top'
                ax.text(bar.get_x() + bar.get_width() / 2, height, m, ha='center', va=position, color='red', fontsize=12)
            # Add labels
            ax.set_xlabel("Factor")
            ax.set_ylabel("Correlation with delta")
            ax.set_title("%s" % group)

            return ax
        
        # Plot each group
        if len(corrs) == 1:
            ax = bargraph(axs, labels, corrs[0], markers[0], groups[0])
        else:
            for i, ax in enumerate(axs):
                ax = bargraph(ax, labels, corrs[i], markers[i], groups[i])

        # Save figure
        fig.set_size_inches(10, 5 * len(corrs))
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_for_fig, "factors_vs_deltas.svg"))
        plt.close()

    def deltas_by_groups(self, deltas, labels):
        """Plot box plot for deltas in each group.

        Parameters
        ----------
        deltas: 2D-Array with deltas; shape=(n, m)
        labels: list of labels for each group; shape=m"""

        # Plot boxplots
        plt.figure(figsize=(10, 5))
        num_groups = len(labels)
        boxes = plt.boxplot(deltas, labels=labels, patch_artist=True)
        for i, box in enumerate(boxes["boxes"]):
            box.set_facecolor(self.cmap(i / num_groups))
        plt.xlabel("Gruop")
        plt.ylabel("Delta")
        plt.savefig(os.path.join(self.path_for_fig, "clinical_groups_box_plot.svg"))
        plt.close()
