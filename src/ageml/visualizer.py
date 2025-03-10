"""Implement the data visualizer.

Used in the AgeML project to enable the plotting of modelling results.

Classes:
--------
Visualizer - manages the visualization of data and results.
"""

import matplotlib

import matplotlib.pyplot as plt
import math
import numpy as np
import os

from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, roc_auc_score

from .utils import insert_newlines, create_directory, NameTag

matplotlib.use("Agg")
plt.rcParams.update({"font.size": 12})

# RNG for reproducibility
seed = 107146869338163146621163044826586732901
rng = np.random.default_rng(seed)


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

    metrics_vs_num_features(self, mae, mae_std, auc, auc_std, title): Plots MAE and AUC against the number of features used.

    auc_vs_num_features(self, aucs, aucs_std, title): Plot AUCs with standard deviation against the number of features used for each model.

    factors_vs_deltas(self, corrs, groups, labels, markers): Plot bar graph for correlation between factors and deltas.

    deltas_by_groups(self, deltas, labels): Plot box plot for deltas in each group.

    classification_auc(self, y, y_pred, groups): Plot ROC curve.
    """

    def __init__(self, out_dir):
        """Initialise variables."""

        # Setup
        self.set_directory(out_dir)

        # Make diectory for saving the file
        self.path_for_fig = os.path.join(self.dir, "figures")
        create_directory(self.path_for_fig)

        # Set color map
        self.cmap = plt.get_cmap("tab10")

    def set_directory(self, path):
        """Set directory to store results."""
        self.dir = path

    def age_distribution(self, Ys: list, labels=None, name: str = ""):
        """Plot age distribution.

        Parameters
        ----------
        Ys: list of np.arrays containing ages; shape=(m, n).
        labels: # TODO
        name: # TODO"""

        # Plot age distribution
        for Y in Ys:
            plt.hist(Y, bins=20, alpha=1 / len(Ys), density=True)
        if labels is not None:
            plt.legend(labels)
        plt.xlabel("Age (years)")
        plt.ylabel("Count")
        plt.title("Age distribution")

        # Save fig
        filename = "age_distribution_" + name.lower().replace(" ", "_") + ".png"
        plt.savefig(os.path.join(self.path_for_fig, filename))
        plt.close()

    def features_vs_age(self, X: list, Y: list, corr: list, order: list, markers, feature_names, tag: NameTag = None, labels: list = None):
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
        covar_lens = [len(y) for y in Y]
        # Get the unique color set. Get color for each point in the data
        if len(Y) == 0 or len(X) == 0:
            raise TypeError("X and Y must be non-empty lists")
        elif len(Y) > 1:
            color_set = [self.cmap(unique_color) for unique_color in np.arange(len(Y))]
        else:  # If only one covariate, use the same color for all points
            color_set = [self.cmap(0)]
        # Color array for each covariate
        color_list = [len * [color_set[i]] for i, len in enumerate(covar_lens)]

        if labels == ["all"]:
            labels = ["population"]

        # Show results
        nplots = len(feature_names)
        plt.figure(figsize=(14, 3 * math.ceil(nplots / 4)))

        for i, o in enumerate(order[0]):  # Default to order[0] because each covar may have different order
            plt.subplot(math.ceil(nplots / 4), 4, i + 1)
            ax = plt.gca()  # Get current axis
            for i in range(len(color_set)):
                ax.scatter(Y[i][:], X[i][:, o], s=15, c=color_list[i], label=labels[i], alpha=1 / len(labels))
            # Set axis labels, title, and legend
            ax.set_ylabel(insert_newlines(feature_names[o], 4))
            ax.set_xlabel("age (years)")
            title = "Correlation values:"
            for n, label in enumerate(labels):
                title += "\n$\\rho_{%s}$: %s%.3f" % (label, markers[n][o], corr[n][o])
            ax.set_title(title)
            ax.legend(labels)
            plt.suptitle(f"Features vs. Age\n{tag.system}", y=0.99)
        plt.tight_layout()

        # Save file
        filename = f"features_vs_age_controls{'_'+tag.system if tag.system != '' else ''}.png"
        plt.savefig(os.path.join(self.path_for_fig, filename))
        plt.close()

    def true_vs_pred_age(self, y_true, y_pred, tag: NameTag):
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
        plt.title(f"Chronological vs Predicted Age \n [Covariate: {tag.covar}, System: {tag.system}]")
        plt.xlabel("Chronological Age")
        plt.ylabel("Predicted Age")

        # Save file
        filename = (
            f"chronological_vs_pred_age"
            f"{'_' + tag.covar if tag.covar != '' else ''}"
            f"{'_' + tag.system if tag.system != '' else ''}.png"
        )
        plt.savefig(os.path.join(self.path_for_fig, filename))
        plt.close()

    def age_bias_correction(self, y_true, y_pred, y_corrected, tag: NameTag):
        """Plot before and after age bias correction procedure.

        Parameters
        ----------
        y_true: 1D-Array with true age; shape=n
        y_pred: 1D-Array with predicted age before age bias correction; shape=n.
        y_corrected: 1D-Array with predicted age after age bias correction; shape=n
        name: name of the figure"""

        # Find min and max age range to fit in graph
        age_range = np.arange(np.min([y_true, y_pred, y_corrected]), np.max([y_true, y_pred, y_corrected]))

        # Before age-bias correction
        LR_age_bias = LinearRegression(fit_intercept=True)
        LR_age_bias.fit(y_true.reshape(-1, 1), y_pred)
        plt.subplot(1, 2, 1)
        plt.plot(age_range, age_range, color="k", linestyle="dashed")
        plt.plot(age_range, LR_age_bias.predict(age_range.reshape(-1, 1)), color="r")
        plt.scatter(y_true, y_pred)
        plt.title("Before age-bias correction")
        plt.ylabel("Predicted Age")
        plt.xlabel("Chronological Age")

        # After age-bias correction
        LR_age_bias.fit(y_true.reshape(-1, 1), y_corrected)
        plt.subplot(1, 2, 2)
        plt.plot(age_range, age_range, color="k", linestyle="dashed")
        plt.plot(age_range, LR_age_bias.predict(age_range.reshape(-1, 1)), color="r")
        plt.scatter(y_true, y_corrected)
        plt.title("After age-bias correction")
        plt.ylabel("Predicted Age")
        plt.xlabel("Chronological Age")
        plt.tight_layout()

        # Save file
        filename = (
            f"age_bias_correction" f"{'_' + tag.covar if tag.covar != '' else ''}" f"{'_' + tag.system if tag.system != '' else ''}.png"
        )
        plt.suptitle(f"[Covariate: {tag.covar}, System: {tag.system}]\n", y=1.00)
        plt.savefig(os.path.join(self.path_for_fig, filename))
        plt.close()

    def ordering(self, mi_age, mi_discr, feature_names, system_dict, title):
        """Plot in the ssame figure the mutual information for age and discrimination."""

        plt.figure(figsize=(10, 5))
        # Name each point with each feature name
        plt.xlabel("Mutual Information with Age")
        plt.ylabel("Mutual Information with Discrimination")
        plt.title(title)
        
        reversed_dict = {}
    
        for key, value_list in system_dict.items():
            for value in value_list:
                reversed_dict[value] = key

        # Create a dictionary to group points by their category
        categories = {}
        for i, feature in enumerate(feature_names):
            category = reversed_dict.get(feature, "Unknown")
            if category not in categories:
                categories[category] = {"x": [], "y": [], "labels": []}
            categories[category]["x"].append(mi_age[i])
            categories[category]["y"].append(mi_discr[i])
            categories[category]["labels"].append(feature)
    
        # Plot each category with a single label
        for category, data in categories.items():
            plt.scatter(data["x"], data["y"], label=category)
        
            # Add annotations for each point
            for i in range(len(data["x"])):
                plt.annotate(data["labels"][i], 
                            (data["x"][i], data["y"][i]),
                            xytext=(5, 0), 
                            textcoords='offset points')
        plt.legend()

        # Save figure
        filename = f"ordering_{title}.png"
        plt.savefig(os.path.join(self.path_for_fig, filename))
        plt.close()


    def metrics_vs_num_features(self, mae, mae_std, auc, auc_std, title):
        """
        Plots MAE and AUC against the number of features used.
    
        Parameters:
        -----------
        mae: numpy array
            Array containing the mean absolute errors.
        mae_std: numpy array
            Array containing the standard deviation of the MAE.
        auc_std: numpy array
            Array containing the standard deviation of the AUC.
        auc_data: numpy array
            Array containing the AUC values.
        title: str
            Title of the plot.
    """
        
        # Create a figure and axis
        fig, ax = plt.subplots()

        # Determine the number of features
        num_features= np.arange(len(mae)) + 1
    
        # Plot the selected metric 
        line1, = ax.plot(num_features, mae, color='blue', linestyle='-')
        ax.fill_between(num_features, mae - mae_std, mae + mae_std, color='blue', alpha=0.2)
        
        # Set labels and title for the primary y-axis
        ax.set_xlabel('Number of Features Used')
        ax.set_ylabel('MAE (Years)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.set_title(title)

        # Create a twin y-axis for AUC
        ax2 = ax.twinx()
        line2, = ax2.plot(num_features, auc, color='green', linestyle='-')
        ax2.fill_between(num_features, auc - auc_std, auc + auc_std, color='green', alpha=0.2)
        ax2.set_ylabel('AUC', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Set y-axis limit for AUC
        ax2.set_ylim(0.5, 1)

        # Save figure 
        plt.tight_layout()
        filename = f"metrics_vs_num_features_{title}.png"
        plt.savefig(os.path.join(self.path_for_fig, filename))
        plt.close()

    def multiple_metrics_vs_num_features(self, metrics_age, metrics_discrimination, title):

        # Create a combined figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        for o_type, metrics in zip(['age', 'discr'], [metrics_age, metrics_discrimination]):
            mae = metrics['mae']
            mae_std = metrics['mae_std']
            auc = metrics['auc']
            auc_std = metrics['auc_std']

            # Change linestyle for discrimination
            if o_type == 'discr':
                linestyle = '-'
            else:
                linestyle = ':'

            # Determine the number of features
            num_features= np.arange(len(mae)) + 1
    
            # Plot the selected metric 
            line1, = ax.plot(num_features, mae, color='blue', linestyle=linestyle)
            ax.fill_between(num_features, mae - mae_std, mae + mae_std, color='blue', alpha=0.2)
        
            # Set labels and title for the primary y-axis
            ax.set_xlabel('Number of Features Used')
            ax.set_ylabel('MAE (Years)', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.set_title(title)

            # Create a twin y-axis for AUC
            ax2 = ax.twinx()
            line2, = ax2.plot(num_features, auc, color='green', linestyle=linestyle)
            ax2.fill_between(num_features, auc - auc_std, auc + auc_std, color='green', alpha=0.2)
            ax2.set_ylabel('AUC', color='green')
            ax2.tick_params(axis='y', labelcolor='green')

            # Set y-axis limit for AUC
            ax2.set_ylim(0.5, 1)

            # Save lines
            if o_type == 'age':
                line_discri_mae = line1
                line_corr_mae = line2
            else:
                line_discri_auc = line1
                line_corr_auc = line2

        # Combine legends for both discrimination and correlation
        combined_legend_lines = [line_discri_mae, line_corr_mae, line_discri_auc, line_corr_auc]
        combined_legend_labels = ['MAE - Discrimination', 'MAE - Correlation', 'AUC - Discrimination', 'AUC - Correlation']
    
        # Add the combined legend to the plot
        ax.legend(combined_legend_lines, combined_legend_labels, loc='lower left', title="Metrics")

        # Set axis labels and title
        groups = title.split('_')
        ax.set_title(f'MAE and AUC vs Number of Features Used \n{groups[0].upper()} vs {groups[1].upper()}')

        # Save figure 
        plt.tight_layout()
        filename = f"metrics_vs_num_features_{title}.png"
        plt.savefig(os.path.join(self.path_for_fig, filename))
        plt.close()

    
    def auc_vs_num_features(self, aucs, aucs_std, title):
        """Plot AUCs with standard deviation against the number of features used for each model.
        
        Parameters
        ----------
        auc: dict
            Dictionary containing the AUC values for each model
        auc_std: dict
            Dictionary containing the standard deviation of the AUC values for each model
        title: str
            Title of the plot"""

        for model in aucs.keys():
            # Determine the number of features
            num_features = np.arange(len(aucs[model])) + 1

            # Plot the AUC values
            if model == 'logistic_regression':
                label = "Non-BrainAge Model"
            else:
                label = f"BrainAge Model: {model}"
            plt.errorbar(num_features, aucs[model], yerr=aucs_std[model], label=label)

        # Add titles and labels
        if '_' in title:
            name = title.split('_')
            plt.title(f'Ordering: {name[0].upper()} Classification: {name[1].upper()} vs {name[2].upper()}')
        else:
            plt.title(f'{title}')
        plt.xlabel('Number of Features Used')
        plt.ylabel('Mean AUC')
        plt.legend(title='Model', loc='upper left')
        plt.ylim(0.5, 1)
        plt.grid(True)

        # Save figure 
        plt.tight_layout()
        filename = f"auc_vs_num_features_{title}.png"
        plt.savefig(os.path.join(self.path_for_fig, filename))
        plt.close()         

    def factors_vs_deltas(self, corrs, groups, labels, markers, tag: NameTag):
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
                    position = "center"
                else:
                    position = "top"
                ax.text(bar.get_x() + bar.get_width() / 2, height, m, ha="center", va=position, color="red", fontsize=12)
            # Add labels
            ax.set_xlabel("Factor")
            ax.set_ylabel("Correlation with age delta")
            # Set limits to y-axis from -1 to 1, and rotate x-axis labels
            ax.set_ylim([-1, 1])
            ax.tick_params(axis="x", labelrotation=30)

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
        fig.suptitle(f"Correlation of factors with age deltas of {tag.group}", y=0.99)
        filename = f"factors_vs_deltas{'_' + tag.group if tag.group != '' else ''}.png"
        plt.tight_layout()
        plt.savefig(os.path.join(self.path_for_fig, filename))
        plt.close()

    def deltas_by_groups(self, deltas, labels, tag: NameTag):
        """Plot box plot for deltas in each group.

        Parameters
        ----------
        deltas: 2D-Array with deltas; shape=(n, m)
        labels: list of labels for each group; shape=m"""

        # Plot boxplots
        plt.figure(figsize=(10, 5))
        ngroups = len(labels)
        clevels = np.linspace(0, 1, ngroups)
        boxes = plt.boxplot(deltas, labels=labels, patch_artist=True, showfliers=False)
        # Plot patches
        for box, clevel in zip(boxes["boxes"], clevels):
            box.set_facecolor(self.cmap(clevel))
            box.set_alpha(0.5)
        # Plot scatter
        for i, (vals, clevel) in enumerate(zip(deltas, clevels)):
            x = rng.normal(i + 1, 0.04, size=len(vals))
            plt.scatter(x, vals, color=self.cmap(clevel))
        plt.xlabel("Group")
        # Add a horizontal line at y=0
        plt.axhline(y=0, color="r", linestyle="--")
        plt.ylabel("Delta")

        # Save file
        filename = f"clinical_groups_box_plot{'_' + tag.system if tag.system != '' else ''}.png"
        plt.suptitle(f"Age Delta by clinical group. System: {tag.system}", y=0.99)
        plt.savefig(os.path.join(self.path_for_fig, filename))
        plt.close()

    def classification_auc(self, y, y_pred, groups, tag: NameTag):
        """Plot ROC curve.

        Parameters
        ----------
        y: 1D-Array with true labels; shape=n
        y_pred: 1D-Array with predicted labels; shape=n
        system: system name"""

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(y, y_pred)
        auc = roc_auc_score(y, y_pred)

        # Plot ROC curve
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC curve %s vs %s" % (groups[0], groups[1]))
        plt.legend(loc="lower right")

        # Save file
        filename = f"roc_curve_{groups[0]}_vs_{groups[1]}{'_' + tag.system if tag.system != '' else ''}.png"
        plt.suptitle(f"System: {tag.system}")
        plt.savefig(os.path.join(self.path_for_fig, filename))
        plt.close()
