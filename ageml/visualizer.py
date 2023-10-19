"""Implement the data visualizer.

Used in the AgeML project to enable the plotting of modelling results.

Classes:
--------
Visualizer - manages the visualization of data and results.
"""

import matplotlib.pyplot as plt
import math
import os

from .utils import insert_newlines
from .processing import find_correlations

class Visualizer:

    """Manages the visualization of data and results.

    This class uses matplotlib to plot results.

    Parameters
    -----------

    Public methods:
    ---------------
    featyresvsage(self): Plots correlation between features and age.
    """

    def __init__(self):
        """Initialise variables."""
        self.dir = None

    def set_directory(self, path):
        """Set directory to store results."""
        self.dir = path

    def featuresvsage(self, X, Y, feature_names):
        """Plot correlation between features and age.

        Parameters
        ----------
        X: 2D-Array with features; shape=(n,m)
        Y: 1D-Array with age; shape=n
        feature_names: list of names of features, shape=n"""

        # Calculate correlation between features and age
        corr, order = find_correlations(X, Y)

        # Show results
        nplots = len(feature_names)
        plt.figure(figsize=(14,3*math.ceil(nplots/4)))
        print('-----------------------------------')
        print('Features by correlation with Age')
        for i, o in enumerate(order):
            print('%d. %s: %.2f' % (i+1, feature_names[o], corr[o]))
            plt.subplot(math.ceil(nplots/4),4,i+1)
            plt.scatter(Y, X[:,o], s=15)
            plt.ylabel(insert_newlines(feature_names[o], 4))
            plt.xlabel('age (years)')
            plt.title("Corr:%.2f" % corr[o])
        plt.tight_layout()
        plt.savefig(os.path.join(self.dir, 'figures/features_vs_age.svg'))
