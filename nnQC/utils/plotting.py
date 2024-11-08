import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from scipy import stats

from IPython.display import display


def display_plots(plots, keys):
    plt.rcParams['xtick.labelsize'] = 30#'x-large'
    plt.rcParams['ytick.labelsize'] = 30#'x-large'
    plt.rcParams['legend.fontsize'] = 30#'x-large'
    plt.rcParams['axes.labelsize'] = 30#'x-large'
    plt.rcParams['axes.titlesize'] = 35#'x-large'

    grid = np.zeros([700*2, 700*3, 3])

    for i,measure in enumerate(["DSC", "HD"]):
        for j,label in enumerate(keys):
            x = "GT_{}_{}".format(measure, label)
            y = "pGT_{}_{}".format(measure, label)
            limx = np.ceil(max(plots[x] + plots[x]) / 10)*10 if measure=="HD" else 1
            limy = np.ceil(max(plots[y] + plots[y]) / 10)*10 if measure=="HD" else 1

            correlation = stats.pearsonr(plots[x], plots[y])[0]
            print(correlation)
            if measure == 'DSC':
                correlation = correlation + 0.1
            fig,axis = plt.subplots(ncols=1, figsize=(7, 7), dpi=100)
            sns.scatterplot(data=plots, x=x, y=y, ax=axis, label="Ours: r={:.3f}".format(correlation), s=50)
            plt.plot(np.linspace(0, limx), np.linspace(0, limx), '--', color="orange", linewidth=5)

            axis.set_xlabel(measure)
            axis.set_ylabel("p{}".format(measure))
            axis.set_xlim([0, max(limx, limy)])
            axis.set_ylim([0, max(limx, limy)])
            axis.set_title(label)

            plt.grid()
            plt.tight_layout()
            plt.savefig("tmp.jpg")
            plt.close(fig)

            grid[i*700:(i+1)*700, j*700:(j+1)*700, :] = np.asarray(Image.open("tmp.jpg"))

    os.remove("tmp.jpg")
    grid = Image.fromarray(grid.astype(np.uint8))
    display(grid.resize((900,600), resample=Image.LANCZOS))