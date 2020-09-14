from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import fire

BASE = Path('test_data/dataset/')
GRAPHICS = BASE.joinpath('graphics/')
INFERENCES = BASE.joinpath('inferences')
OPTIMIZED = BASE.joinpath('targets/')


def main(data_file, methods, save=None):
    data = np.loadtxt(data_file)

    # Create an axes instance
    fig = plt.figure(1, figsize=(7, 7))
    ax = fig.add_subplot(111)
    
    for index, method in enumerate(methods):
        psnr = data[:, index]
        ax.plot(psnr, label=method)

    ax.set_xlabel('Images')
    ax.set_ylabel('PSNR')
    ax.legend(loc=0)
    ax.grid(True)
    
    fig.show()

    # Save the figures
    if save:
        fig.savefig(save, bbox_inches='tight')


if __name__ == '__main__':
    fire.Fire(main)
