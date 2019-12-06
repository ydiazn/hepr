from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

BASE = Path('test_data/dataset/')
GRAPHICS = BASE.joinpath('graphics/')
INFERENCES = BASE.joinpath('inferences')
OPTIMIZED = BASE.joinpath('targets/')

def get_data(type):
    inference = INFERENCES.joinpath(type)
    output = GRAPHICS.joinpath('{}/psnr'.format(type))
    return inference, output


def main():
    parser = argparse.ArgumentParser(
        description='PSNR graphic')
    parser.add_argument("type", metavar="type", help="train or test")
    args = parser.parse_args()

    # data
    inference, output = get_data(args.type)
    CONVOLUTION = inference.joinpath('convolution.dat')
    MULTIOUTPUT = inference.joinpath('multiouput_regression.dat')
    TARGET = OPTIMIZED.joinpath('{}.dat'.format(args.type))

    qk_resnet18 = np.loadtxt(CONVOLUTION, usecols=(0))
    qk_multioutput = np.loadtxt(MULTIOUTPUT, usecols=(2))
    target = np.loadtxt(TARGET, usecols=(3, 4))
    qk_optimized = target[:, 0]
    qk_dct = target[:, 1]

    mask = qk_multioutput > 10
    qk_resnet18 = qk_resnet18[mask]
    qk_multioutput = qk_multioutput[mask]
    qk_optimized = qk_optimized[mask]
    qk_dct = qk_dct[mask]

    # Delete psnr values equals to cero from multiobjetive
    #qk_multioutput = qk_multioutput[qk_multioutput != 0]

    # Create an axes instance
    fig = plt.figure(1, figsize=(7, 7))
    ax = fig.add_subplot(111)

    ax.plot(qk_multioutput, label="qk-multiouput")
    ax.plot(qk_resnet18, label="Convolution")
    ax.plot(qk_optimized, label="qk-optimized")
    ax.plot(qk_dct, label="qk-dct")
    ax.set_xlabel('Images')
    ax.set_ylabel('PSNR')
    ax.legend(loc=0)
    ax.grid(True)

    # Save the figures
    fig.savefig(str(output), bbox_inches='tight')
    fig.savefig('{}.eps'.format(str(output)), bbox_inches='tight')

    return 0

if __name__ == '__main__':
    main()
