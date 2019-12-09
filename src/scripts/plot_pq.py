from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

BASE = Path('test_data/dataset/')
PARAMETERS = BASE.joinpath('parameters/')
GRAPHICS = BASE.joinpath('graphics/')
OPTIMIZED = BASE.joinpath('targets/')

def get_data(model, type):
    param = PARAMETERS.joinpath(type)
    output = GRAPHICS.joinpath('{}/pq'.format(type))
    param = param.joinpath('{}.dat'.format(model))
    output = output.joinpath(model)
    data = np.loadtxt(param, usecols=(0, 1))
    mask = data[:, 1] < 5
    data = data[mask]

    return data, output


def main():
    parser = argparse.ArgumentParser(
        description='PQ graphic')
    parser.add_argument("type", metavar="type", help="train or test")
    parser.add_argument("-m", "--model", help="Model", required=True)
    args = parser.parse_args()

    # data
    qk_infered, output = get_data(args.model, args.type)
    qk_optimized = np.loadtxt(
        OPTIMIZED.joinpath('{}.dat'.format(args.type)), usecols=(1, 2))

    # Create an axes instance
    s = 50
    a = 0.4
    fig = plt.figure(1, figsize=(7, 7))
    ax = fig.add_subplot(111)

    ax.scatter(
        qk_infered[:, 0],
        qk_infered[:, 1],
        edgecolor='k',
        c="red",
        s=s,
        marker="s",
        alpha=a,
        label=args.model
    )
    ax.scatter(
        qk_optimized[:, 0],
        qk_optimized[:, 1],
        edgecolor='k',
        c="cornflowerblue",
        s=s,
        marker="s",
        alpha=a,
        label="qk-optimized"
    )

    ax.set_xlabel('P')
    ax.set_ylabel('Q')
    ax.legend(loc=0)
    ax.grid(True)

    # Save the figures
    fig.savefig(str(output), bbox_inches='tight')
    fig.savefig('{}.eps'.format(str(output)), bbox_inches='tight')

    return 0

if __name__ == '__main__':
    main()
