import argparse
import json

import numpy as np

from src.utils import train
from src.utils import get_net


def main():
    parser = argparse.ArgumentParser(
        description='Tool optimize data hiding methods with orthogonal moments')
    parser.add_argument("output", metavar="output", help="Path to save model")
    parser.add_argument(
        "-t",
        "--targets",
        required=True,
        help="target data for raining")
    parser.add_argument(
        "-c",
        "--coeficients",
        required=True,
        help="directory with images")

    parser.add_argument(
        "-e",
        "--epochs",
        required=True,
        help="directory with images")

    args = parser.parse_args()
    kwargs = {}
    if hasattr(args, 'epochs'):
        kwargs.update({'epochs': int(args.epochs)})

    targets = np.loadtxt(args.targets, usecols=(1, 2))
    coeficients = np.loadtxt(args.coeficients)

    train.slinear_network_block(coeficients, targets, args.output, **kwargs)


if __name__ == "__main__":
    main()