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
        "-i",
        "--images",
        required=True,
        help="directory with images")
    parser.add_argument(
        "-n",
        "--normalization",
        required=True,
        help="file with mean and std of dataset")
    parser.add_argument(
        "-e",
        "--epochs",
        help="file with mean and std of dataset")

    args = parser.parse_args()

    targets = np.loadtxt(args.targets, usecols=(1, 2))
    normalization = np.loadtxt(args.normalization)
    kwargs = {}
    if hasattr(args, 'epochs'):
        kwargs.update({'epochs': int(args.epochs)})

    train.regresion_mse(args.images, targets, normalization, args.output, **kwargs)


if __name__ == "__main__":
    main()