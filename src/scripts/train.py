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
        help="target data for raining")
    parser.add_argument(
        "-i",
        "--images",
        help="directory with images")

    args = parser.parse_args()

    targets = np.loadtxt(args.targets, usecols=(1, 2))

    train.regresion_mse(args.images, targets, args.output)


if __name__ == "__main__":
    main()