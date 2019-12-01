import argparse
import json

import numpy as np

from src.utils import train
from src.utils import get_net


def main():
    parser = argparse.ArgumentParser(
        description='Tool optimize data hiding methods with orthogonal moments')
    parser.add_argument("output", metavar="outpu", help="Path to save model")
    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        help="Dataset")
    parser.add_argument(
        "-t",
        "--target",
        help="target data for raining")
    parser.add_argument(
        "-e",
        "--epochs",
        help="Number of epochs")

    args = parser.parse_args()

    data = np.loadtxt(args.data)
    target = np.loadtxt(args.target)
    kwargs = dict(n_feature=8, n_output=2)
    if hasattr(args, 'epoch'):
        kwargs.update({'epoch': args.epoch})
    net = get_net(args.net, **kwargs)

    train.regresion_mse(data, target, args.output, net)


if __name__ == "__main__":
    main()