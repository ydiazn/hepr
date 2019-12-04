import argparse
import json

import numpy as np

from src.utils.regression import multiouput_regressor
from src.utils import get_net


def main():
    parser = argparse.ArgumentParser(
        description='Tool optimize data hiding methods with orthogonal moments')
    parser.add_argument("output", metavar="output", help="Path to save PSRN values")
    parser.add_argument(
        "-t",
        "--target",
        help="file with target data")
    parser.add_argument(
        "-ts",
        "--target_test",
        help="file with target data")
    parser.add_argument(
        "-i",
        "--input",
        help="file with input data")
    parser.add_argument(
        "-is",
        "--input_test",
        help="file with input data")

    args = parser.parse_args()

    target = np.loadtxt(args.target, usecols=(1, 2))
    target_test = np.loadtxt(args.target_test, usecols=(1, 2))
    input = np.loadtxt(args.input)
    input_test = np.loadtxt(args.input_test)

    multiouput_regressor(input, target, input_test, target_test, args.output)


if __name__ == "__main__":
    main()