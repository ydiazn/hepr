import argparse

import numpy as np

from src.utils import process


def main():
    parser = argparse.ArgumentParser(
        description='Tool optimize data hiding methods with orthogonal moments')
    parser.add_argument("indir", metavar="indir", help="Images directory")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="File where PSNR values will be saved")
    parser.add_argument(
        "-d",
        "--data",
        required=True,
        help="File with data to hide"
    )
    parser.add_argument(
        "-p",
        "--parameters",
        required=True,
        help="Network model file"
    )
    args = parser.parse_args()

    with open(args.data, 'r') as file:
        data = file.read()

    parameters = np.loadtxt(args.parameters)

    process.qkrawtchouk8x8_DCT(
        indir=args.indir, file=args.output, data=data, parameters=parameters)


if __name__ == "__main__":
    main()