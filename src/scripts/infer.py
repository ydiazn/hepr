import argparse
import json
import sys

import torch

from src.utils import process


def main():
    parser = argparse.ArgumentParser(
        description='Tool optimize data hiding methods with orthogonal moments')
    parser.add_argument("indir", metavar="indir", help="Images directory")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="File where fitness will be saved")
    parser.add_argument(
        "-d",
        "--data",
        required=True,
        help="File with data to hide"
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Network model file"
    )
    args = parser.parse_args()

    with open(args.data, 'r') as file:
        data = file.read()

    model = torch.load(args.model)

    process.qkrawtchouk8x8_trained(
        indir=args.indir, file=args.output, data=data, model=model)


if __name__ == "__main__":
    main()