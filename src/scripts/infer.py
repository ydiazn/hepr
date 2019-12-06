import argparse
import json
import sys

import torch

from src.utils import infer


def main():
    parser = argparse.ArgumentParser(
        description='Tool optimize data hiding methods with orthogonal moments')
    parser.add_argument("indir", metavar="indir", help="Images directory")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="File where parameters values will be saved")
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Network model file"
    )
    args = parser.parse_args()

    model = torch.load(args.model)

    infer.convolutional_inference(
        indir=args.indir, file=args.output, model=model)


if __name__ == "__main__":
    main()