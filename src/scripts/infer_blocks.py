import argparse

import numpy as np
import torch

from src.utils import infer


def main():
    parser = argparse.ArgumentParser(
        description='Tool optimize data hiding methods with orthogonal moments')
    parser.add_argument("data", metavar="data", help="data to infer")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="File parameters values will be saved")
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="inference model"
    )

    args = parser.parse_args()

    data = np.loadtxt(args.data)
    model = torch.load(args.model)

    infer.lineal_inference(data, model, args.output)


if __name__ == "__main__":
    main()