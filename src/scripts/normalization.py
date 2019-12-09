import argparse
import json
import sys

import torch
from torch.utils.data import DataLoader
import numpy as np

from src.datasets.head_mri import  HeadMRIDataset


def main():
    parser = argparse.ArgumentParser(
        description='Tool optimize data hiding methods with orthogonal moments')
    parser.add_argument("indir", metavar="indir", help="Images directory")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="File where values will be saved")

    args = parser.parse_args()

    dataset = HeadMRIDataset(args.indir)
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=10,
    )


    mean = 0.
    std = 0.
    nb_samples = 0.
    for data in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1).float()
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print("Mean: {}".format(mean))
    print("Std: {}".format(std))

    mean = mean.numpy()
    std = std.numpy()

    result = np.array([mean, std])

    np.savetxt(args.output, result)


if __name__ == "__main__":
    main()