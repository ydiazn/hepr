import argparse
import json
import sys

from src.utils import optimizer


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
        "-c",
        "--config",
        required=True,
        help="Config file with optimization and data hidding parameters"
    )
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = json.loads(file.read())

    with open(args.data, 'r') as file:
        data = file.read()

    optimizer.qkrawtchouk8x8(
        indir=args.indir, config=config, output=args.output, data=data)


if __name__ == "__main__":
    main()