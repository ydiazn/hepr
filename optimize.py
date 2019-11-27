import argparse
import json
import sys


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
        "-c",
        "--config",
        required=True,
        help="Config file with optimization and data hidding parameters"
    )
    args = parser.parse_args()


if __name__ == "__main__":
    main()