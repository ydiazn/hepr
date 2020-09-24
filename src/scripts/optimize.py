import json

import fire

from src.utils import process


def main(indir, output, data, config):
    with open(config, 'r') as file:
        config = json.loads(file.read())

    with open(data, 'r') as file:
        data = file.read()
        file.close()

    process.qkrawtchouk8x8(
        indir=indir, config=config, output=output, data=data)


if __name__ == "__main__":
    fire.Fire(main)
