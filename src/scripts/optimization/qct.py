import json

import fire

from src.factories import QCharlierThebichefBitHiderFactory
from src.optimization.functions import weighted_agregation
from src.processors import pnsr_ber_index_scaled
from src.utils.process import generic


def main(indir, output, data, config, **kwargs):
    with open(config, 'r') as file:
        config = json.loads(file.read())

    with open(data, 'r') as file:
        data = file.read()
        file.close()

    generic(
        indir, config, output, data, weighted_agregation,
        processor=pnsr_ber_index_scaled,
        hider_factory=QCharlierThebichefBitHiderFactory, **kwargs
    )


if __name__ == "__main__":
    fire.Fire(main)
