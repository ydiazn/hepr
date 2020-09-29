import fire
import numpy as np

from src.factories import QCharlierThebichefBitHiderFactory
from src.processors import pnsr_ber_index
from src.utils.process import performance


def main(indir, output, data, parameters):
    swarm = np.loadtxt(parameters, usecols=(2, 3))

    with open(data, 'r') as file:
        data = file.read()
        file.close()

    performance(
        indir, output, data, swarm, processor=pnsr_ber_index,
        hider_factory=QCharlierThebichefBitHiderFactory,
    )


if __name__ == "__main__":
    fire.Fire(main)
