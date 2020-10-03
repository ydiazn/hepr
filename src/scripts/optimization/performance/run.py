import fire
import numpy as np

from src.optimization import utils
from src.processors import pnsr_ber_index
from src.utils.process import performance


def main(
        indir, output, data, parameters, factory,
        attacks=None, **kwargs):
    swarm = np.loadtxt(parameters, usecols=(2, 3))

    with open(data, 'r') as file:
        data = file.read()
        file.close()

    factory = utils.hider_factories[factory]

    if attacks:
        attacks = [
            utils.get_attack(
                utils.attack_callables[attack['name']],
                *attack['args'],
                **attack['kwargs']
            )
            for attack in attacks
        ]

    performance(
        indir, output, data, swarm, processor=pnsr_ber_index,
        hider_factory=factory, attacks=attacks
    )


if __name__ == "__main__":
    fire.Fire(main)
