import json
from pathlib import Path

import fire
import numpy as np

from src.optimization import utils
from src.processors import pnsr_ber_index
from src.utils.process import performance


def main(config, output, data, **kwargs):

    with open(config, 'r') as file:
        config = json.loads(file.read())

    data_file = str(Path(config['data']).joinpath(data))
    with open(data_file, 'r') as file:
        data = file.read()
        file.close()

    output = str(Path(config['output']).joinpath(output))
    o_file = str(Path(config['poutput']).joinpath(output))
    swarm = np.loadtxt(p_file, usecols=config['parameters'])
    indir = config['indir']
    output = config['poutput']
    factory = config['hider']
    attacks = config['attacks']
    factory = utils.hider_factories[factory]

    if attacks:
        attacks = [
            utils.get_attack(
                utils.attack_callables[attack['name']],
                **attack['parameters']
            )
            for attack in attacks
        ]

    performance(
        indir, output, data, swarm, processor=pnsr_ber_index,
        hider_factory=factory, attacks=attacks
    )


if __name__ == "__main__":
    fire.Fire(main)
