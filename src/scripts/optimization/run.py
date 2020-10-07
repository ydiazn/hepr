import json
from pathlib import Path

import fire

from src.optimization import utils
from src.processors import pnsr_ber_index
from src.utils.process import generic


def main(*args, config, output, data, **kwargs):
    with open(config, 'r') as file:
        config = json.loads(file.read())

    data_file = str(Path(config['data']).joinpath(data))
    with open(data_file, 'r') as file:
        data = file.read()
        file.close()

    indir = config['indir']
    output = str(Path(config['output']).joinpath(output)) 
    factory = config['hider']
    function = config['objective']
    attacks = config['attacks']
    factory = utils.hider_factories[factory]
    if function == "dwa":
        kwargs.update({'counter': True})

    if attacks:
        attacks = [
            utils.get_attack(
                utils.attack_callables[attack['name']],
                **attack['parameters']
            )
            for attack in attacks
        ]

    generic(
        indir, config, output, data, utils.objective_functions[function],
        *args, processor=pnsr_ber_index, hider_factory=factory,
        attacks=attacks, reference_psnr=config['reference_psnr'], **kwargs
    )


if __name__ == "__main__":
    fire.Fire(main)
