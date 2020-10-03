import json

import fire

from src.optimization import utils
from src.processors import binary_pnsr_ber_index_scaled
from src.utils.process import binary_generic


def main(
        indir, output, data, config, factory,
        function="cwa", attacks=None, **kwargs):
    with open(config, 'r') as file:
        config = json.loads(file.read())

    with open(data, 'r') as file:
        data = file.read()
        file.close()

    args = []
    factory = utils.hider_factories[factory]
    if function == "dwa":
        kwargs.update({'counter': True})

    if attacks:
        attacks = [
            utils.get_attack(
                utils.attack_callables[attack['name']],
                *attack['args'],
                **attack['kwargs']
            )
            for attack in attacks
        ]

    binary_generic(
        indir, config, output, data, utils.objective_functions[function],
        *args, processor=binary_pnsr_ber_index_scaled, hider_factory=factory,
        attacks=attacks, **kwargs
    )


if __name__ == "__main__":
    fire.Fire(main)
