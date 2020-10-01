import json

import fire
from almiky.attacks.noise import salt_paper_noise

from src.factories import DCTBitHiderFactory
from src.optimization.functions import weighted_agregation
from src.processors import pnsr_ber_index_scaled
from src.utils.process import generic


def get_attack(density):
    def wrapper(ws_work):
        return salt_paper_noise(ws_work, density)

    return wrapper


def main(indir, output, data, config, density=0.01, **kwargs):
    with open(config, 'r') as file:
        config = json.loads(file.read())

    with open(data, 'r') as file:
        data = file.read()
        file.close()

    attack = get_attack(density)

    generic(
        indir, config, output, data, weighted_agregation,
        processor=pnsr_ber_index_scaled, hider_factory=DCTBitHiderFactory,
        attacks=(attack,), **kwargs
    )


if __name__ == "__main__":
    fire.Fire(main)
