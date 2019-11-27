from pathlib import Path

import imageio
import numpy as np
import pyswarms as ps

from src.optimization import functions as fx
from src.optimization import hide
from src.utils.reduction import average_first_eight_coeficients

def qkrawtchouk8x8(indir, config, output, data):

    def calculate(image):
        cover_work = imageio.imread(str(image))
        # First eight coeficient averaging
        coeficients = average_first_eight_coeficients(cover_work[:, :, 1], 8)

        kwargs = dict(cover_work=cover_work, data=data, get_ws_work=hide.qkrawtchouk8x8)
        cost, pos = optimizer.optimize(fx.psnr, config['iterations'], config['n_processes'], **kwargs)
        return (image.name, *coeficients, -cost, *pos)

    # Create bounds
    max_bound = config['optimizer']['bounds']['min']
    min_bound = config['optimizer']['bounds']['max']
    bounds = (min_bound, max_bound)

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(
        n_particles=config['optimizer']['n_particle'],
        dimensions=config['optimizer']['dimensions'],
        options=config['optimizer']['options'],
        bounds=bounds
    )
    # Perform optimization
    indir = Path(indir)
    output = Path(output)
    results = [calculate(image) for image in indir.iterdir()]
    np.savetxt(str(output), results, fmt='%s')
