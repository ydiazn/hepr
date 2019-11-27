from pathlib import Path

import imageio
import numpy as np
import pyswarms as ps
from almiky.moments.matrix import Transform
from almiky.utils.ortho_matrix import dct
from almiky.hidders.frequency import HidderEightFrequencyCoeficients
from almiky.metrics.metrics import psnr

from src.optimization import functions as fx
from src.optimization import hide

# Test data
base = Path(__file__).parent.parent
image = base.joinpath('images/000003.png')
cover_work = imageio.imread(image)
data = 'asdl aksdj lasjd asdlkja sdldsfk skdf sdkfj'
# Create bounds
max_bound = np.array([100, 1])
min_bound = np.array([0, 0])
bounds = (min_bound, max_bound)

# Set up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(
    n_particles=10, dimensions=2, options=options, bounds=bounds)
# Perform optimization
kwargs = dict(cover_work=cover_work, data=data, get_ws_work=hide.qkrawtchouk8x8)
cost, pos = optimizer.optimize(fx.psnr, 20, **kwargs)

hider = HidderEightFrequencyCoeficients(Transform(dct))
ws_work = hider.insert(cover_work, data)
print('PSNR with DCT: {}'.format(psnr(cover_work, ws_work)))