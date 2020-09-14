import logging

import pyswarms as ps


logging.basicConfig(level=logging.INFO)


options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}
optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=1, options=options, bounds=([-100], [100]))

def parabole(x):
    return (x ** 2).reshape(-1)


def inverse_parabole(x):
    return -1 * ((-1 * (x ** 2)).reshape(-1))


for i in range(100):
    optimizer.optimize(inverse_parabole, iters=50)