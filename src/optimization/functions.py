'''
Single objective functions

All objective functions must accept a ̈́(numpy.ndarray) with shape
(n_particles, dimensions). Each row represents a  particle, and
each column represents its position on a specific dimension of
the search-space.

functions must return an array of size (n_particles, )
that contains all the computed fitness for each particle.
'''

import numpy as np

from almiky.metrics import metrics

def psnr(swarm, cover_work, data, get_ws_work):
    '''
    psnr(swarm, block, msg) => np.array: return swarm fitness

    fitness is related to PSNR metric between cover work an watermarked
    (stego) work wich is obtained with (get_ws_work) callback. 
    This function recive a swarm particle, cover work and data
    '''
    fitness = np.empty(swarm.shape[0])
    for i, particle in enumerate(swarm):
        try:
            ws_work = get_ws_work(particle, cover_work, data)
            psnr = metrics.psnr(cover_work, ws_work)
        except:
            psnr = 0
        finally:
            fitness[i] = -psnr
    return fitness
