'''
Single objective functions

All objective functions must accept a ̈́(numpy.ndarray) with shape
(n_particles, dimensions). Each row represents a  particle, and
each column represents its position on a specific dimension of
the search-space.

functions must return an array of size (n_particles, )
that contains all the computed fitness for each particle.
'''
import logging
import math

from almiky.metrics import imperceptibility
import numpy as np


logging.basicConfig(level=logging.INFO)


def psnr(swarm, cover_work, data, get_ws_work):
    '''
    psnr(swarm, block, msg) => np.array: return swarm fitness

    fitness is related to PSNR metric between cover work an watermarked
    (stego) work wich is obtained with (get_ws_work) callback.
    This function recive a swarm particle, cover work and data
    '''
    fitness = np.empty(swarm.shape[0])
    for i, particle in enumerate(swarm):
        ws_work = (particle, cover_work, data)
        try:
            psnr = imperceptibility.psnr(cover_work, ws_work)
        except:
            psnr = 0
        finally:
            fitness[i] = -psnr
    return fitness


def generic(swarm, caller, *args, **kwargs):
    fitness = [caller(particle, *args, **kwargs) for particle in swarm]

    return np.array(fitness)


def weighted_agregation(
        swarm, cover_work, data, processor,
        hider_factory, max_psnr, w1=0.5, **kwargs):
    '''
    Calculate and return weighted agregation between psnr and ber.

    Arguments:
    swarm -- np.array: swarm
    cover_work -- np.array: cover work
    data -- str: binary data to hide
    processor -- calable: calculate and return psnr and ber
    w1 -- double: pnsr weight; must be a value between 0 and 1
    w2 -- double: ber weight; must be a equal to (1 - w1)
    '''

    def agregation(fx, w1, w2):
        psnr, ber = fx
        a = max_psnr - 44
        b = a if psnr > 44 else 44
        psnr = abs(psnr - 44) / b

        return w1 * psnr + w2 * ber

    w2 = 1 - w1

    fitness = map(
        lambda fx: agregation(fx, w1, w2),
        (
            processor(hider_factory, cover_work, data, *particle, **kwargs)
            for particle in swarm
        )
    )

    return np.array(list(fitness))


def dynamic_weighted_agregation(
        swarm, cover_work, data, processor,
        hider_factory, alpha, get_iteration, **kwargs):
    '''
    calculate and return weighted agregation between psnr and ber.

    Arguments:
    swarm -- np.array: swarm
    cover_work -- np.array: cover work
    data -- str: binary data to hide
    processor -- calable: calculate and return psnr and ber
    get_iteration -- calable: return iteration of optimization algorithm
    alpha -- double: adaptation frequency
    '''
    t = get_iteration()
    w1 = abs(math.sin(2 * math.pi * t / alpha))
    logging.info('weigth: {}'.format(w1))

    return weighted_agregation(
        swarm, cover_work, data, processor, hider_factory, w1=w1)
