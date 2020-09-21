'''
Single objective functions

All objective functions must accept a ̈́(numpy.ndarray) with shape
(n_particles, dimensions). Each row represents a  particle, and
each column represents its position on a specific dimension of
the search-space.

functions must return an array of size (n_particles, )
that contains all the computed fitness for each particle.
'''
import math

import numpy as np

from almiky.attacks.noise import salt_paper_noise
from almiky.embedding.dpc.qim import BinaryQuantizationIndexModulation
from almiky.exceptions import NotMatrixQuasiOrthogonal
from almiky.hiders.base import SingleBitHider
from almiky.hiders.base import TransformHider
from almiky.hiders.block import BlockBitHider
from almiky.metrics import imperceptibility
from almiky.metrics import robustness
from almiky.moments.matrix import dct, QKrawtchoukMatrix, Transform
from almiky.utils.scan.scan import ScanMapping
from almiky.utils.utils import max_psnr


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


def psnr_ber_qkrawtchouk(swarm, cover_work, data):

    fitness = []
    scan = ScanMapping()

    for index, step, p, q in swarm:
        try:
            transform = QKrawtchoukMatrix((8, 8), p=p, q=q)
            hider = BlockBitHider(
                TransformHider(
                    SingleBitHider(
                        scan,
                        BinaryQuantizationIndexModulation(step)
                    ),
                    transform
                )
            )

            ws_work = hider.insert(cover_work, data, index=index)
            extracted = hider.extract(ws_work, index=index)

            psnr = imperceptibility.psnr(cover_work, ws_work)
            ber = robustness.ber(extracted, data)
            scale = max_psnr(cover_work.shape)
            performance = 1 - psnr / scale + ber

        except NotMatrixQuasiOrthogonal:
            performance = 1

        fitness.append(performance)

    return np.array(fitness)


def psnr_ber_dct(swarm, cover_work, data):
    fitness = []
    scan = ScanMapping()

    for index, step in swarm:
        hider = BlockBitHider(
            TransformHider(
                SingleBitHider(
                    scan,
                    BinaryQuantizationIndexModulation(step)
                ),
                Transform(dct)
            )
        )

        ws_work = hider.insert(cover_work, data, index=index)
        extracted = hider.extract(ws_work, index=index)

        psnr = imperceptibility.psnr(cover_work, ws_work)
        ber = robustness.ber(extracted, data)
        scale = max_psnr(cover_work.shape)

        performance = 1 - psnr / scale + ber
        fitness.append(performance)

    return np.array(fitness)