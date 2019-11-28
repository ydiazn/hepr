import numpy as np
from almiky.utils.ortho_matrix import dct
from almiky.moments.matrix import Transform, QKrawtchoukMatrix
from almiky.hidders import frequency
from almiky.exceptions import NotMatrixQuasiOrthogonal

def ident(matrix):
    I = np.identity(matrix.shape[0])
    return sum(sum(abs(np.around(np.dot(matrix.T, matrix)))-I))

def qkrawtchouk8x8(particle, cover_work, data):
    p, q = particle
    kwargs = dict(p=p, q=q)
    transform = QKrawtchoukMatrix(8, **kwargs)
    if not(ident(transform.values) == 0 and ident(transform.values.T) == 0):
        raise NotMatrixQuasiOrthogonal
    hidder = frequency.HidderEightFrequencyCoeficients(transform)

    watermarked_array = hidder.insert(cover_work, data)
    return watermarked_array


def dct8x8(cover_work, data):
    transform = Transform(dct)
    hidder = frequency.HidderEightFrequencyCoeficients(transform)

    watermarked_array = hidder.insert(cover_work, data)
    return watermarked_array