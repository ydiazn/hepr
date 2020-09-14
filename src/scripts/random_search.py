'''
Hide a message into a cover work using QKrawtchoukMatrix
based and DCT transform and compare PSNR values

Usage 
python evaluate --image=path_to_image --data=path_to_message --p=NUMBER --q=NUMBER
'''

import imageio
import fire
import logging

from almiky.hidders import frequency
from almiky.metrics import metrics
from almiky.moments.matrix import Transform, QKrawtchoukMatrix
from almiky.utils.ortho_matrix import dct
from pyswarms.utils.search.random_search import RandomSearch
from pyswarms.single import GlobalBestPSO


logging.basicConfig(level=logging.INFO)


def psnr(cover_work, message, p, q):
    transform = QKrawtchoukMatrix(8, p=p, q=q)
    hider = frequency.HidderEightFrequencyCoeficients(transform)
    stego_work = hider.insert(cover_work, message)
    psnr = metrics.psnr(cover_work, stego_work)

    return psnr

def dct_insert(cover_work, message):
    transform = Transform(dct)
    hider = frequency.HidderEightFrequencyCoeficients(transform)
    try:
        stego_work = hider.insert(cover_work, message)
        psnr = metrics.psnr(cover_work, stego_work)
    except:
        psnr = 0

    return psnr

    return psnr

def qKrawtchouk_hide(x, cover_work, message):
    return [
        -1 * psnr(cover_work, message, particle[0], particle[1])
        for particle in x
    ]


def main(image, data):
    '''
    Parametes:
    image: path to image file
    data: path to file that contain the message
    p: QKrawtchoukMatrix p parameter
    q: QKrawtchoukMatrix q parameter
    '''
    with open(data, 'r') as file:
        message = file.read()
        file.close()

    cover_work = imageio.imread(image)

    options = {
        'c1': [0.1, 3],
        'c2': [0.1, 3],
        'w': [0.6, 2],
        'k': [0, 10],
        'p': 1
    }

    g = RandomSearch(
        GlobalBestPSO,
        n_particles=20,
        dimensions=2,
        options=options,
        objective_func=lambda x: qKrawtchouk_hide(x, cover_work, message),
        iters=10,
        n_selection_iters=10,
        bounds=([0, 0], [1, 1])
    )

    best_score, best_options = g.search()
    dct_score = dct_insert(cover_work, message)
    logging.info('Score: {}'.format(best_score))
    logging.info('DCT score {}'.format(dct_score))
    logging.info(best_options)


if __name__ == "__main__":
    fire.Fire(main)
