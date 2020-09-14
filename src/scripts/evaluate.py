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


logging.basicConfig(level=logging.INFO)


def main(image, data, p, q):
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

    transform = QKrawtchoukMatrix(8, p=p, q=q)
    hider = frequency.HidderEightFrequencyCoeficients(transform)
    stego_work = hider.insert(cover_work, message)
    psnr = metrics.psnr(cover_work, stego_work)
    logging.info('QKrawtchouk: {}'.format(psnr))

    transform = Transform(dct)
    hider = frequency.HidderEightFrequencyCoeficients(transform)
    stego_work = hider.insert(cover_work, message)
    psnr = metrics.psnr(cover_work, stego_work)
    logging.info('DCT: {}'.format(psnr))


if __name__ == "__main__":
    fire.Fire(main)
