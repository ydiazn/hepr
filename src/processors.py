import numpy as np

from almiky.metrics import imperceptibility, robustness
from almiky.utils import utils


def ber_mean(extract, ws_work, watermark, attacks):
    return np.mean(
        [
            robustness.ber(
                extract(attack(ws_work)),
                watermark
            )
            for attack in attacks
        ]
    )


def pnsr_ber_index(
        hider_factory, cover_work, data,
        index, step, *args, attacks=None, reference_psnr=44, **kwargs):

    def get_extractor(extractor, index):
        def wrapper(ws_work):
            return extractor(ws_work, index=index)

        return wrapper

    index = int(round(index))
    step = int(round(step))

    data = utils.char2bin(data)
    hider = hider_factory.build(step, *args, **kwargs)
    ws_work = hider.insert(cover_work, data, index=index)
    psnr = imperceptibility.psnr(cover_work, ws_work)

    watermark = hider.extract(ws_work, index=index)
    if attacks:
        extract = get_extractor(hider.extract, index)
        ber = ber_mean(extract, ws_work, watermark, attacks)
    else:
        ber = robustness.ber(watermark, data)

    return psnr, ber


def pnsr_ber_index_scaled(
        hider_factory, cover_work, data,
        index, step, *args, **kwargs):

    psnr, ber = pnsr_ber_index(
        hider_factory, cover_work, data,
        index, step, *args, **kwargs)

    return 1 - psnr, ber


def binary_pnsr_ber_index_scaled(
        hider_factory, cover_work, data,
        *particle, **kwargs):

    particle = np.array(list(particle))
    step = particle[:7]
    index = particle[7:]
    index = index.dot(2**np.arange(index.size)[::-1])
    step = step.dot(2**np.arange(step.size)[::-1])

    if step == 0:
        step = 1

    scale = utils.max_psnr(cover_work.shape)
    psnr, ber = pnsr_ber_index(
        hider_factory, cover_work, data,
        index, step, scale=scale, **kwargs)

    return 1 - psnr, ber
