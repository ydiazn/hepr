import numpy as np

from almiky.metrics import imperceptibility, robustness
from almiky.utils import utils


def mean_ber(extract, ws_work, watermark, attacks):
    return np.mean(
        (
            robustness.ber(
                extract(attack(ws_work)),
                watermark
            )
            for attack in attacks
        )
    )


def pnsr_ber_index(
        hider_factory, cover_work, data,
        index, step, *args, attacks=None, scale=1, **kwargs):

    def get_extractor(extractor, index):
        def wrapper(ws_work):
            return extractor(ws_work, index=index)

        return wrapper

    index = round(index)
    step = round(step)
    data = utils.char2bin(data)
    hider = hider_factory.build(step, *args, **kwargs)

    ws_work = hider.insert(cover_work, data, index=index)
    psnr = imperceptibility.psnr(cover_work, ws_work) / scale

    watermark = hider.extract(ws_work, index=index)
    if attacks:
        extract = get_extractor(hider.extract, index)
        ber = mean_ber(extract, ws_work, watermark, attacks)
    else:
        ber = robustness.ber(watermark, data)

    return psnr, ber


def pnsr_ber_index_scaled(
        hider_factory, cover_work, data,
        index, step, *args, **kwargs):

    scale = utils.max_psnr(cover_work.shape)
    psnr, ber = pnsr_ber_index(
        hider_factory, cover_work, data,
        index, step, *args, scale=scale, **kwargs)

    return 1 - psnr, ber
