from almiky.metrics import imperceptibility, robustness
from almiky.utils import utils


def pnsr_ber_index(hider_factory, cover_work, data, index, *args, **kwargs):
    hider = hider_factory.build(*args, **kwargs)

    ws_work = hider.insert(cover_work, data, index=index)
    extracted = hider.extract(ws_work, index=index)

    scale = utils.max_psnr(cover_work.shape)
    psnr = 1 - (imperceptibility.psnr(cover_work, ws_work) / scale)
    ber = robustness.ber(extracted, data)

    return psnr, ber
