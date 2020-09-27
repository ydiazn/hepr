from almiky.attacks import noise
from almiky.metrics import imperceptibility, robustness
from almiky.utils import utils


def pnsr_ber_index(
        hider_factory, cover_work, data,
        index, step, *args, scale=1, **kwargs):
    index = round(index)
    step = round(step)
    data = utils.char2bin(data)
    hider = hider_factory.build(step, *args, **kwargs)

    ws_work = hider.insert(cover_work, data, index=index)
    psnr = imperceptibility.psnr(cover_work, ws_work) / scale

    ws_work = noise.salt_paper_noise(ws_work, 0.01)
    extracted = hider.extract(ws_work, index=index)
    ber = robustness.ber(extracted, data)

    return psnr, ber


def pnsr_ber_index_scaled(
        hider_factory, cover_work, data,
        index, step, *args, **kwargs):

    scale = utils.max_psnr(cover_work.shape)
    psnr, ber = pnsr_ber_index(
        hider_factory, cover_work, data,
        index, step, *args, scale=scale, **kwargs)

    return 1 - psnr, ber
