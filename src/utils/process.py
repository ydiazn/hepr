from pathlib import Path

import imageio
import numpy as np
import pyswarms as ps
from almiky.metrics import metrics
from almiky.utils.blocks_class import BlocksImage

from src.optimization import functions as fx
from src.hidders import hidders as hide
from src.utils.reduction import average_first_eight_coeficients

def qkrawtchouk8x8(indir, config, output, data):

    def calculate(image):
        cover_work = imageio.imread(str(image))
        # First eight coeficient averaging

        kwargs = dict(cover_work=cover_work, data=data, get_ws_work=hide.qkrawtchouk8x8)
        cost, pos = optimizer.optimize(fx.psnr, config['iterations'], config['n_processes'], **kwargs)

        # Comparing with DCT
        ws_work = hide.dct8x8(cover_work, data)
        psnr = metrics.psnr(cover_work, ws_work)
        return (image.name, *pos, -cost, psnr)

    # Create bounds
    max_bound = config['optimizer']['bounds']['max']
    min_bound = config['optimizer']['bounds']['min']
    bounds = (min_bound, max_bound)

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(
        n_particles=config['optimizer']['n_particle'],
        dimensions=config['optimizer']['dimensions'],
        options=config['optimizer']['options'],
        bounds=bounds
    )
    # Perform optimization
    indir = Path(indir)
    output = Path(output)
    results = [calculate(image) for image in sorted(indir.iterdir())]
    np.savetxt(str(output), results, fmt='%s')


def qkrawtchouk8x8_per_block(indir, config, output, data):

    def calculate(cover_block):
        kwargs = dict(cover_work=cover_block, data=data, get_ws_work=hide.qkrawtchouk8x8)
        cost, pos = optimizer.optimize(fx.psnr, config['iterations'], config['n_processes'], **kwargs)
        coeficients = cover_block.reshape(-1).tolist()

        # Comparing with DCT
        ws_block = hide.dct8x8(cover_block, data)
        psnr = metrics.psnr(cover_block, ws_block)
        p, q = pos
        print("QKrawtchouk (p: {}, q: {}, psnr: {}), DCT: {}".format(p, q, -cost, psnr))
        return (*coeficients, *pos, -cost, psnr)

    # Create bounds
    max_bound = config['optimizer']['bounds']['min']
    min_bound = config['optimizer']['bounds']['max']
    bounds = (min_bound, max_bound)

    # Call instance of PSO
    optimizer = ps.single.GlobalBestPSO(
        n_particles=config['optimizer']['n_particle'],
        dimensions=config['optimizer']['dimensions'],
        options=config['optimizer']['options'],
        bounds=bounds
    )
    # Perform optimization
    indir = Path(indir)
    output = Path(output)
    results = []
    for image in indir.iterdir():
        cover_work = imageio.imread(str(image))
        block_manage = BlocksImage(cover_work)
        for i in range(block_manage.max_num_blocks()):
            cover_block = block_manage.get_block(i)
            results.append(calculate(cover_block))

    np.savetxt(str(output), results, fmt='%s')


def dct8x8(indir, output, data):

    def calculate(image):
        cover_work = imageio.imread(str(image))
        ws_work = hide.dct8x8(cover_work, data)
        psnr = metrics.psnr(cover_work, ws_work)
        return (image.name, psnr)

    indir = Path(indir)
    output = Path(output)
    results = [calculate(image) for image in indir.iterdir()]
    np.savetxt(str(output), results, fmt='%s')


def qkrawtchouk8x8_trained(indir, file, data, model):

    def calculate(image):
        import torch
        from torch.autograd import Variable
        from src.nets.regression import RegressionNet
        from src.hidders import hidders
        from almiky.exceptions import NotMatrixQuasiOrthogonal

        cover_work = imageio.imread(str(image))
        # First eight coeficient averaging
        coeficients = average_first_eight_coeficients(cover_work[:, :, 1], 8)

        model.eval()
        coeficients = torch.from_numpy(coeficients).float()
        imput = Variable(coeficients)
        output = model(imput)

        p, q = output.detach().numpy()
        try:
            ws_work = hidders.qkrawtchouk8x8((p, q), cover_work, data)
            psnr_qk = metrics.psnr(cover_work, ws_work)
        except NotMatrixQuasiOrthogonal:
            psnr_qk = 0

        ws_work = hide.dct8x8(cover_work, data)
        psnr_dct = metrics.psnr(cover_work, ws_work)
        return (psnr_qk, psnr_dct)

    indir = Path(indir)
    results = [calculate(image) for image in indir.iterdir()]
    np.savetxt(file, results, fmt='%s')
