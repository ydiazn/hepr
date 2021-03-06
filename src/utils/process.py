import logging
from pathlib import Path

from almiky.utils import utils
import imageio
import numpy as np
import pyswarms as ps


logging.basicConfig(level=logging.INFO)


'''def qkrawtchouk8x8(indir, config, output, data):

    # Create bounds
    max_bound = config['optimizer']['bounds']['max']
    min_bound = config['optimizer']['bounds']['min']
    bounds = (min_bound, max_bound)

    def calculate(image):
        cover_work = imageio.imread(str(image))
        # First eight coeficient averaging

        kwargs = dict(cover_work=cover_work, data=data, get_ws_work=hide.qkrawtchouk8x8)
        # Call instance of PSO
        optimizer = ps.single.GlobalBestPSO(
            n_particles=config['optimizer']['n_particle'],
            dimensions=config['optimizer']['dimensions'],
            options=config['optimizer']['options'],
            bounds=bounds
        )
        cost, pos = optimizer.optimize(fx.psnr, config['iterations'], **kwargs)
        logging.info('image: {}'.format(image.name))
        logging.info('psnr: {}'.format(-cost))
        logging.info('p: {}, q: {}'.format(pos[0], pos[1]))

        # Comparing with DCT
        ws_work = hide.dct8x8(cover_work, data)
        psnr = metrics.psnr(cover_work, ws_work)
        return (image.name, *pos, -cost, psnr)

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


def qkrawtchouk8x8_DCT(indir, file, data, parameters):

    def calculate(image, preprocess, parameters):
        import torch
        from torch.autograd import Variable
        from src.nets.regression import RegressionNet
        from src.hidders import hidders
        from almiky.exceptions import NotMatrixQuasiOrthogonal

        cover_work = imageio.imread(image)
        p, q = parameters
        try:
            ws_work = hidders.qkrawtchouk8x8((p, q), cover_work, data)
            psnr_qk = metrics.psnr(cover_work, ws_work)
        except (NotMatrixQuasiOrthogonal, ValueError):
            psnr_qk = 0

        ws_work = hide.dct8x8(cover_work, data)
        psnr_dct = metrics.psnr(cover_work, ws_work)
        return (psnr_qk, psnr_dct)

    indir = Path(indir)
    preprocess = transforms.Compose([transforms.ToTensor()])
    results = [
        calculate(image, preprocess, parameters[i])
        for i, image in enumerate(sorted(indir.iterdir()))
    ]
    np.savetxt(file, results, fmt='%s')


def average_per_block(indir, output):
    from src.utils import reduction

    def calculate(file):
        image = imageio.imread(file)[:,:,0]
        average = reduction.average_first_eight_pixels(image, 8)
        average /= np.linalg.norm(average)
        return average.tolist()

    results = [calculate(file) for file in sorted(Path(indir).iterdir())]
    np.savetxt(output, results)


def qkrawtchouk8x8_regression(indir, file, data, parameters):

    def calculate(image, parameters):
        from src.hidders import hidders
        from almiky.exceptions import NotMatrixQuasiOrthogonal

        cover_work = imageio.imread(image)[:,:,0]
        p, q = parameters

        try:
            ws_work = hidders.qkrawtchouk8x8((p, q), cover_work, data)
            psnr_qk = metrics.psnr(cover_work, ws_work)
        except (NotMatrixQuasiOrthogonal, ValueError):
            psnr_qk = 0

        ws_work = hide.dct8x8(cover_work, data)
        psnr_dct = metrics.psnr(cover_work, ws_work)
        return (p, q, psnr_qk, psnr_dct)

    indir = Path(indir)
    results = [calculate(image, parameters[i]) for i, image in enumerate(sorted(indir.iterdir()))]
    np.savetxt(file, results, fmt='%s')'''


def generic(indir, config, output, data, objective, *args, counter=False, **kwargs):
    '''

    Arguments:
    config -- dict: optimizer settings
    objective -- callable: objective function
    hider_factory: hider factory
    '''

    # Create bounds
    max_bound = config['optimization']['optimizer']['bounds']['max']
    min_bound = config['optimization']['optimizer']['bounds']['min']
    bounds = (min_bound, max_bound)

    def calculate(image):
        cover_work = imageio.imread(str(image))
        kwargs.update(
            {
                'cover_work': cover_work,
                'data': data,
                'max_psnr': utils.max_psnr(cover_work.shape),
                'args': args
            }
        )
        if counter:
            kwargs.update({'get_iteration': utils.get_counter()})

        # Call instance of PSO
        optimizer = ps.single.GlobalBestPSO(
            n_particles=config['optimization']['optimizer']['n_particle'],
            dimensions=config['optimization']['optimizer']['dimensions'],
            options=config['optimization']['optimizer']['options'],
            bounds=bounds
        )

        cost, pos = optimizer.optimize(
            objective, config['optimization']['iterations'], **kwargs)
        logging.info('image: {}'.format(image.name))
        logging.info('performance: {}'.format(cost))
        logging.info('particle: {}'.format(pos))

        index, step, *rest = pos
        index = int(round(index))
        step = int(round(step))

        return (image.name, cost, index, step, *rest)

    # Perform optimization
    indir = Path(indir)
    output = Path(output)
    results = [calculate(image) for image in sorted(indir.iterdir())]
    np.savetxt(str(output), results, fmt='%s')


def binary_generic(
        indir, config, output, data, objective, counter=False, **kwargs):
    '''

    Arguments:
    config -- dict: optimizer settings
    objective -- callable: objective function
    hider_factory: hider factory
    '''

    def calculate(image):
        cover_work = imageio.imread(str(image))
        kwargs.update(
            {
                'cover_work': cover_work[:, :, 1],
                'data': data
            }
        )
        if counter:
            kwargs.update({'get_iteration': utils.get_counter()})

        # Call instance of PSO
        optimizer = ps.discrete.BinaryPSO(
            n_particles=config['optimizer']['n_particle'],
            dimensions=config['optimizer']['dimensions'],
            options=config['optimizer']['options'],
        )
        cost, pos = optimizer.optimize(
            objective, config['iterations'], **kwargs)
        logging.info('image: {}'.format(image.name))
        logging.info('performance: {}'.format(cost))
        logging.info('particle: {}'.format(pos))

        step = pos[:7]
        index = pos[7:]
        index = index.dot(2**np.arange(index.size)[::-1])
        step = step.dot(2**np.arange(step.size)[::-1])

        return (image.name, cost, index, step)

    # Perform optimization
    indir = Path(indir)
    output = Path(output)
    results = [calculate(image) for image in sorted(indir.iterdir())]
    np.savetxt(str(output), results, fmt='%s')


def performance(
        indir, output, data, swarm,
        processor, hider_factory, *args, **kwargs):
    '''

    Arguments:
    config -- dict: optimizer settings
    swarm -- arguments optimized
    processor -- callable: get performace
    hider_factory: hider factory
    '''

    def calculate(image, particle):
        cover_work = imageio.imread(str(image))
        psnr, ber = processor(
            hider_factory, cover_work, data, *particle, *args, **kwargs)

        return image.name, psnr, ber

    # Perform optimization
    indir = Path(indir)
    output = Path(output)

    results = [
        calculate(image, particle)
        for image, particle
        in zip(sorted(indir.iterdir()), swarm)
    ]
    np.savetxt(str(output), results, fmt='%s')
