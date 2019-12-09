from pathlib import Path

import imageio
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
from torch.autograd import Variable
from almiky.metrics import metrics
from src.utils import reduction


def convolutional_inference(indir, file, model, normalization):

    def calculate(image, preprocess):
        import torch
        from torch.autograd import Variable
        from src.nets.regression import RegressionNet

        model.eval()
        input = Image.open(image)
        input = preprocess(input)
        input = input.unsqueeze(0)
        output = model(input)
        p, q = output.detach().numpy()[0]

        return (p, q)

    indir = Path(indir)
    mean, std = normalization
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    results = [calculate(image, preprocess) for image in sorted(indir.iterdir())]
    np.savetxt(file, results, fmt='%s')


def lineal_inference(data, model, file):
    model.eval()
    data = Variable(torch.from_numpy(data).float())
    output = model(data)
    results = output.detach().numpy()
    np.savetxt(file, results, fmt='%s')


def average_per_block(indir, output):
    from src.utils import reduction

    def calculate(file):
        image = imageio.imread(file)[:,:,0]
        average = reduction.average_first_eight_coeficients(image, 8)
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
    np.savetxt(file, results, fmt='%s')
