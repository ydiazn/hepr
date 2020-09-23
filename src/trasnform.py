from scipy import fftpack


class DCT:
    def __init__(self, norm='ortho'):
        self.norm = norm

    def direct(self, cover_work):
        return fftpack.dct(cover_work, norm=self.norm)

    def inverse(self, ws_work):
        return fftpack.idct(ws_work, norm=self.norm)
