from almiky.embedding.dpc.qim import BinaryQuantizationIndexModulation
from almiky.hiders.base import SingleBitHider
from almiky.hiders.base import TransformHider
from almiky.utils.scan.maps import ROW_MAJOR_8x8
from almiky.moments.matrix import QKrawtchoukMatrix

from src import factories

class QKrawtchoukHiderFactoryTest(unittest.TestCase):
    def test_insert(self, insert_mock):
        cover_work = np.random.rand(8, 8)
        particle = np.random.rand(2)
        data = 'hola'

        hider = factories.QKrawtchoukHiderFactory(p=5, q=6)

        ws_work = hider.hide(cover_work, data)

        np.testing.assert_array_equal(cover_work, ws_work)