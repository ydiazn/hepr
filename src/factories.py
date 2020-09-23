from almiky.embedding.dpc.qim import BinaryQuantizationIndexModulation
from almiky.hiders.base import SingleBitHider
from almiky.hiders.base import TransformHider
from almiky.hiders.block import BlockBitHider
from almiky.moments.matrix import QKrawtchoukMatrix
from almiky.utils.scan.scan import ScanMapping

from src.trasnform import DCT


class QKrawtchoukBitHiderFactory:
    def build(self, p, q, step, dimensions=(8, 8)):
        return BlockBitHider(
            TransformHider(
                SingleBitHider(
                    ScanMapping(),
                    BinaryQuantizationIndexModulation(step)
                ),
                QKrawtchoukMatrix(dimensions, p=p, q=q)
            )
        )


class DCTBitHiderFactory:
    def build(self, step):
        return BlockBitHider(
            TransformHider(
                SingleBitHider(
                    ScanMapping(),
                    BinaryQuantizationIndexModulation(step)
                ),
                DCT()
            )
        )
