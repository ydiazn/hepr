from almiky.embedding.dpc.qim import BinaryQuantizationIndexModulation
from almiky.hiders.base import SingleBitHider
from almiky.hiders.base import TransformHider
from almiky.hiders.block import BlockBitHider
from almiky.moments.matrix import ImageTransform, QKrawtchoukMatrix
from almiky.moments.matrix import SeparableTransform
from almiky.moments.transform import QCHARLIER, TCHEBICHEF
from almiky.utils.scan.scan import ScanMapping

from src.trasnform import DCT


class QKrawtchoukBitHiderFactory:

    @staticmethod
    def build(step, p, q, dimensions=8):
        return BlockBitHider(
            TransformHider(
                SingleBitHider(
                    ScanMapping(),
                    BinaryQuantizationIndexModulation(step)
                ),
                ImageTransform(QKrawtchoukMatrix(dimensions, p=p, q=q))
            )
        )


class QCharlierThebichefBitHiderFactory:

    @staticmethod
    def build(step):
        return BlockBitHider(
            TransformHider(
                SingleBitHider(
                    ScanMapping(),
                    BinaryQuantizationIndexModulation(step)
                ),
                ImageTransform(SeparableTransform(QCHARLIER, TCHEBICHEF))
            )
        )


class DCTBitHiderFactory:

    @staticmethod
    def build(step):
        return BlockBitHider(
            TransformHider(
                SingleBitHider(
                    ScanMapping(),
                    BinaryQuantizationIndexModulation(step)
                ),
                ImageTransform(DCT())
            )
        )
