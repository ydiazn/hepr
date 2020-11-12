from almiky.embedding.dpc.qim import BinaryQuantizationIndexModulation
from almiky.hiders.base import SingleBitHider
from almiky.hiders.base import TransformHider
from almiky.hiders.block import BlockBitHider
from almiky.moments import matrix
from almiky.moments.matrix import ImageTransform, QKrawtchoukMatrix
from almiky.moments.matrix import TchebichefMatrix
from almiky.moments.matrix import SeparableTransform
from almiky.moments import transform
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
                ImageTransform(
                    SeparableTransform(
                        transform.QCHARLIER, transform.TCHEBICHEF))
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


class QCharlierThebichefBitHiderFactory2:

    @staticmethod
    def build(step, a, q, dimension=8):
        qc = QCharlierMatrix(dimension, a=a, q=q)
        th = TchebichefMatrix(dimension, N=dimension)
        transform = SeparableTransform(
                qc.get_values(), th.get_values()
        )
        return BlockBitHider(
            TransformHider(
                SingleBitHider(
                    ScanMapping(),
                    BinaryQuantizationIndexModulation(step)
                ),
                ImageTransform(transform)
            )
        )


class ThebichefBitHiderFactory:

    @staticmethod
    def build(step):
        return BlockBitHider(
            TransformHider(
                SingleBitHider(
                    ScanMapping(),
                    BinaryQuantizationIndexModulation(step)
                ),
                ImageTransform(
                    SeparableTransform(
                        transform.TCHEBICHEF, transform.TCHEBICHEF))
            )
        )


class DCTThebichefBitHiderFactory:

    @staticmethod
    def build(step):
        return BlockBitHider(
            TransformHider(
                SingleBitHider(
                    ScanMapping(),
                    BinaryQuantizationIndexModulation(step)
                ),
                ImageTransform(
                    SeparableTransform(
                        transform.DCT2, transform.TCHEBICHEF))
            )
        )


class ThebichefDCTBitHiderFactory:

    @staticmethod
    def build(step):
        return BlockBitHider(
            TransformHider(
                SingleBitHider(
                    ScanMapping(),
                    BinaryQuantizationIndexModulation(step)
                ),
                ImageTransform(
                    SeparableTransform(
                        transform.TCHEBICHEF, transform.DCT2))
            )
        )


class CharlierDCTBitHiderFactory:

    @staticmethod
    def build(step, alpha, dimension=8):
        CHARLIER = matrix.CharlierMatrix(dimension, alpha=alpha).get_values()

        return BlockBitHider(
            TransformHider(
                SingleBitHider(
                    ScanMapping(),
                    BinaryQuantizationIndexModulation(step)
                ),
                ImageTransform(
                    SeparableTransform(
                        CHARLIER, transform.DCT2))
            )
        )


class CharlierBitHiderFactory:

    @staticmethod
    def build(step, alpha, dimension=8):
        CHARLIER = matrix.CharlierMatrix(dimension, alpha=alpha).get_values()

        return BlockBitHider(
            TransformHider(
                SingleBitHider(
                    ScanMapping(),
                    BinaryQuantizationIndexModulation(step)
                ),
                ImageTransform(
                    SeparableTransform(CHARLIER, CHARLIER))
            )
        )


class CharlierSobolevBitHiderFactory:

    @staticmethod
    def build(step, alpha, beta, gamma, dimension=8):
        return BlockBitHider(
            TransformHider(
                SingleBitHider(
                    ScanMapping(),
                    BinaryQuantizationIndexModulation(step)
                ),
                ImageTransform(
                    matrix.CharlierSobolevMatrix(
                        dimension,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma
                    )
                )
            )
        )
