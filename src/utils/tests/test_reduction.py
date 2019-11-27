'''
Test for optimization functions
'''
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np

from src.utils.reduction import average_first_eight_coeficients


class BlockAveragingTest(unittest.TestCase):
    def test_block_averaging(self):
        blocks = np.array([
            [
                i + 1 for i in range(8)
            ] for _ in range(8)
        ])

        coficients = average_first_eight_coeficients(blocks, size=2)
        np.testing.assert_array_equal(
            coficients,
            [5, 4]
        )
