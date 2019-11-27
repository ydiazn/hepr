'''
Test for optimization functions
'''
import unittest
from unittest.mock import patch

import numpy as np
from almiky.metrics import metrics

from src.optimization import functions


def psnr_mock(cover_work, ws_work):
    return 0

class TestSingleObjetive(unittest.TestCase):

    @patch.object(metrics, 'psnr', side_effect=psnr_mock)
    def test_psnr(self, psnr_mock):
        swarm = np.empty((5, 2))
        cover_work = np.empty((8, 8))
        data = 'hola'
        fitness = functions.psnr(swarm, cover_work, data, lambda p, c, d: cover_work)
        np.testing.assert_array_equal(fitness, np.zeros(5))


if __name__ == '__main__':
    unittest.main()