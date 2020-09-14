'''
Test for optimization functions
'''
import numpy as np
import unittest
from unittest.mock import Mock, patch

from almiky.metrics import imperceptibility
from src.optimization import functions


def psnr_mock(cover_work, ws_work):
    return 0


@unittest.expectedFailure
class TestSingleObjetive(unittest.TestCase):

    @patch.object(imperceptibility, 'psnr', side_effect=psnr_mock)
    def test_psnr(self, psnr_mock):
        swarm = np.empty((5, 2))
        cover_work = np.empty((8, 8))
        data = 'hola'
        fitness = functions.psnr(
            swarm, cover_work, data, lambda p, c, d: cover_work)

        np.testing.assert_array_equal(fitness, np.zeros(5))


class GenericObjectiveFunctionTest(unittest.TestCase):

    def test_fitness(self):
        expected_fitness = [4, 6]
        caller = Mock(side_effect=expected_fitness)
        swarm = np.array([
            [0.29358913, 0.46025107],
            [0.92129178, 0.17637063]
        ])

        fitness = functions.generic(swarm, caller)

        self.assertEqual(caller.call_count, 2)
        np.testing.assert_array_equal(fitness, expected_fitness)


if __name__ == '__main__':
    unittest.main()
