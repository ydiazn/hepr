'''
Test for optimization functions
'''
import unittest
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
from almiky.moments.matrix import QKrawtchoukMatrix
from almiky.hidders.frequency import HidderEightFrequencyCoeficients

from src.hidders import hidders as hide


def insert_mock(cover_work, data):
    return cover_work

class QKrawtchoukTest(unittest.TestCase):

    @patch.object(HidderEightFrequencyCoeficients, 'insert', side_effect=insert_mock)
    def test_hidder(self, insert_mock):
        cover_work = np.random.rand(8, 8)
        particle = np.random.rand(2)
        data = 'hola'

        ws_work = hide.qkrawtchouk8x8(particle, cover_work, data)
        np.testing.assert_array_equal(cover_work, ws_work)


if __name__ == '__main__':
    unittest.main()