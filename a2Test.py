import a2
import imageIO
import numpy
import random
import unittest


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.im = imageIO.imread()

    def test_0_imageLoad(self):
        self.assertEqual(self.im.shape, (85, 128, 3), "Size of input image is wrong. Have you modified in.png by accident?")

    def test_1_prt(self):
        a2.scaleNN(self.im, 2)

        
suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
