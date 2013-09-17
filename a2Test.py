import a2
import imageIO as io
import numpy
import random
import math
import unittest


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.im = io.imread()

    def test_0_imageLoad(self):
        self.assertEqual(self.im.shape, (85, 128, 3), "Size of input image is wrong. Have you modified in.png by accident?")

    # def test_1_scale(self):
    #     io.imwrite(a2.scaleNN(io.imread('panda2.png'), 3.2), 'out_scale.png')

    # def test_2_lin(self):
    #     io.imwrite(a2.scaleLin(io.imread('in.png'), 3), 'out_lin.png')

    # def test_3_rotate(self):
    #     io.imwrite(a2.rotate(io.imread('panda2.png'), math.pi / 4), 'out_rotate_45.png')
    #     io.imwrite(a2.rotate(io.imread('panda2.png'), math.pi), 'out_rotate_180.png')

    def test_3_warp1(self):
        io.imwrite(a2.warpBy1(io.imread('bear.png'), a2.segment(0,0,10,0), a2.segment(10,10,30,15)), 'warp1.png')

        
suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
