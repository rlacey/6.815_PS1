import a2
import imageIO as io
import numpy as np
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
    #     io.imwrite(a2.scaleLin(io.imread('panda2.png'), 3.5), 'out_lin.png')

    # def test_3_rotate(self):
    #     io.imwrite(a2.rotate(io.imread('panda2.png'), math.pi / 4), 'out_rotate_45.png')
    #     io.imwrite(a2.rotate(io.imread('panda2.png'), math.pi), 'out_rotate_180.png')

    # def test_4_warp1(self):
    #     io.imwrite(a2.warpBy1(io.imread('bear.png'), a2.segment(0,0,10,0), a2.segment(10,10,30,15)), 'warp1.png')

    # def test_5_warp(self):
    #     # io.imwrite(a2.warp(io.imread('bear.png'), [a2.segment(0,0,10,0)], [a2.segment(10,10,30,15)]), 'warp.png')
    #     image = io.imread("bear.png")
    #     segmentsBefore=np.array([a2.segment(13, 10, 50, 10), a2.segment(119, 72, 119, 28)])
    #     segmentsAfter=np.array([a2.segment(8, 12, 38, 34), a2.segment(117, 71, 91, 36)])
    #     io.imwrite(a2.warp(image, segmentsBefore, segmentsAfter), "bearWarp2.png")

    def test_6_morph(self):
    	segmentsBefore = np.array([a2.segment(87,131,109,129), a2.segment(142,126,165,129)])
    	segmentsAfter = np.array([a2.segment(81,112,107,107), a2.segment(140,102,163,101)])
        images = a2.morph(io.imread('fredo2.png'), io.imread('werewolf.png'), segmentsBefore, segmentsAfter)
        io.imwrite(images[0], 'morph1.png')
        io.imwrite(images[1], 'morph2.png')
        io.imwrite(images[2], 'morph3.png')

        
suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
