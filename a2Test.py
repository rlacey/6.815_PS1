import a2
import imageIO as io
import numpy as np
import math
import unittest


class TestSequenceFunctions(unittest.TestCase):

    def setUp(self):
        self.im = io.imread()

    def test_0_imageLoad(self):
        self.assertEqual(self.im.shape, (85, 128, 3), "Size of input image is wrong. Have you modified in.png by accident?")

    def test_1_scale(self):
        io.imwrite(a2.scaleNN(io.imread('panda2.png'), 3.5), 'scaleNN-3_5.png')

    def test_2_interpolate_scale(self):
        io.imwrite(a2.scaleLin(io.imread('panda2.png'), 3.5), 'interpolate-scale-3_5.png')

    def test_3_rotate(self):
        io.imwrite(a2.rotate(io.imread('panda2.png'), math.pi/4), 'rotate_45.png')
        io.imwrite(a2.rotate(io.imread('panda2.png'), math.pi), 'rotate_180.png')

    def test_4_warp1(self):
        io.imwrite(a2.warpBy1(io.imread('bear.png'), a2.segment(0,0,10,0), a2.segment(10,10,30,15)), 'warp1.png')

    def test_5_warp(self):
        segmentsBefore=np.array([a2.segment(89, 130, 106, 125), a2.segment(150, 125, 164, 129), a2.segment(98, 199, 132, 202)])
        segmentsAfter=np.array([a2.segment(42, 109, 106, 108), a2.segment(131, 106, 179, 104), a2.segment(44, 180, 140, 180)])
        io.imwrite(a2.warp(io.imread("fredo2.png"), segmentsBefore, segmentsAfter), "warp.png")

    def test_6_morph(self):
        segmentsBefore = np.array([a2.segment(87,131,109,129), a2.segment(142,126,165,129)])
        segmentsAfter = np.array([a2.segment(81,112,107,107), a2.segment(140,102,163,101)])
        images = a2.morph(io.imread('fredo2.png'), io.imread('werewolf.png'), segmentsBefore, segmentsAfter, 7)
        for img in images:
            io.imwriteSeq(img)
        
suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)
