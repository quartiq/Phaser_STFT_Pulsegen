
import unittest
import numpy as np
import matplotlib.pyplot as plt
from migen import *
from misoc.interconnect.stream import Endpoint

from super_hbf import SuperHbfUS
from super_cic import SuperCicUS

class TestHbfUS(unittest.TestCase):
    """HBF upsampler tests"""

    def setUp(self):
        pass

    def run_hbf_ss_sim(self, x, in_samples):
        """hbf filter test simulation"""
        y = np.zeros(in_samples * 2).astype('int')

        def hbf_sim():
            yield self.hbf.inp.stb.eq(1)
            yield self.hbf.ss.eq(1)
            for i in range(in_samples):
                yield
                yield self.hbf.inp.data.eq(x[i])
                y[i*2] = yield self.hbf.out1.data
                y[(i*2)+1] = yield self.hbf.out2.data

        run_simulation(self.hbf, hbf_sim(), vcd_name="imp_resp.vcd")
        return y

    def run_hbf_norm_sim(self, x, in_samples):
        """hbf filter test simulation"""
        y = np.zeros(in_samples * 2).astype('int')

        def hbf_sim():
            p = 0
            yield self.hbf.inp.stb.eq(1)
            for i in range(in_samples*2):
                #if i == 8:  # break input datastream
                #    yield self.hbf.inp.stb.eq(0)
                if i == 12:
                    yield self.hbf.inp.stb.eq(1)
                yield
                yield self.hbf.inp.data.eq(x[i//2])
                if (yield self.hbf.out1.stb):  # check for valid output data
                    p += 1
                    y[p] = yield self.hbf.out1.data

        run_simulation(self.hbf, hbf_sim(), vcd_name="imp_resp.vcd")
        return y

    def test_supersampled_response(self):
        in_samples = 50
        x = np.zeros(in_samples).real.astype('int').tolist()
        x[0] = 1
        h = [1,0,5, 0, 10, 0, 30, 0, 70, 1, 70, 0, 30, 0, 10, 0, 5,0,1]
        self.hbf = SuperHbfUS(h)
        y_sim = self.run_hbf_ss_sim(x, in_samples)[(1 + (len(h))):].real.astype('int').tolist()
        x_zs = np.zeros(in_samples*2).real.astype('int').tolist()
        for i in range(in_samples):
            x_zs[i*2]=x[i]
        y_model = np.convolve(x_zs, h)[:-len(h) - (len(h))].real.astype('int').tolist()
        self.assertEqual(y_sim, y_model)

    def test_normal_response(self):
        in_samples = 50
        x = np.zeros(in_samples).real.astype('int').tolist()
        x[0] = 1
        x[1] = 32
        x[4] = -26
        h = [1,0,5, 0, 10, 0, 30, 0, 70, 1, 70, 0, 30, 0, 10, 0, 5,0,1]
        self.hbf = SuperHbfUS(h)
        y_sim = self.run_hbf_norm_sim(x, in_samples)[(len(h)):].real.astype('int').tolist()
        x_zs = np.zeros(in_samples * 2).real.astype('int').tolist()
        for i in range(in_samples):
            x_zs[i * 2] = x[i]
        y_model = np.convolve(x_zs, h)[:-len(h) + 1 - (len(h))].real.astype('int').tolist()
        self.assertEqual(y_sim, y_model)

class TestCicUs(unittest.TestCase):
    pass

