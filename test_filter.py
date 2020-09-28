
import unittest
import numpy as np
import matplotlib.pyplot as plt
from migen import *
from misoc.interconnect.stream import Endpoint

from super_hbf import SuperHbfUS
from super_cic import SuperCicUS

class TestHbfUS(unittest.TestCase):
    """HBF upsampler tests"""

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
        x = [0] * in_samples
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
        x = [0] * in_samples
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
    '''Supersampled CIC tests'''

    def run_cic_sim(self, x, in_samples, r):

        y = []
        def sim(r):
            yield
            yield self.cic.r.eq(r)
            yield
            for i in range(500):
                yield
                if i < len(x):
                    yield self.cic.input.data.eq(x[i])
                yield self.cic.input.stb.eq(1)
                if (yield self.cic.output.stb):  # check for valid output data
                    v0 = yield self.cic.output.data0
                    v1 = yield self.cic.output.data1
                    y.append(v0)
                    y.append(v1)
        run_simulation(self.cic, sim(r))
        return y

    def compute_cic_resp(self, x, r, n, tweaks, shifts, bitshift_lut_width, width_lut):
        '''computes the cic response as implemented in SuperCicUs'''
        h_cic = 1
        for i in range(n):
            h_cic = np.convolve(np.ones(r), h_cic)
        h_cic = h_cic.astype('int')
        print(h_cic.sum())
        for i, e in enumerate(x):
            x[i] = (e * tweaks[r]) >> (width_lut - bitshift_lut_width - 1)
        y_full = np.convolve(x, h_cic)
        for i, e in enumerate(y_full):
            y_full[i] = e >> shifts[r]
        return y_full.astype('int').tolist()


    def test_response(self):
        n = 6
        r_max = 2048
        width_lut = 18
        self.cic = SuperCicUS( 16, n, r_max, True, width_lut)
        in_samples = 50
        x = [0] * in_samples
        x[0] = 10000
        r = 7
        tweaks = np.arange(r_max)
        tweaks[0] = 1
        shifts = np.ceil(np.log2(tweaks ** (n - 1))).astype('int').tolist()
        bitshift_lut_width = int(np.ceil(np.log2(max(shifts))))
        tweaks = (np.ceil(np.log2(tweaks ** (n - 1))) - np.log2(tweaks ** (n - 1)))
        tweaks = (2 ** tweaks)
        tweaks = tweaks * 2 ** (width_lut - bitshift_lut_width - 1)
        tweaks = tweaks.astype('int').tolist()
        y_sim = self.run_cic_sim(x, in_samples, r)
        y_model = self.compute_cic_resp(x, r, n, tweaks, shifts, bitshift_lut_width, width_lut)
        self.assertEqual(y_sim[64:64+len(y_model)], y_model)












