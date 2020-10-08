import unittest
import numpy as np
from migen import *

from  super_interpolator import  SuperInterpolator




class TestInterpolator(unittest.TestCase):

    def calc_response(self, r):
        assert (r % 4 == 0) | r == 2, "unsupported rate"
        #  HBF0 impulse response:
        h_0 = [9, 0, -32, 0, 83, 0, -183, 0,
               360, 0, -650, 0, 1103, 0, -1780, 0,
               2765, 0, -4184, 0, 6252, 0, -9411, 0,
               14803, 0, -26644, 0, 83046, 131072, 83046, 0,
               -26644, 0, 14803, 0, -9411, 0, 6252, 0,
               -4184, 0, 2765, 0, -1780, 0, 1103, 0,
               -650, 0, 360, 0, -183, 0, 83, 0,
               -32, 0, 9]
        #  HBF1 impulse response:
        h_1 = [69, 0, -418, 0, 1512, 0, -4175, 0,
               9925, 0, -23146, 0, 81772, 131072, 81772, 0,
               -23146, 0, 9925, 0, -4175, 0, 1512, 0,
               -418, 0, 69]
        if r == 2:
            return h_0
        h_12 = np.convolve(h_0, h_1)
        if r == 4:
            return h_12

    def calc_delay(self, r):
        assert (r % 4 == 0) | r == 2, "unsupported rate"
        if r == 2:
            return 18 + 20

    def interpolator_model(self, x, r):
        h_0 = self.calc_response(r)
        x_stuffed = []
        for xx in x:
            x_stuffed.append(xx)
            x_stuffed.append(0)
        return (np.convolve(x_stuffed, h_0) * 2 ** -17).astype('int').tolist()
    
    def run_sim(self, x, r):
        y = []

        def sim():
            yield
            yield self.inter.r.eq(r)
            yield
            for i in range(500):
                yield
                if i < len(x):
                    yield self.inter.input.data.eq(x[i])
                else:
                    yield self.inter.input.data.eq(0)
                yield self.inter.input.stb.eq(1)
                if (yield self.inter.output.stb):  # check for valid output data
                    v0 = yield self.inter.output.data0
                    if v0<0:
                        v0 += 1
                    v1 = yield self.inter.output.data1
                    if v1<0:
                        v1 += 1
                    y.append(v0)
                    y.append(v1)

        run_simulation(self.inter, sim(), vcd_name="interpolator_sim.vcd")
        return y

    def setUp(self):

        n = 20  # nr input samples
        a_max = (2 ** 14) - 1  # max ampl

        seed = np.random.randint(2 ** 32)
        np.random.seed(seed)
        print(f'random seed: {seed}')
        self.x = (np.round(np.random.rand(n) * a_max)).astype('int').tolist()
        # self.x = [0] * n
        # self.x[0] = a_max
        # self.x[1] = a_max
        # self.x[14] = a_max
        self.inter = SuperInterpolator()

    def test_response_hbf0(self):
        r = 2

        y_model = self.interpolator_model(self.x, r)
        y_sim = self.run_sim(self.x, r)
        delay = self.calc_delay(r)
        y_sim = y_sim[delay : delay + len(y_model)]
        self.assertEqual(y_model, y_sim)



