import unittest
import numpy as np
from migen import *
from fft_generator_migen import Fft
from fft_model import FftModel


class TestFft(unittest.TestCase):
    def test_bitreversed(self):
        """ bitreversed input test for 128 point fft with randomized phases.

        compares the simulation output against a bit-accurate numeric model
        maximum amplitude input coefficients with randomized phase
        """
        self.fft = Fft(n=128, ifft=True)
        x = np.ones(self.fft.n, dtype="complex")
        ampl = 2048
        seed = np.random.randint(2 ** 32)
        np.random.seed(seed)
        print(f'random seed: {seed}')
        phase = np.random.rand(self.fft.n) * 2 * np.pi
        x = x * ampl * np.exp(1j * phase)
        fft_model = FftModel(x, w_p=14)
        x_o_model = fft_model.full_fft(scaling='one', ifft=True)  # model output
        x_o_sim = np.zeros(self.fft.n, dtype="complex")  # simulation output
        
        x_mem = np.zeros(self.fft.n)
        y = np.zeros(self.fft.n, dtype="complex")
        for i, k in enumerate(x):
            x_mem[i] = (int(k.real) & int("0x0000ffff", 0)) | (int(k.imag) << self.fft.width_int)
        for i, k in enumerate(x_mem):  # bit reverse
            binary = bin(i)
            reverse = binary[-1:1:-1]
            pos = int(reverse + (self.fft.log2n - len(reverse)) * '0', 2)
            y[i] = x_mem[pos]
        y = y.real.astype('int').tolist()
        
        def io_brev_tb():
            """ input output testbench for 128 point fft"""
            
            p = 0
            for i in range(1024):
                yield
                yield self.fft.start.eq(0)
                if i < self.fft.n:                  # load in values
                    yield self.fft.x_in_we.eq(1)
                    yield self.fft.x_in_adr.eq(i)
                    yield self.fft.x_in.eq(y[i])
                if i == self.fft.n + 1:             # start fft
                    yield self.fft.x_in_we.eq(0)
                    yield self.fft.start.eq(1)
                    yield self.fft.en.eq(1)
                if (yield self.fft.done):           # retrieve ifft output
                    yield self.fft.x_out_adr.eq(p)
                    p += 1
                    xr2cpl = yield self.fft.x_out[:self.fft.width_o]  # x real in twos complement
                    xi2cpl = yield self.fft.x_out[self.fft.width_o:]  # x imag in twos complement
                    if xr2cpl & (1 << self.fft.width_o - 1):
                        xr = xr2cpl - 2 ** self.fft.width_o
                    else:
                        xr = xr2cpl
                    if xi2cpl & (1 << self.fft.width_o - 1):
                        xi = xi2cpl - 2 ** self.fft.width_o
                    else:
                        xi = xi2cpl
                    if p >= 3:
                        x_o_sim[p - 3] = xr + 1j * xi

                if p >= (self.fft.n + 2):
                    break

        run_simulation(self.fft, io_brev_tb(), vcd_name="unittest.vcd")
        print(x_o_sim)
        self.assertEqual(x_o_model.tolist(), x_o_sim.tolist())  # in console output second input is displayed first


if __name__ == '__main__':
    unittest.main()
