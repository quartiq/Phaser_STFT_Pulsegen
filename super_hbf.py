# SingularitySurfer 2020

import numpy as np
import matplotlib.pyplot as plt
from migen import *
from misoc.interconnect.stream import Endpoint


class SuperHbfUS(Module):
    """Supersampled half-band fir filter upsampler. Every second output sample is just the input delayed.
    This module computes the nontrivial sample at every cycle. Fully pipelined DSPs
    delay = 2 + (n / 2)

    Parameters
    ----------
    coeff: filter coeffiecients. fixedpoint is assumed to be at width_c, so all bits are fractional.
            Has to be length 4*n-1. All uneven tabs are ignored (have to be zero or one)
    width_d: width of the data in- and output
    width_c: coefficient width.
    """

    def __init__(self, coeff, width_d=16, width_c=18):
        n = (len(coeff) + 1) // 4
        if len(coeff) != n * 4 - 1:
            raise ValueError("HBF length must be 4*n-1", coeff)
        elif n < 2:
            raise ValueError("Need order n >= 2")
        for i, c in enumerate(coeff):
            if i == n * 2 - 1:
                if not c:
                    raise ValueError("HBF center tap must not be zero")
                scale = log2_int(c)
            elif i & 1:
                if c:
                    raise ValueError("HBF even taps must be zero", (i, c))
            elif not c:
                raise ValueError("HBF needs odd taps", (i, c))
            elif c != coeff[-1 - i]:
                raise ValueError("HBF must be symmetric", (i, c))

        ###
        self.ss = Signal()  # Supersample switch signal
        self.inp = Endpoint([("data", (width_d, True))])
        self.out1 = Endpoint([("data", (width_d, True))])
        self.out2 = Endpoint([("data", (width_d, True))])
        ###

        even = Signal()

        a = [Signal((width_d, True), reset_less=True) for _ in range(int((len(coeff) + 1) / 2))]
        b = [Signal((width_d + 1, True), reset_less=True) for _ in range(n)]
        c = [Signal((width_d + 1 + width_c, True), reset_less=True) for _ in range(n)]
        d = [Signal((width_d + 1 + width_c, True), reset_less=True) for _ in range(n)]

        self.sync += [
            If(self.inp.stb & self.inp.ack,  # if new input sample
               Cat(a).eq(Cat(self.inp.data, a))
               )
        ]
        for idx, cof in enumerate(coeff[:n * 2:2]):
            self.sync += [
                If(self.inp.stb & self.inp.ack,  # if new input sample
                   b[idx].eq(a[idx] + a[-(idx + 1)]),  # first dsp reg: preadd
                   c[idx].eq(b[idx] * cof),  # second dsp reg: mult
                   )
            ]
            if idx >= 1:  # don't accumulate the output at the input to avoid meltdown
                self.comb += d[idx].eq(c[idx] + d[idx - 1])  # accumulate from the left

        self.comb += d[0].eq(c[0])

        self.sync += [
            If(self.ss,
               self.inp.stb.eq(1),
               self.out1.ack.eq(self.inp.stb),  # always emit new sample if new input
               self.out2.ack.eq(self.inp.stb),  # always emit new sample if new input
               self.out1.data.eq(d[-1][:]),  # last dsp accu is nontrivial output shifted
               self.out2.data.eq(a[n + 1]),  # last dsp accu is nontrivial output
               ).Else(
                self.out2.ack.eq(0),  # dont use second output
                self.out2.data.eq(0),
                If(self.inp.ack & self.inp.stb,
                   even.eq(1),
                   self.inp.stb.eq(0),
                   self.out1.ack.eq(1),  # new outsample
                   self.out1.data.eq(d[-1][:]),  # last dsp accu is nontrivial output shifted
                   ).Else(
                    self.out1.ack.eq(Mux(even, 1, 0)),  # if second sample, set out valid
                    even.eq(0),
                    self.inp.stb.eq(1),
                    self.out1.data.eq(a[n+2]),  # trivial output
                )
            )
        ]