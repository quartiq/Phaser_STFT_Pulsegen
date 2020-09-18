# SingularitySurfer 2020


import numpy as np
from migen import *
from misoc.interconnect.stream import Endpoint

class SuperCicUS(Module):
    """Supersampled CIC filter upsampler. Interpolates the input by variable rate r.
    Processes two new output samples every clockcycle.

    TODO: figure out uneven ratechange and changing r during operation

    Parameters
    ----------
    width_d: width of the data in- and output
    n: cic order
    r_max: maximum interpolation rate
    """

    def __init__(self, width_d=16, n=6, r_max=2048):
        if r_max < 2:
            raise ValueError()
        if n < 1:
            raise ValueError()
        if r_max < 1:
            raise ValueError()
        b_max = np.ceil(np.log2(r_max))  # max bit growth

        ###
        self.input = Endpoint([("data", (width_d, True))])
        self.output = Endpoint([("data0", (width_d, True)),
                                ("data1", (width_d, True))])
        self.r = Signal(int(np.ceil(np.log2(r_max))))  # rate input (inherently multiplied by two due to supersampling)
        ###

        i = Signal.like(self.r)
        comb_ce = Signal()
        inp_stall = Signal()


        r_reg = Signal.like(self.r)  # i think changing r while filter is running will lead to instability
        log2r = self._ceil_log2(self.r)
        scale = Signal(int(np.ceil(np.log2(r_max) + np.log2(n))))

        self.comb += [
            comb_ce.eq(self.input.ack & self.input.stb),
            self.output.ack.eq(~inp_stall),
            self.input.ack.eq((i == r_reg) | inp_stall),
        ]

        self.sync += [
            inp_stall.eq( self.input.ack & ~self.input.stb)
        ]

        self.sync += [
            r_reg.eq(self.r[1:]),  # times two due to supersampling
            scale.eq(log2r * (n-1)),  # TODO: wrong
            If(~inp_stall,
               i.eq(i+1),
               ),
            If(i == r_reg,
                i.eq(0),
               ),
        ]

        sig = self.input.data
        width = len(sig)

        # comb stages, one pipeline stage each
        for _ in range(n):
            old = Signal((width, True))
            width += 1
            diff = Signal((width, True))
            self.sync += [
                If(comb_ce,
                   old.eq(sig),
                   diff.eq(sig - old)
                   ),
            ]
            sig = diff

        # zero stuffer, gearbox, and first integrator, one pipeline stage
        width -= 1
        sig_a = Signal((width, True))
        sig_b = Signal((width, True))
        sig_i = Signal((width, True))
        self.comb += [
            sig_i.eq(sig_b + sig),
        ]
        self.sync += [
            sig_a.eq(sig_b),
            If(comb_ce,
                sig_b.eq(sig_i),
            )
        ]

        # integrator stages, two pipeline stages each
        for _ in range(n - 1):
            sig_a0 = Signal((width, True))
            sum_ab = Signal((width + 1, True))
            width += int(b_max - 1)
            sum_a = Signal((width, True))
            sum_b = Signal((width, True))
            self.sync += [
                If(~inp_stall,
                    sig_a0.eq(sig_a),
                    sum_ab.eq(sig_a + sig_b),
                    sum_a.eq(sum_b + sig_a0),
                    sum_b.eq(sum_b + sum_ab),
                )
            ]
            sig_a, sig_b = sum_a, sum_b

        self.comb += [
            self.output.data0.eq(sig_a >> scale),
            self.output.data1.eq(sig_b >> scale),
        ]

    def _ceil_log2(self, x):
        """combinatorial ceil(log2(x)) computation"""
        temp = Signal(int(np.ceil(np.log2(len(x)))))
        log2x = Signal(int(np.ceil(np.log2(len(x)))))
        for i in range(len(x)):
            self.comb += [
                If(x[i],
                   temp.eq(i)
                   ),
                log2x.eq(temp+1)
            ]
        return log2x


    def sim(self):
        yield
        yield self.input.data.eq(1)
        yield self.input.stb.eq(1)
        yield self.r.eq(30)
        yield
        for i in range(1000):
            if i==100:
                yield self.input.stb.eq(0)
                print(i)
            if i==200:
                yield self.input.stb.eq(1)
                print(i)
            #yield self.input.data.eq(0)
            yield

if __name__ == "__main__":
    test = SuperCicUS()
    run_simulation(test, test.sim(), vcd_name="cic_sim.vcd")