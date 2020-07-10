#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# SingularitySurfer 2020


import numpy as np
from migen import *
from functools import reduce
from operator import and_


class Fft(Module):
    """Migen FFT generator

        Parameters
        ----------
        n : FFT size
        ifft : forward or inverse fft
        width_i : input width
        width_o : output width
        width_int : internal computation and memory width
        width_wram : twiddle memory width
        input_order : natural or bit-reversed input
        cmult : complex multiplier option
    """

    def __init__(self, n=128, ifft=False, width_i=16, width_o=16, width_int=16,
                 width_wram=18, input_order='natural', cmult='4_dsp'):
        # Parameters
        # =============================================================
        # TODO: Input parameter checks

        self.n = n
        self.width_int = width_int
        self.width_i = width_i
        self.width_o = width_o
        self.width_wram = width_wram
        self.cmult = cmult
        self.ifft = ifft
        m = np.log2(n)
        assert m % 1 == 0, "input vector length needs to be power of two long"
        self.log2n = int(m)

        # IO signals
        # =============================================================

        self.x_in = Signal((width_i, True))
        self.x_in_addr = Signal((self.log2n))
        self.x_out = Signal((width_o, True))
        self.x_out_addr = Signal((self.log2n))
        self.start = Signal()  # input start signal
        self.busy = Signal()  # busy indicator
        self.done = Signal()  # output done signal
        self.oflw = Signal()  # overflow indicator

        ###

        # internal Signals
        # =============================================================
        ar = Signal((self.width_int, True))
        ai = Signal((self.width_int, True))
        br = Signal((self.width_int, True))
        bi = Signal((self.width_int, True))
        self.stage = Signal(int(np.ceil(np.log2(self.log2n)))+1)  # global stage counter
        self.en = Signal()  # global bfl computation enable Signal

        # Instantiate Butterfly
        # =============================================================

        cr, ci, dr, di = self._bfl_core(ar, ai, br, bi)

        # Data Memories
        # =============================================================

        # debug: fill data memory with initial tone
        x = np.zeros(n)
        y = np.zeros(n, dtype="complex")
        x[3] = ((1 << self.width_i - 1) - 1) << self.width_int  # shift to complex data
        # maximal single real tone at 3rd coef (without DC). shifted to mem offset for real values.
        for i, k in enumerate(x):
            binary = bin(i)
            reverse = binary[-1:1:-1]
            pos = int(reverse + (self.log2n - len(reverse)) * '0', 2)
            y[i] = x[pos]
        # Y=np.arange(0,self.n)
        # Y=Y|(-Y)<<self.width_i
        tempmem1 = y[::2].real.astype('int').tolist()
        tempmem2 = y[1::2].real.astype('int').tolist()

        xram1 = Memory(width_int * 2, int(n / 2), init=tempmem1, name="data1")
        xram2a = Memory(width_int * 2, int(n / 2), init=tempmem2, name="data2a")
        xram2b = Memory(width_int * 2, int(n / 2), init=tempmem2, name="data2b")
        xram1_port1 = xram1.get_port(mode=READ_FIRST)
        xram1_port2 = xram1.get_port(write_capable=True)
        xram2a_port1 = xram2a.get_port(mode=READ_FIRST)
        xram2a_port2 = xram2a.get_port(write_capable=True)
        xram2b_port1 = xram2b.get_port(mode=READ_FIRST)
        xram2b_port2 = xram2b.get_port(write_capable=True)
        dat_r = Signal(width_int * 2)

        self.specials += xram1, xram1_port1, xram1_port2, xram2a, \
                         xram2b, xram2a_port1, xram2a_port2, xram2b_port1, xram2b_port2

        # Memory Wiring
        # =============================================================
        a_mux_l, c_mux, a_x2_mux_l, c_x2_mux, x1p1_adr, x1p2_adr, x2p1_adr, x2p2_adr, we = self._data_addr_calc()

        self.comb += [  # fetching ports
            xram1_port1.adr.eq(x1p1_adr),
            xram2a_port1.adr.eq(x2p1_adr),
            xram2b_port1.adr.eq(x2p1_adr),
            ar.eq(Mux(a_mux_l == 0, xram1_port1.dat_r[:self.width_int], dat_r[:self.width_int])),
            ai.eq(Mux(a_mux_l == 0, xram1_port1.dat_r[self.width_int:], dat_r[self.width_int:])),
            br.eq(Mux(a_mux_l, xram1_port1.dat_r[:self.width_int], dat_r[:self.width_int])),
            bi.eq(Mux(a_mux_l, xram1_port1.dat_r[self.width_int:], dat_r[self.width_int:])),
            dat_r.eq(Mux(a_x2_mux_l, xram2a_port1.dat_r, xram2b_port1.dat_r)),
        ]
        self.comb += [  # writeback ports
            xram1_port2.adr.eq(x1p2_adr),
            xram2a_port2.adr.eq(x2p2_adr),
            xram2b_port2.adr.eq(x2p2_adr),
            xram1_port2.we.eq(we),
            xram2a_port2.we.eq((we & c_x2_mux)),
            xram2b_port2.we.eq((we & (~c_x2_mux))),
            xram1_port2.dat_w.eq(Cat(Mux(c_mux == 0, cr, dr), Mux(c_mux == 0, ci, di))),
            xram2a_port2.dat_w.eq(Cat(Mux(c_mux, cr, dr), Mux(c_mux, ci, di))),
            xram2b_port2.dat_w.eq(Cat(Mux(c_mux, cr, dr), Mux(c_mux, ci, di))),
        ]

        # IO logic
        # =============================================================
        self.comb+=[
            If(self.stage==self.log2n,
                self.done.eq(1),
                self.busy.eq(0)
               )
        ]


    def _data_addr_calc(self):
        """data ram address and ram multiplexer position calculator."""
        pos_r = Signal(self.log2n - 1, reset=0)  # read position reg
        pos_w = Signal(self.log2n - 1, reset=0)  # pipeline delay delayed couter
        stage_w = Signal(int(np.ceil(np.log2(self.log2n))), reset=0)  # write stage position; resets to -1 at fft start
        a_mux = Signal()  # a ram muxing signal
        c_mux = Signal()  # c ram muxing signal
        a_mux_l = Signal()  # (last) 1 clk delayed mux; needed to route data at ram output one clk after addr was set
        a_x2_mux = Signal()  # a muxing signal for double buffered x2 ram
        c_x2_mux = Signal()  # c muxing signal for double buffered x2 ram
        a_x2_mux_l = Signal()  # (last) 1 clk delayed a x2 muxing signal
        we = Signal()  # ram write enable
        posbit_r = Signal()  # one bit of read position counter
        posbit_w = Signal()  # one bit or write position counter
        laststart = Signal()    # start signal one clk cycle ago
        x1p1_adr = Signal(self.log2n - 1)
        x1p2_adr = Signal(self.log2n - 1)
        x2p1_adr = Signal(self.log2n - 1)
        x2p2_adr = Signal(self.log2n - 1)

        self.sync += [
            laststart.eq(self.start),
            If(self.en & self.busy, pos_r.eq(pos_r + 1)),  # count only if enabled; overflows at stage transition
            If(reduce(and_, pos_r), self.stage.eq(self.stage + 1)),
            If(reduce(and_, pos_w), stage_w.eq(stage_w + 1)),
            If((self.start & (~laststart)),  # if start signal was set
               self.busy.eq(1),
               self.stage.eq(0),
               stage_w.eq(-1)  # reset to -1
               ),
            If(reduce(and_, pos_w) & reduce(and_, stage_w) & (self.busy == 1), we.eq(1)),
            # enable write on next (stage_w==0) cycle
            # write enable if at first write pos in first stage
            pos_w.eq(pos_r - self.PIPE_DELAY-1),  # writeback needs to be delayed 4 cycles due to pipelining
            a_mux_l.eq(a_mux),
            a_x2_mux_l.eq(a_x2_mux)
        ]

        for i in range(self.log2n)[1:]:  # Makeshift Mux
            self.comb += If(self.stage == i, posbit_r.eq(pos_r[i - 1]))
        for i in range(self.log2n - 1):  # Makeshift Mux
            self.comb += If(stage_w == i, posbit_w.eq(pos_w[i]))

        self.comb += [
            # fetching logic
            a_x2_mux.eq(self.stage[0]),  # use last bit of stage to toggle between x2 mems
            a_mux.eq(Mux(self.stage == 0, 0, posbit_r)),
            # input multiplexer needs to switch every self.stage cycles (so never in the 0th stage)
            x1p1_adr.eq(pos_r),  # ram 1 is just always sorted
            x2p1_adr.eq((Cat(0, pos_r) ^ (1 << self.stage)) >> 1),
            # flip bit at self.stage position to shuffle ram 2;
            # first append 0 at LSB and then shift out to effectively make self.stage-1.

            # writeback logic
            c_x2_mux.eq(~stage_w[0]),  # use last bit of stage to toggle between x2 mems
            c_mux.eq(posbit_w),
            x1p2_adr.eq(pos_w),  # ram 1 is just always sorted
            x2p2_adr.eq(pos_w)  # ram 2
        ]
        return a_mux_l, c_mux, a_x2_mux_l, c_x2_mux, x1p1_adr, x1p2_adr, x2p1_adr, x2p2_adr, we

    def _bfl_core(self, ar, ai, br, bi):
        """full butterfly core with computation pipeline, twiddle rom and twiddle address calculator."""
        w_idx = self._twiddle_addr_calc()
        wr, wi = self._twiddle_mem_gen(w_idx)
        return self._bfl_pipe4(ar, ai, br, bi, wr, wi)

    def _bfl_pipe4(self, ar, ai, br, bi, wr, wi):
        """Butterfly computation pipe.
        1 fetching stage, 3 computation stages (with last addition and writeback in one cycle) and uses 4 DSPs.
        Scaling is performed at the end of the computation.
        """
        self.PIPE_DELAY = 4
        s1_dsp1 = Signal((self.width_int, True))
        s1_dsp2 = Signal((self.width_int, True))
        wi_reg = Signal((self.width_wram, True))
        wr_reg = Signal((self.width_wram, True))
        bi_reg = Signal((self.width_int, True))
        ar_reg = [Signal((self.width_int + 1, True)) for i in range(2)]
        ai_reg = [Signal((self.width_int + 1, True)) for i in range(2)]
        s2_dsp1 = Signal((self.width_int + 1, True))
        s2_dsp2 = Signal((self.width_int + 1, True))
        cr_full = Signal((self.width_int + 2, True))
        ci_full = Signal((self.width_int + 2, True))
        dr_full = Signal((self.width_int + 2, True))
        di_full = Signal((self.width_int + 2, True))
        cr = Signal((self.width_int, True))
        ci = Signal((self.width_int, True))
        dr = Signal((self.width_int, True))
        di = Signal((self.width_int, True))
        self.sync += [
            #       (zeroth stage: fetching)

            #       first stage
            s1_dsp1.eq((br * wr) >> self.w_p),
            s1_dsp2.eq((br * wi) >> self.w_p),
            wi_reg.eq(wi),
            wr_reg.eq(wr),
            bi_reg.eq(bi),
            ar_reg[0].eq(ar),
            ai_reg[0].eq(ai),
            #       second stage
            s2_dsp1.eq(s1_dsp1 - ((bi_reg * wi_reg) >> self.w_p)),
            s2_dsp2.eq(s1_dsp2 + ((bi_reg * wr_reg) >> self.w_p)),
            ar_reg[1].eq(ar_reg[0]),
            ai_reg[1].eq(ai_reg[0]),
            #       third stage         TODO: nonfixed scaling and overflow
            cr_full.eq(s2_dsp1 + ar_reg[1]),
            ci_full.eq(s2_dsp2 + ai_reg[1]),
            dr_full.eq(ar_reg[1] - s2_dsp1),
            di_full.eq(ai_reg[1] - s2_dsp2),
        ]
        self.comb += [
            # just shift by one for now
            cr.eq(cr_full >> 1),
            ci.eq(ci_full >> 1),
            dr.eq(dr_full >> 1),
            di.eq(di_full >> 1)
        ]
        return cr, ci, dr, di

    def _twiddle_mem_gen(self, w_idx):
        """generates twiddle rom and logic for assembling the twiddles from one quater circle"""

        pos = np.linspace(0, np.pi / 2, int(self.n / 4), False)
        self.w_p = self.width_wram - 2  # Fixed point position of twiddles. One bit is sign and one is nonfractional (ie 1 at the 0th twiddle)
        twiddles = [(int(_.real) | int(_.imag) << self.width_wram)
                    for _ in np.round((1 << (self.width_wram - 2)) * np.exp(-1j * pos))]
        wram = Memory(self.width_wram * 2, int(self.n / 4), init=twiddles, name="twiddle")
        wram_port = wram.get_port()
        self.specials += wram, wram_port
        wr = Signal((self.width_wram, True))
        wi = Signal((self.width_wram, True))
        wr_ram = Signal((self.width_wram, True))
        wi_ram = Signal((self.width_wram, True))
        w_idx_l = Signal()  # last upper index bits
        self.comb += [
            wram_port.adr.eq(w_idx[:-1]),
            wr_ram.eq(wram_port.dat_r[:self.width_wram]),  # get twiddle real
            wi_ram.eq(wram_port.dat_r[self.width_wram:]),  # get twiddle imag
        ]
        if (self.ifft):
            self.comb += [
                wr.eq(Mux(w_idx_l, wi_ram, wr_ram)),
                wi.eq(Mux(w_idx_l, wr_ram, -wi_ram))
            ]
        else:
            self.comb += [
                wr.eq(Mux(w_idx_l, wi_ram, wr_ram)),
                wi.eq(Mux(w_idx_l, -wr_ram, wi_ram))
            ]
        self.sync += w_idx_l.eq(w_idx[-1])
        return wr, wi

    def _twiddle_addr_calc(self):
        """ calculates address for twiddle rotator """
        w_idx = Signal(self.log2n - 1, reset=0)
        step = Signal(self.log2n)  # make one bigger than w_idx to have overflow every step in 0th stage

        for i in range(self.log2n):  # Makeshift Mux
            self.comb += If(self.stage == i, step.eq(1 << (self.log2n - i - 1)))

        self.sync += If(self.en, w_idx.eq(w_idx + step))
        return w_idx

    def internal_tb(self):
        """ basic testbench for ifft with fixed ram pre-initialization """
        for i in range(1024):
            yield
            if i == 0:
                yield self.start.eq(1)
                yield self.en.eq(1)


if __name__ == "__main__":
    test = Fft(n=128, ifft=True)

    run_simulation(test, test.internal_tb(), vcd_name="internal_BIG.vcd")
    # run_simulation(test, test.twiddle_tb(), vcd_name="twiddle.vcd")
    # run_simulation(test, test.bfl_tb(), vcd_name="bfl.vcd")
    # print(o.main_source)
