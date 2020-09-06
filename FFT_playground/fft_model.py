#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt


class FftModel:
    """fixed point radix2 dit fft with fixed point scaling numerical model
    
    Takes a complex fixed point vector with fixed point position to take the fft of.
    The data is stored in an internal vector and successive fft stages can be performed.
    
    Fixed point ismodeled via integers and bitshifting.
    Unfortunately this means some more interpretation of intermediate results...
    Overflows are captured by the model, if the total nr of bits for real/complex data is provided.
    Sign bit is not included.
    
    Dataformat example: X[0]= 1024 + 32j ; x_p = 8   ==>  X_inout[0] = 2 + 0.125j
    
    Truncation is used after the multipliers and for fft stage scaling as more
    adder logic is required for rounding. (Essentially an adder at every point or rounding)
    
    
    
    https://www.ti.com/lit/an/spra948/spra948.pdf
    
    Parameters
        ----------
    x_in : complex array
        complex input vector
    x_p : int
        input array fixed point position / nr fractional bits
    w_p : int
        twiddle factor (fractinal) bits. 
    x_bits: int
        total number of bits in input, used for overflow checking

    """

    def __init__(self, x_in, x_p=0, w_p=16, x_bits=1024):  # default nr of bits is _very_ high so no oflw problems
        self.x_bits = x_bits
        self.w_p = w_p
        self.x_p = x_p  # data fixed point position. scales during fft.
        self.size = len(x_in)  # Nr samples
        assert np.log2(self.size) % 1 == 0, 'input length has to be power of two!'
        self.stages = int(np.log2(self.size))  # Nr stages (als nr. bits of index)
        self.stage = 0  # current stage
        self._bfls = int(self.size / 2)  # nr _bfls per stage

        x_brev = self.bit_reverse(x_in, self.stages)  # bit reverse
        # self.xr=np.arange(0,self.size)# debug data
        self.xr = (x_brev.real * 2 ** x_p).astype(int)  # real fixedpoint mem
        # self.xi=np.arange(0,-self.size,-1)# debug data
        self.xi = (x_brev.imag * 2 ** x_p).astype(int)  # imag fixedpoint mem

        w = np.exp(-2j * (np.pi / self.size) * np.arange(self.size / 2))  # only uses half circle twiddles
        self.wr = np.round((w.real * 2 ** self.w_p)).astype(int)  # real twiddle mem
        self.wi = np.round((w.imag * 2 ** self.w_p)).astype(int)  # imag twiddle mem

    def full_fft(self, scaling='one', ifft=False):
        """
        perform full fft and return output data

        Parameters
        ----------
        scaling : string, optional
            scaling option. The default is 'one'.
        ifft : bool, optional
            ifft of fft stage. The default is False.

        Returns
        -------
        X : complex array
            output data vector 

        """


        for i in range(self.stages):
            if ifft:
                self.x_p += 1  # always divide by 2 in ifft stage. ADDITIONAL scaling from the scaling parameter possible!!
            if scaling == 'none':  # no scaling
                self.fft_stage(0, ifft)
            elif scaling == 'one':  # scale by one in each stage
                self.fft_stage(1, ifft)
                self.x_p -= 1
            elif scaling == 'no_oflw':  # scale by one in first and second stage and by two in later
                if i < 2:  # no overflows in real arch guaranteed
                    self.fft_stage(1, ifft)
                    self.x_p -= 1
                else:
                    self.fft_stage(2, ifft)
                    self.x_p -= 2
            elif scaling == '4tone_ifft':  # scale on first two stages and then no more
                if i < 2:
                    self.fft_stage(1, ifft)
                    self.x_p -= 1
                else:
                    self.fft_stage(0, ifft)
            elif isinstance(scaling, int):
                self.fft_stage(int(i < self.stages - scaling), ifft)
                self.x_p -= int(i < self.stages - scaling)

        return self.xr * 2 ** -self.x_p + 1j * self.xi * 2 ** -self.x_p

    def fft_stage(self, s, ifft=False):
        """
        perform radix2 stage with scaling on data

        Parameters
        ----------
        s : int
            nr. scale bits 
        ifft : bool, optional
            fft or ifft stage. The default is False.

        Returns
        -------
        None.

        """
        assert (np.amax(abs(self.xr)) < 2 ** (self.x_bits + 1)), "OVERFLOW!"
        assert (np.amax(abs(self.xi)) < 2 ** (self.x_bits + 1)), "OVERFLOW!"
        t_s = (self._bfls) >> self.stage  # twiddle index step size
        for i in range(self._bfls):
            w_idx = (t_s * i) % (self._bfls)  # twiddle factor index for each stage. wraps around.
            q = (1 << (self.stage)) - 1  # lower bits bitmask ie 000000000011 for s=3. responsible for consecuteve parts
            x_idx = (((i & ~q) << 1) | (i & q)) + ( 1 << self.stage)  # compute memory adress.
            ar, ai = self.xr[x_idx - (1 << self.stage)], self.xi[x_idx - (1 << self.stage)]  # mem access
            br, bi = self.xr[x_idx], self.xi[x_idx]
            wr, wi = self.wr[w_idx], self.wi[w_idx]
            wi = -wi if ifft else wi  # complex conjugate for ifft
            cr, ci, dr, di = self._bfl(ar, ai, br, bi, wr, wi, self.w_p, s)  # butterfly with no scaling
            self.xr[x_idx - (1 << self.stage)], self.xi[x_idx - (1 << self.stage)] = cr, ci
            self.xr[x_idx], self.xi[x_idx] = dr, di

        self.stage += 1
        assert (np.amax(abs(self.xr)) < 2 ** (self.x_bits + 1)), "OVERFLOW!"
        assert (np.amax(abs(self.xi)) < 2 ** (self.x_bits + 1)), "OVERFLOW!"

    def _bfl(self, ar, ai, br, bi, wr, wi, p, s):
        """
        Fixedpoint butterfly computation with scaling.
        Parameters
        ----------
        ar : fixed point
            input a real
        ai : fixed point
            input a imag
        br : fixed point
            input b real
        bi : fixed point
            input b imag
        wr : fixed point 
            twiddle factor real
        wi : fixed point 
            twiddle factor imag
        p : int
            twiddle factor length
        s : int
            out scale factor in bits

        Returns
        -------
        cr : fixed point
            output c real
        ci : fixed point
            output c imag
        dr : fixed point
            output d real
        di : fixed point
            output d imag
        """
        b_w_r, b_w_i = self.cmult4(br, bi, wr, wi, p)
        cr = (ar + b_w_r) >> s
        ci = (ai + b_w_i) >> s
        dr = (ar - b_w_r) >> s
        di = (ai - b_w_i) >> s
        return cr, ci, dr, di

    def test_bfl(self, a, b, ab_p, w, w_p):
        """ quick _bfl eval with complex inputs
        
            ab_p fixed point position of a and b from LSB
            w_p fixed point position of w from LSB
        """
        ar = int(a.real * 2 ** ab_p)
        ai = int(a.imag * 2 ** ab_p)
        br = int(b.real * 2 ** ab_p)
        bi = int(b.imag * 2 ** ab_p)
        wr = int(w.real * 2 ** w_p)
        wi = int(w.imag * 2 ** w_p)

        cr, ci, dr, di = self._bfl(ar, ai, br, bi, wr, wi, w_p, 0)
        c = (cr + 1j * ci) * 2 ** -ab_p
        d = (dr + 1j * di) * 2 ** -ab_p
        print(f'c_fixed={c * 2 ** -ab_p} \t d_fixed={d * 2 ** -ab_p}')
        print(f'c_float={(b * w) + a} \t d_float={-(b * w) + a}')

    def cmult4(self, br, bi, wr, wi, p):
        """
        Fixedpoint complex multiplier using 4 real multipliers 
        with p bitshift truncation after the multipliers.

        Parameters
        ----------
        wr : fixed point
            a real
        wi : fixed point
            a imag
        br : fixed point
            b real
        bi : fixed point
            b imag
        p : int
            shift

        Returns
        -------
        cr : fixed point
            output real
        ci : fixed point 
            output imag

        """
        br_wr = (br * wr) >> p  # b rebl times w rebl
        bi_wi = (bi * wi) >> p  # b imbg times w imbg
        br_wi = (br * wi) >> p
        bi_wr = (bi * wr) >> p
        cr = br_wr - bi_wi
        ci = br_wi + bi_wr
        return cr, ci

    def cmult3(self, ar, ai, br, bi, p):
        """
        Fixedpoint complex multiplier using 3 real multipliers
        with p bitshift truncation after the multipliers.

        Parameters
        ----------
        ar : fixed point
            a real
        ai : fixed point
            a imag
        br : fixed point
            b real
        bi : fixed point
            b imag
        p : int
            shift

        Returns
        -------
        cr : fixed point
            output real
        ci : fixed point 
            output imag

        """
        ar_br = (ar * br) >> p  # a real times b real
        ai_bi = (ai * bi) >> p  # a imag times b imag
        cr = ar_br - ai_bi  # real part of output
        ar_p_ai = (ar + ai) >> 1  # real plus imag of a. shift by one to not exceed DSP input width.
        br_p_bi = br + bi  # real plus imag of b
        temp = (ar_p_ai * br_p_bi) >> ( p - 1)
        # temporary product for imag output. shift by one less to get back to the same fixedpoint pos.
        ci = temp - ar_br - ai_bi  # imag output
        return cr, ci

    def bit_reverse(self, x, bits):
        """
        index bit reverse input array

        Parameters
        ----------
        x : complex array
            inp vector
        bits : int
            nr bits of vector index (e.g. 4 for len(x)=16)

        Returns
        -------
        x_brev : complex array
            output with bit reversed index (i.e. 100 for 001)

        """
        x_brev = np.empty(len(x), 'complex')
        for i, k in enumerate(x):
            binary = bin(i)
            reverse = binary[-1:1:-1]
            pos = int(reverse + (bits - len(reverse)) * '0', 2)
            x_brev[i] = x[pos]
        return x_brev

    def evaluate_slot(self, size, x_bits, w_bits, scaling='none', plot=True):
        """
        Evaluate fft dynamic range performance using the slot noise (Xilinx datasheet) technique.
        See https://www.xilinx.com/support/documentation/ip_documentation/xfft/v9_0/pg109-xfft.pdf.
        However, the datasheet either leaves out critical info or displays wrong plots.
        As only noise in slot is of interest and precise noise power outside slot is not critical,
        the spectrum is set to 0 like in the datasheet.

        Parameters
        ----------
        size : int
            size of input vector
        x_bits : int
            number of bits for input vector
        w_bits : int
            number of bits for twiddle factors
        scaling: string
            scaling option
        plot :  bool, optional
            plot out setting
        
        Returns
        -------
        None.

        """

        # TODO: research on noise modeling and "full scale noise" ie. "where do I cut the bell curve??"
        sigma = 10  # cut off gauss distribution after sigma*std. deviation
        X_t = np.random.normal(0, sigma ** -1, size)  # +1j*np.random.normal(0,sigma**-1,size)  # draw random samples
        X_f = np.fft.fft(X_t)  # take fft or random samples
        X_f[int(len(X_f) / 2):(int(len(X_f) / 2) + int(len(X_f) / 20))] = 0  # cut slot
        X_t = np.fft.ifft(X_f)
        X_t = X_t * 2 ** (x_bits - 1)
        X_t = np.rint(X_t.real) + 1j * np.rint(X_t.imag)  # quantize to nr bits
        X_t = X_t * 2 ** -x_bits
        X_f_float = 20 * np.log10(abs(np.fft.fft(X_t)))  # ideal fft on quantized data
        X_f_float[:int(len(X_f) / 2)] = 0  # cut out region of interest like in xilinx datasheet
        X_f_float[(int(len(X_f) / 2) + int(len(X_f) / 20)):] = 0
        # fft_mod=fft_model(X_t,0,w_bits)         # make new model inside eval for convenience
        self.__init__(X_t, x_bits, w_bits)
        X_f_model = 20 * np.log10(abs(self.full_fft(scaling)))  # model fft on quantized data
        X_f_model[:int(len(X_f) / 2)] = 0  # cut out region of interest like in xilinx datasheet
        X_f_model[(int(len(X_f) / 2) + int(len(X_f) / 20)):] = 0
        if plot:
            plt.rc('font', size=18)
            plt.figure(1, [20, 10])
            plt.title('Slot noise performance:')
            plt.plot(X_f_float, label='ideal')
            plt.plot(X_f_model, label='model')
            plt.legend()
            plt.grid()
            plt.show()

        return X_f_model

    def evaluate_tone(self, size, x_bits, w_bits, scaling='none', plot=True):
        """
        Evaluate using full scale single complex tone. Calculate SNR.

        Parameters
        ----------
        size : int
            size of input vector
        x_bits : int
            number of bits for input vector
        w_bits : int
            number of bits for twiddle factors
        scaling: string
            scaling option
        plot :  bool, optional
            plot out setting

        Returns
        -------
        None.

        """
        tone = 3
        x_t = np.exp(1j * tone * np.linspace(0, 2 * np.pi, size, False))  # make fullscale amplitude 1 (+-0.5)
        x_t = ((x_t + np.random.normal(0, (2 ** -(x_bits - 1)) / np.sqrt(12),
                                       size)) * 2 ** x_bits)  # add one LSB qunatization noise
        x_t = np.rint(x_t.real) + 1j * np.rint(x_t.imag)  # quantize to nr bits
        x_t = x_t * 2 ** -x_bits
        x_f_float = abs(np.fft.fft(x_t)) / size  # ideal fft on quantized data
        x_f_float_db = 20 * np.log10(x_f_float)  # ideal fft on quantized data
        # fft_mod=fft_model(x_t,0,w_bits)         # make new model inside eval for convenience
        self.__init__(x_t, x_bits, w_bits, x_bits)
        x_f_model = abs((self.full_fft(scaling))) / size
        x_f_model_db = 20 * np.log10(x_f_model)  # model fft on quantized data
        if plot:
            plt.rc('font', size=18)
            plt.figure(1, [20, 10])
            plt.title('Single tone performance:')
            plt.plot(x_f_float_db.real, label='ideal')
            plt.plot(x_f_model_db.real, label='model')
            plt.legend()
            plt.grid()
            plt.show()

        # wow, calculating SNR and understanding whats going on is not trivial.. Effects:
        # * SNR of a complex signal is generally slightly higher as the real signal
        # * calculating SNR from spectrum makes a tiny mistake because some noise falls into the signal bin
        snr_in = self.calc_snr(x_f_float, tone)
        snr_out = self.calc_snr(x_f_model, tone)
        print('---------------- \n tone eval:')
        print(f'input SNR: {snr_in} \t output SNR: {snr_out}')

    def evaluate_ifft(self, size, x_bits, w_bits, plot=True):
        """
        evaluate the ifft of a single tone without noise at the input

        Parameters
        ----------
        size : int
            size of input vector
        x_bits : int
            number of bits for input vector
        w_bits : int
            number of bits for twiddle factors
        plot :  bool, optional
            plot out setting

        Returns
        -------
        None.

        """
        x_p = 0 # x_bits - (np.log2(size) - 1)  # nr fractional bits. log2(size) bits before point for signle tone.
        tone = 63
        x_f = np.zeros(size, dtype='complex')
        x_f[tone] = 32767
        # single real tone at tone with max input ampl. will lead to and real cosine and complex sine in time domain

        self.__init__(x_f, x_p, w_bits)
        x_t = (self.full_fft(scaling='one', ifft=True))
        if plot:
            plt.rc('font', size=18)
            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            ax[0].set_title('ifft output:')
            ax[0].plot(x_t)
            x_f = 20 * np.log10(abs(np.fft.fft(x_t).real))  # with only one tone, the even/odd bins will see no noise..
            x_f[x_f == -np.inf] = np.min(x_f[x_f != -np.inf])  # set -inf values to lowest occurring value in plot
            ax[1].set_title('ifft output spectrum:')
            ax[1].plot(x_f.real)

        SNR = self.calc_snr(x_t, tone, False)
        print('---------------- \n ifft eval:')
        print(f'SNR: {SNR} ')

    @staticmethod
    def calc_snr(x, tone, freq_domain=True):
        """ 
        helper to calc SNR of X at complex freq tone
        Data can be given in freq or time domain
        """
        if not freq_domain:
            x = np.fft.fft(x)
        return 10 * np.log10((x[tone] * np.conj(x[tone])) / (np.sum(x * np.conj(x)) - x[tone] * np.conj(x[tone])))
        # calc SNR by integrating over noise


if __name__ == "__main__":
    a = FftModel([.1, .1, .1, .1, .1, .1, .1, .1], 6, 18)
    # eval not yet with scaling
    # a.evaluate_slot(1024,24,16,'none')
    # a.evaluate_tone(1024,16,8,'one')      # very low twiddle precision! noise concentrates at harmonics
    # a.evaluate_tone(1024,16,16,'no_oflw')     # 18 fractional twiddle bits, i think I can build this with 18b wide rom
    a.evaluate_ifft(128, 16, 14)
