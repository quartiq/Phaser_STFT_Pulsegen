
from migen import *
from misoc.cores.spi2 import SPIMachine, SPIInterface
from misoc.cores.duc import PhasedDUC

from .FFT_playground.fft_generator_migen import Fft
from crg import CRG




class Phaser_STFT(Module):
    def __init__(self, platform):

        self.submodules.crg = CRG(platform)

        self.submodules.fft = Fft(128, ifft=True)

        outp = platform.request("test_point", 1)
        inp = platform.request("test_point", 0)
        sr = Signal(32)
        self.comb += [
            self.fft.x_in.eq(sr),
            self.fft.x_in_we(sr),
            self.fft.x_in_adr.eq(sr),
            self.fft.start(sr),
            self.fft.x_out_adr.eq(sr),
            outp.eq(self.fft.x_out == 1)
        ]

        self.sync += [
            Cat(sr).eq(Cat(sr,inp))
        ]



if __name__ == "__main__":
    from phaser_platform import Platform
    platform = Platform(load=False)
    test = Phaser_STFT(platform)
    platform.build(test, build_name="phaser_stft")