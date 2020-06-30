#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SingularitySurfer 2020

# Numerical Model for fixed point radix2 dit fft with variable scaling.
# First and second stage shift decimal point by one, later stages by 2.
# https://www.ti.com/lit/an/spra948/spra948.pdf


import numpy as np
import matplotlib.pyplot as plt

class fft_model:
    """fixed point radix2 dit fft with fixed point scaling numerical model
    
    Takes a complex fixed point vector with fixed point position to take the fft of.
    The data is stored in an internal vector and successive fft stages can be performed.
    
    The fixed point is just modeled via integers and bitshifting. 
    Unfortunately this means some more interpretation of intermediate results...
    No overflows are captured by the model!
    
    Truncation is used after the multipliers and for fft stage scaling as more
    adder logic is required for rounding.
    
    Parameters
        ----------
    X : complex array
        complex input vector
    x_p : int
        input array fixed point position / nr fractional bits
    w_p : int
        twiddle factor (fractinal) bits. only fractional bits are saved 
        in real implementation. 1/j/0 are treated seperate

    """
    def __init__(self,X,x_p,w_p):
        self.w_p=w_p
        self.size =len(X)                           #Nr samples
        assert np.log2(self.size)%1==0              ,'input length has to be power of two!'
        self.stages=int(np.log2(self.size))         #Nr stages (als nr. bits of index)
        self.stage=0                                #current stage
        self.bfls=int(self.size/2)                       #nr bfls per stage
        
        X_brev=self.bit_reverse(X,self.stages)      #bit reverse
        self.Xr=(X_brev.real*2**x_p).astype(int)          #real fixedpoint mem
        self.Xi=(X_brev.imag*2**x_p).astype(int)          #imag fixedpoint mem
        
        W=np.exp(-2j*(np.pi/len(X))*np.arange(len(X)/2)) #only uses half circle twiddles
        self.Wr=(W.real*2**w_p).astype(int)          #real twiddle mem
        self.Wi=(W.imag*2**w_p).astype(int)          #real twiddle mem


    def bit_reverse(self,X,bits):   
        """
        index bit reverse input array

        Parameters
        ----------
        X : complex array
            inp vector
        bits : int
            nr bits of vector index (e.g. 4 for len(X)=16)

        Returns
        -------
        X_brev : complex array
            output with bit reversed index (i.e. 100 for 001)

        """
        X_brev=np.empty(len(X),'complex')
        for i,x in enumerate(X):
            binary = bin(i) 
            reverse = binary[-1:1:-1] 
            pos=int(reverse + (bits - len(reverse))*'0',2)
            X_brev[i]= X[pos]
        return X_brev
        

    
    def fft_stage(self):
        """
        perform radix2 stage on data
        """
        t_s=(self.bfls)>>self.stage                 # twiddle index step size
        for i in range(self.bfls):
            w_idx=(t_s*i)%(self.bfls)               # twiddle factor index for each stage. wraps around.
            q=(1<<(self.stage))-1                   # lower bits bitmask ie 000000000011 for s=3. responsible for consecuteve parts
            #q_n=~q                                 # higher bits bitmask ie 111111111100 for s=3. responsible for the jumps
            x_idx=(((i & ~q) << 1) | (i & q)) + (1<<self.stage)  # compute memory adress. the + is crap, i have to understand this better
            ar,ai=self.Xr[x_idx-(1<<self.stage)],self.Xi[x_idx-(1<<self.stage)]  #mem access
            br,bi=self.Xr[x_idx],self.Xi[x_idx]
            wr,wi=self.Wr[w_idx],self.Wi[w_idx]
            cr,ci,dr,di=self.bfl(ar,ai,br,bi,wr,wi,self.w_p,0)    #butterfly with no scaling
            self.Xr[x_idx-(1<<self.stage)],self.Xi[x_idx-(1<<self.stage)]=cr,ci
            self.Xr[x_idx],self.Xi[x_idx]=dr,di
            
        self.stage +=1
    
    
    def bfl(self,ar,ai,br,bi,wr,wi,p,s):
        """
        Fixedpoint butterfly computation with scaling.
        Uses complex multiplication with 3 multipliers and 5 adders.
        
        
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
            prescale

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
        b_w_r,b_w_i=self.cmult3(br,bi,wr,wi,p)
        cr=(ar+b_w_r)>>s
        ci=(ai+b_w_i)>>s
        dr=(ar-b_w_r)>>s
        di=(ai-b_w_i)>>s
        return cr,ci,dr,di
        
        
    def test_bfl(self,a,b,ab_p,w,w_p):
        """ quick bfl eval with complex inputs
        
            ab_p fixed point position of a and b from LSB
            w_p fixed point position of w from LSB
        """
        ar=int(a.real*2**ab_p)
        ai=int(a.imag*2**ab_p)
        br=int(b.real*2**ab_p)
        bi=int(b.imag*2**ab_p)
        wr=int(w.real*2**w_p)
        wi=int(w.imag*2**w_p)
        
        
        cr,ci,dr,di=self.bfl(ar,ai,br,bi,wr,wi,w_p,0)
        c=(cr+1j*ci)*2**-ab_p
        d=(dr+1j*di)*2**-ab_p
        print(f'c_fixed={c*2**-ab_p} \t d_fixed={d*2**-ab_p}')
        print(f'c_float={(b*w)+a} \t d_float={-(b*w)+a}')
    
    
        
    def cmult3(self,ar,ai,br,bi,p):
        """
        Fixedpoint complex multiplier with p bitshift truncation after the multipliers.

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
        ar_br=(ar*br)>>p            #a real times b real
        ai_bi=(ai*bi)>>p            #a imag times b imag
        cr=ar_br-ai_bi              #real part of output
        ar_p_ai=ar+ai               #real plus imag of a
        br_p_bi=br+bi               #real plus imag of b
        temp=(ar_p_ai*br_p_bi)>>p   #temporary product for imag output
        ci=temp-ar_br-ai_bi         #imag output
        return cr, ci
        
        
    def full_fft(self):
        """
        perform full fft and return output data
        """
        
        for i in range(self.stages):
            self.fft_stage() 
        return (self.Xr+1j*self.Xi)
    
    
    def evaluate_slot(self,size,x_bits,w_bits,plot=True):
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
        plot :  bool, optional
            plot out setting
        
        Returns
        -------
        None.

        """
        
        #TODO: research on noise modeling and "full scale noise" ie. "where do I cut the bell curve??"
        sigma=3             # cut off gauss distribution after sigma*std. deviation
        X_t=np.random.normal(0,sigma**-1,size)#+1j*np.random.normal(0,sigma**-1,size)    # draw random samples
        X_f=np.fft.fft(X_t)             #take fft or random samples
        X_f[int(len(X_f)/2):(int(len(X_f)/2)+int(len(X_f)/20))]=0       #cut slot
        X_t=np.fft.ifft(X_f)
        X_t=X_t*2**x_bits
        X_t=np.rint(X_t.real)+1j*np.rint(X_t.imag)          #quantize to nr bits
        X_f_float=20*np.log10(abs(np.fft.fft(X_t).real*2**-x_bits))     #ideal fft on quantized data
        X_f_float[:int(len(X_f)/2)]=0           # cut out region of interest like in xilinx datasheet
        X_f_float[(int(len(X_f)/2)+int(len(X_f)/20)):]=0
        #fft_mod=fft_model(X_t,0,w_bits)         # make new model inside eval for convenience
        self.__init__(X_t,0,w_bits)
        X_f_model=20*np.log10(abs(self.full_fft().real*2**-x_bits))  #model fft on quantized data
        X_f_model[:int(len(X_f)/2)]=0           # cut out region of interest like in xilinx datasheet
        X_f_model[(int(len(X_f)/2)+int(len(X_f)/20)):]=0
        if plot:
            plt.rc('font', size=18)
            plt.figure(1,[20,10])
            plt.title('Slot noise performance:')
            plt.plot(X_f_float,label='ideal')            
            plt.plot(X_f_model,label='model')
            plt.legend()
            plt.grid()
            plt.show()
        
        return X_f_model

    
    def evaluate_tone(self,size,x_bits,w_bits,plot=True):
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
        plot :  bool, optional
            plot out setting

        Returns
        -------
        None.

        """
        X_t=np.exp(3j*np.linspace(0,2*np.pi,size,False))
        X_t=((X_t+np.random.normal(0,(2**-x_bits)/np.sqrt(12),size))*2**x_bits)     #add textbook sine qunatization noise
        X_t=np.rint(X_t.real)+1j*np.rint(X_t.imag)          #quantize to nr bits
        X_f_float=20*np.log10(abs(np.fft.fft(X_t)*2**-x_bits)/size)     #ideal fft on quantized data
        #fft_mod=fft_model(X_t,0,w_bits)         # make new model inside eval for convenience
        self.__init__(X_t,0,w_bits)
        X_f_model=20*np.log10(abs(self.full_fft()*2**-x_bits)/size)  #model fft on quantized data
        if plot:
            plt.rc('font', size=18)
            plt.figure(1,[20,10])
            plt.title('Single tone performance:')
            plt.plot(X_f_float.real,label='ideal')            
            plt.plot(X_f_model.real,label='model')
            plt.legend()
            plt.grid()
            plt.show()
        
        

a=fft_model([.1,.1,.1,.1,.1,.1,.1,.1],6,18)
# eval not yet with scaling
a.evaluate_slot(1024,16,8)
a.evaluate_tone(1024,16,8)      # very low twiddle precision! noise concentrates at harmonics
a.evaluate_tone(1024,16,18)     # 18 fractional twiddle bits, i think I can build this with 18b wide rom
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        