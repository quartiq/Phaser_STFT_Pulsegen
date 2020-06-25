#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# basic radix2 division in time fft 

# debugging this is pain...


import numpy as np
import matplotlib.pyplot as plt


def reverseBits(num,bitSize): 
     binary = bin(num) 
     reverse = binary[-1:1:-1] 
     return int(reverse + (bitSize - len(reverse))*'0',2)

def bfft(X):
    """
    basic radix2 division in time Cooley Tukey fft algorithm 
    no optimizations or anything, just to understand whats going on
    
    
    Parameters
    ----------
    X : input vector. has to be a power of two.

    Returns
    -------
    Y : DFT of X

    """
    bfls=int(len(X)/2)   # number butterflies per stage
    
    
    stages=int(np.log2(len(X)))

    
    t=np.exp(-2j*(np.pi/len(X))*np.arange(len(X)/2)) # only uses half circle twiddles
    
     # bit reverse order
    Y=np.zeros(len(X),'complex')
    for i,x in enumerate(X):
        irev=reverseBits(i,stages)
        Y[i]=X[irev]
    
    
    Y1=np.zeros(len(X),'complex')
    for s in range(stages):
        
        q = (1 << s) - 1  # lower bit stage mask
        
        for i in range(bfls):   
            
            t_s=bfls>>s             # twiddle index step size
            idxt=(t_s*i)%bfls     # twiddle factor index for each stage. wraps around.
            
            q=(1<<(s))-1            # lower bits bitmask ie 000000000011 for s=3. responsible for consecuteve parts
            #q_n=~q                  # higher bits bitmask ie 111111111100 for s=3. responsible for the jumps
            idx=(((i & ~q) << 1) | (i & q)) + (1<<s)  # the + is crap, i have to understand this better
            temp=Y[idx]*t[idxt] # temporary multiplication of second btfl input with twiddle
            Y1[idx-(1<<s)]=Y[idx-(1<<s)]+temp
            Y1[idx]=Y[idx-(1<<s)]-temp
            lala=1
        Y=np.copy(Y1)
            
    
    return Y
        

N=16

periods=5

cos=np.cos(2*np.pi*(periods/N)*np.arange(0,N))+1j
        
out=bfft(cos)

plt.plot(out.real)
plt.plot(out.imag)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        