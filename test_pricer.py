#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

class Dynamics:
    pass

dynamics_bs=Dynamics()
dynamics_bs.volcoeff = 0.270271
dynamics_bs.alpha = 0
dynamics_bs.r = 0.0169
dynamics_bs.S0 = 124.58

class Contract:
    pass

contract_bs=Contract()
contract_bs.T = 14/252
contract_bs.K = 145

class FD:
    pass


FD_bs=FD()
FD_bs.SMax=150
FD_bs.SMin=50
FD_bs.deltaS=0.1
FD_bs.deltat=14/252/10000



# AAPL 5.21 put K=130, price=5.48


# In[2]:


def pricer_put_CEV_CrankNicolson(contract,dynamics,FD): # volcoeff = option value (or can get through implied vol using BS),
    #r=(yf.Ticker("^TNX").info['regularMarketPrice'])/100,deltat=T/10000,
    #T=user input? or initiate some , K=user input., 
    #,alpha=0
    #deltaS=0.1 or 0.01?, Smin=Underlying /3, Smax=Underlying *2 
    

    volcoeff=dynamics.volcoeff
    alpha=dynamics.alpha
    r=dynamics.r

    T=contract.T
    K=contract.K

    SMax=FD.SMax
    SMin=FD.SMin
    deltaS=FD.deltaS
    deltat=FD.deltat
    N=round(T/deltat)
    
    print(abs(N-T/deltat))
    if abs(N-T/deltat)>1e-12:
        raise ValueError('Bad time step')
    numS=round((SMax-SMin)/deltaS)+1
    if abs(numS-(SMax-SMin)/deltaS-1)>1e-12:
        raise ValueError('Bad time step')
    S=np.linspace(SMax,SMin,numS)   
    S_lowboundary=SMin-deltaS

    putprice=np.maximum(K-S,0)

    ratio=deltat/deltaS
    ratio2=deltat/deltaS**2
    f = 0.5 * volcoeff**2 * S**(2*(1+alpha))   
    g = 0   
    h = -r   
    F = 0.5*ratio2*f+0.25*ratio*g
    G = ratio2*f-0.50*deltat*h
    H = 0.5*ratio2*f-0.25*ratio*g
    
    RHSmatrix = diags([H[:-1], 1-G, F[1:]], [1,0,-1], shape=(numS,numS), format="csr")
    LHSmatrix = diags([-H[:-1], 1+G, -F[1:]], [1,0,-1], shape=(numS,numS), format="csr")

    for t in np.arange(N-1,-1,-1)*deltat:
        rhs = RHSmatrix * putprice
        
        rhs[-1]=rhs[-1]+2*H[-1]*(K-S_lowboundary)

        putprice = spsolve(LHSmatrix, rhs)  
        putprice = np.maximum(putprice, K-S)
    
    return(S, putprice)


# In[3]:


(S0_all_bs, putprice_bs) = pricer_put_CEV_CrankNicolson(contract_bs,dynamics_bs,FD_bs)


# In[4]:



displayStart = dynamics_bs.S0-FD_bs.deltaS*1.5 
displayEnd   = dynamics_bs.S0+FD_bs.deltaS*1.5
displayrows=np.logical_and(S0_all_bs>displayStart, S0_all_bs<displayEnd)
np.set_printoptions(precision=4, suppress=True)


# In[5]:


print(np.stack((S0_all_bs, putprice_bs),1)[displayrows])


# In[6]:


def IVofCall(C,S,rGrow,r,K,T):
    F=S*np.exp(rGrow*T)
    lowerbound = np.max([0,(F-K)*np.exp(-r*T)])
    if C<lowerbound:
        return np.nan
    if C==lowerbound:
        return 0
    if C>=F*np.exp(-r*T):
        return np.nan 
    hi=0.2
    while BScallPrice(hi,S,rGrow,r,K,T)>C:
        hi=hi/2
    while BScallPrice(hi,S,rGrow,r,K,T)<C:
        hi=hi*2
    lo=hi/2
   
    
    impliedVolatility = bisect(lambda x: BScallPrice(x,S,rGrow,r,K,T)-C, lo, hi)    
    return impliedVolatility


# In[ ]:




