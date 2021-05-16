#!/usr/bin/env python
# coding: utf-8


import flask
from flask import Flask, request, render_template,redirect,url_for
import yfinance as yf

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

import datetime
from datetime import datetime, date

app = Flask(__name__)

class Dynamics:
    pass

class Contract:
    pass

class FD:
    pass
        

def pricer_option_bs_CrankNicolson(contract,dynamics,FD):

    volcoeff=dynamics.volcoeff
    r=dynamics.r

    T=contract.T
    K=contract.K

    SMax=FD.SMax
    SMin=FD.SMin
    deltaS=FD.deltaS
    deltat=FD.deltat
    N=round(T/deltat)

    if abs(N-T/deltat)>1e-12:
        raise ValueError('Bad time step')
    numS=round((SMax-SMin)/deltaS)+1
    if abs(numS-(SMax-SMin)/deltaS-1)>1e-12:
        raise ValueError('Bad time step')
    S=np.linspace(SMax,SMin,numS)   
    S_lowboundary=SMin-deltaS
    
    callprice=np.maximum(S-K,0)
    putprice=np.maximum(K-S,0)
    
    ratio=deltat/deltaS
    ratio2=deltat/deltaS**2
    f = 0.5 * volcoeff**2 * S**2   
    g = 0   
    h = -r   
    F = 0.5*ratio2*f+0.25*ratio*g
    G = ratio2*f-0.50*deltat*h
    H = 0.5*ratio2*f-0.25*ratio*g
    
    RHSmatrix = diags([H[:-1], 1-G, F[1:]], [1,0,-1], shape=(numS,numS), format="csr")
    LHSmatrix = diags([-H[:-1], 1+G, -F[1:]], [1,0,-1], shape=(numS,numS), format="csr")

    for t in np.arange(N-1,-1,-1)*deltat:
        rhs = RHSmatrix * callprice
        
        rhs[-1]=rhs[-1]+2*H[-1]*(K-S_lowboundary)

        callprice = spsolve(LHSmatrix, rhs)  
        callprice = np.maximum(callprice, S-K)
    
    for t in np.arange(N-1,-1,-1)*deltat:
        rhs = RHSmatrix * putprice
        
        rhs[-1]=rhs[-1]+2*H[-1]*(K-S_lowboundary)

        putprice = spsolve(LHSmatrix, rhs)  
        putprice = np.maximum(putprice, K-S)
        
    return(S, callprice, putprice)


def historical_month_vol(symbol):                #historical volatility in the past month calculation
    period='1mo'
    interval='1d'
    quote = yf.Ticker(symbol)
    hist=quote.history(period=period, interval=interval)
    hist['Close']=hist['Close'].dropna()
    daily_return=hist['Close'].pct_change()
    hist_vol=daily_return.std()*np.sqrt(252)
    
    return hist_vol


@app.route('/')
def my_form():
#symbol=symbol, T=T, K=K 
    return render_template('index.html')

@app.route('/price',methods=['Get','POST'])
def post_underlying():
    if request.method == 'POST':
        
        #print("Post is triggered...")
        
        symbol = request.form['symbol']
        T = float(request.form['expiration'])
        K = float(request.form['strike'])
        # checking edge case that the input symbol does not exist, or T not positive, or K not positive.
        if yf.Ticker(symbol).info['logo_url'] == '' or T<=0.0 or K<=0.0:   
            return render_template('existence.html') 
        
        current_price=yf.Ticker(symbol).info['open']
        
        dynamics_bs=Dynamics()                     #getting inputs ready for call & call spread calculation
        dynamics_bs.volcoeff = historical_month_vol(symbol)
        dynamics_bs.alpha = 0
        dynamics_bs.r = (yf.Ticker("^TNX").info['regularMarketPrice'])/100  # getting interest rate
        dynamics_bs.S0 = current_price                                    
        
        contract_bs=Contract()
        contract_bs.T = T
        contract_bs.K = K
        
        FD_bs = FD()
        FD_bs.SMax = round(current_price*2)
        FD_bs.SMin = round(current_price/3)
        FD_bs.deltaS = 0.1
        FD_bs.deltat = T/1000
        
        (S0_all_bs, callprice_bs, putprice_bs) = pricer_option_bs_CrankNicolson(contract_bs,dynamics_bs,FD_bs)

        displayStart = dynamics_bs.S0-FD_bs.deltaS*1.5 
        displayEnd   = dynamics_bs.S0+FD_bs.deltaS*1.5
        displayrows=np.logical_and(S0_all_bs>displayStart, S0_all_bs<displayEnd)
        np.set_printoptions(precision=4, suppress=True)
        
        # getting option prices near S0
        call_prices_near_S0 = [a[1] for a in np.stack((S0_all_bs, callprice_bs, putprice_bs),1)[displayrows]]
        put_prices_near_S0 = [a[2] for a in np.stack((S0_all_bs, callprice_bs, putprice_bs),1)[displayrows]]
        call_option=sum(call_prices_near_S0)/len(call_prices_near_S0)  #getting expected call_price
        put_option=sum(put_prices_near_S0)/len(put_prices_near_S0)  #getting expected put_price
        call_spread=call_option - put_option
        
        result=[]
        call_option=round(call_option,2)
        call_spread=round(call_spread,2)
        str1="The opening price today is $" + str(current_price)
        str2="The call option value with expiration " +str(T) + " and strike " + str(K) + " is $" +str(call_option)
        str3="The call spread value is $" + str(call_spread)
        result.append(str1)
        result.append(str2)
        result.append(str3)

        return render_template('price.html',result=result) 
    else:
        print("GET is triggered ...")
        
        return {"status": "check the function"}
    
    if request.method=="Get":
        return 'None'

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
    

# @app.route('/',methods=['POST'])






