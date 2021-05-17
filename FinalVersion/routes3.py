#!/usr/bin/env python
# coding: utf-8

# In[11]:


import flask
from flask import Flask, request, render_template,redirect,url_for,send_file
import yfinance as yf

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from scipy.stats import norm
from scipy.optimize import bisect, brentq

import datetime
from datetime import datetime, date

from mpl_toolkits import mplot3d
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


import io
import random
from flask import Response
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import base64

app = Flask(__name__)

class Dynamics:
    pass

class Contract:
    pass

class FD:
    pass
        
    
def BScallPrice(sigma,S,rGrow,r,K,T):  # function for calculating Implied Vol using Black Scholes model
    F=S*np.exp(rGrow*T)
    sd = sigma*np.sqrt(T)
    d1 = np.log(F/K)/sd+sd/2
    d2 = d1-sd
    return np.exp(-r*T)*(F*norm.cdf(d1)-K*norm.cdf(d2))

def IVofCall(C,S,rGrow,r,K,T):                 # getting each implied volatility

    
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

def nan_helper(y):                          # function for helping interpolate missing values in implied vol
    return np.isnan(y), lambda z: z.nonzero()[0]

def get_vol_surface_2(symbol,S,r):
    quote=yf.Ticker(symbol)
    
    if quote.info['quoteType']=='CRYPTOCURRENCY':
        return "No options of it exist. Please choose another symbol."
    
    option_dates = list(quote.options)
    start_day_index = round(len(option_dates)/4)
    end_day_index = start_day_index*3

    option_dates = option_dates[start_day_index:end_day_index]      # using a middle slice of option dates 
    today = date.today()
    Ts=[((datetime.strptime(a, '%Y-%m-%d').date())-today).days/252 for a in option_dates]
    #print(option_dates)
    
    strike_len=len(quote.option_chain(option_dates[0]).calls['strike'])
    K_range=quote.option_chain(option_dates[0]).calls['strike'][round(strike_len/4):round(strike_len*3/4)].to_numpy()
    #print(K_range)
    all_vols=[]
    for ind_T in range(len(option_dates)):
        df=quote.option_chain(option_dates[ind_T]).calls
        
        if df.shape[0]==0:                            #if system says empty dataframe, go to the next iteration
            #print(option_dates[ind_T]+ "is skipped")
            continue
            
        call_prices=[np.nan]*len(K_range)           
        imp_vols_for_Ti=[np.nan]*len(K_range)
        
        for ind_K in range(len(K_range)):
            #print(option_dates[ind_T])
            index = df.index[df['strike'] == K_range[ind_K]].tolist()

            if index != [] :
                call_prices[ind_K] = quote.option_chain(option_dates[ind_T]).calls['lastPrice'][index[0]]

                imp_vols_for_Ti[ind_K] = IVofCall(C=call_prices[ind_K],S=S,rGrow=r,r=r,K=K_range[ind_K],T=Ts[ind_T])
        #print(imp_vols_for_Ti)
        
        #interpolate missing values linearly
        imp_vols_for_Ti=np.array(imp_vols_for_Ti)
        nans,x=nan_helper(imp_vols_for_Ti)
        imp_vols_for_Ti[nans]= np.interp(x(nans), x(~nans), imp_vols_for_Ti[~nans])
        imp_vols_for_Ti=imp_vols_for_Ti.round(4) 
        
        all_vols.append(imp_vols_for_Ti)
    
    Ts = np.array(Ts)        
    all_vols = np.row_stack(all_vols)
    #print(all_vols)
    
    return Ts, K_range, all_vols           # returning all the T, K and implied volatilities for plotting surface


def pricer_option_bs_CrankNicolson(contract,dynamics,FD):   # function for calculating call & call spread
                                                            # using Crank-Nicolson method and Black-Scholes model
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


def historical_month_vol(symbol):                # historical volatility in the past month calculation
    period='1mo'
    interval='1d'
    quote = yf.Ticker(symbol)
    hist=quote.history(period=period, interval=interval)
    hist['Close']=hist['Close'].dropna()
    daily_return=hist['Close'].pct_change()
    hist_vol=daily_return.std()*np.sqrt(252)
    
    return hist_vol


@app.route('/')
def my_form():                                  # getting user input of symbol, T and K
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
        
        dynamics_bs=Dynamics()                     # getting inputs ready for call & call spread calculation
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
        call_option=sum(call_prices_near_S0)/len(call_prices_near_S0)          #getting expected call_price
        put_option=sum(put_prices_near_S0)/len(put_prices_near_S0)             #getting expected put_price
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
        
        # making volatility surface
        
        Ts , Ks , Vols =  get_vol_surface_2(symbol=symbol , S=current_price,  r=dynamics_bs.r)
        fig = create_figure(Ts,Ks,Vols)
        # Convert plot to PNG image
        pngImage = io.BytesIO()
        #FigureCanvas(fig).print_png(pngImage)
    
        # Encode PNG image to base64 string
        # pngImageB64String is "data:image/png;base64, {result2}"
        
        plt.savefig(pngImage, format='png')
        pngImage.seek(0)  
        figdata_png = base64.b64encode(pngImage.getvalue())
        
        return render_template('price.html', result=result, result2=figdata_png.decode('utf8')) 
    
    else:
        print("GET is triggered...")
        
        return {"status": "check the function"}


def create_figure(Ts,Ks,Vols):           # creating the volatility surface plot
 
    T_, K_ = np.meshgrid(Ts, Ks)

     
    Z=Vols.reshape(len(Ts),len(Ks)).T
    fig = Figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(K_, T_, Z, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
    ax.set_title('Volatility Surface');
    ax.set_xlabel('K')
    ax.set_ylabel('T')
    ax.set_zlabel('Vol');
    
    return fig

if __name__ == '__main__':
    app.run(host='localhost', port=5000)
    

# @app.route('/',methods=['POST'])


# In[ ]:




