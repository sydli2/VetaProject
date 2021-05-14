#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#This block is the main app
# creating API

#!pip install yfinance



import flask
from flask import Flask, request, render_template,redirect,url_for
import yfinance as yf

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

app = Flask(__name__)

        
def call(price):
    return price+10000

@app.route('/', methods=['GET','POST'])
def my_form():
    if request.method == 'POST':
        symbol=request.form['text']
        return redirect(url_for('post_underlying',symbol=symbol))
    return render_template('index.html')

@app.route('/price/<string:symbol>')
def post_underlying(symbol):
    result=[]
    current_price=yf.Ticker(symbol).info['open']
    call_option=pricer_put_CEV_CrankNicolson(current_price)
    str1="The opening price today is " + str(current_price)
    str2="The call option value is " +str(call_option)
    result.append(str1)
    result.append(str2)

    return render_template('price.html',result=result) #' '.join([str(elem) for elem in result])


# @app.route('/',methods=['POST'])
# def vol_surface():
    
#     return redirect(url_for('my_form'))

if __name__ == '__main__':
    app.run()
    


# In[ ]:


#yf.Ticker("AAPL").option_chain('2021-07-16').puts# this line is used to see the keys.


# In[ ]:


#THIS IS JUST FOR TESTING AND LEARNING

@app.route("/histInfo")                  # create a route
def HistoryInfo():
    
    symbol = request.args.get('symbol', default="MSFT")
    period = request.args.get('period', default="1y")             ## valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    interval = request.args.get('interval', default="1mo")        ## valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    quote = yf.Ticker(symbol)   
    hist = quote.history(period=period, interval=interval)
    data = hist.to_json()
    return data

@app.route("/current")
def CurrentInfo():
    
    symbol = request.args.get('symbol', default="MSFT")
    quote = yf.Ticker(symbol)
    current = quote.info
    return current

#@app.route('/')
#def home():
#   return render_template('home.html')

if __name__ == '__main__':
    app.run()
    
# use http://localhost:5000/histInfo?symbol=AAPL&interval=3mo


# In[ ]:



#THIS IS JUST FOR TESTING AND LEARNING

import flask
from flask import Flask, request, render_template
import yfinance as yf

app = Flask(__name__)

@app.route("/basic-template")
def basicTemplate():
    return render_template('basic-template.html')

@app.route("/")
@app.route("/setup")
def firstPage():
    return render_template('part-one.html')

@app.route("/punchline")
def secondPage():
    return render_template("part-two.html")

if __name__ == '__main__':
    app.run()


# In[ ]:


# version 1 just for testing.
import flask
from flask import Flask, request, render_template
import yfinance as yf

app = Flask(__name__)

def call(price):
    return price+10000

@app.route('/')
def my_form():

    return render_template('index.html')

@app.route('/', methods=['POST'])
def post_underlying():
    
        input_symbol = request.form['text']
        result=[]
        current_price=yf.Ticker(input_symbol).info['open']
        call_option=call(current_price)
        str1="The opening price today is " + str(current_price)
        str2="The call option value is " +str(call_option)
        result.append(str1)
        result.append(',')
        result.append(str2)
    #render = render_template("result.html")
    #return render_template("result.html")
        return ' '.join([str(elem) for elem in result])


# @app.route('/',methods=['POST'])
# def vol_surface():
    
#     return redirect(url_for('my_form'))

if __name__ == '__main__':
    app.run()


# In[ ]:


yf.Ticker("^TNX").info['regularMarketPrice']


# In[ ]:




