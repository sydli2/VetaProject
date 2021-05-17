# Welcome to the project readme!

## Description

This project contains an API and UI using Python Flask to price a call option and a call spread. It runs on your local html. Today's opening price of the underlying will be generated, along with the 3-D volatility surface.

This project is created for the Capability Test for Veta Investment Partners.


## Languages and Packages

Python, HTML

Flask, Jinja, yfinance, numpy, scipy, matplotlib, datetime, io, base64


## Content

**routes3.py**: a file that contains the python implementations of API, call and call spread pricing function, and the volatility surface function. There are some other pieces of functions, which are used to support the main functions just mentioned.  

**index.html**: an html file that serves as the homepage.  It allows user inputs of an underlying symbol, an expiration time, and a strike price.

**price.html**: an html that will return the calculated call and call spread value, and the calculated volatility surface.

**existence.html**: an html that deals with user input of invalid symbol (that does not come with an option, example: Bitcoin 'BTC-USD'), negative or zero expiration time and strike.


## Calculation Methods

**Black Scholes Model**: the implied volatility surface is obtained assuming this model. 

**Crank-Nicolson**: this method approximate option price with less computation efforts while maintaining accuracy, compared with the direct calculation using Black Scholes. It is an improvement over Implicit Finite Differences and Explicit Finite Differences method. 

**Binary Search**: binary search is used to find and optimize the result of implied volatility.

## Usage and Run

Run routes3.py on your Command Prompt and you should expect a link to the local html, where you will be asked to input symbol, expiration and strike.
If it does not run, please try and see if you have Flask, yfinance installed. If your input is not valid, it will allow you to re-enter a new one.
For Expiration T and Strike K inputs, any 0 entered before a number will be ignored.

## Repository

This repository contains 3 versions of the project. It records the progress made. However, only the final version should be used.

## Author

Xinyi (Sydney) Li

## Acknowledgement

I want to thank Trey Barton, the Senior Developer at Veta Investment Partners, for the directions and help on this project.
I have also referenced Colin Kraczkowsky's blog https://ckraczkowsky.medium.com/building-modern-user-interfaces-with-flask-23016d453792 when I was stuck on connecting API and UI.
