Aim: In this project we are try to predict the 'Close' value of a stock for a day.

Structure and Usage: 
1. Data folder contains all the data required. Here I have used quandl data-set for Google Stocks. We can change that to any other company by just changing the name.
2. Just run stock_predict.py. It will automatically do the following steps:
    a) Fetch the data.
    b) Pre-process the Data.
    c) Divide data into training and testing sets.
    d) Make a LSTM based model using KERAS API.
    e) Train and test the model.
    d) Plot the results