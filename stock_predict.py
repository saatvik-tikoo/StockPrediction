import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM


#Get data and remove unnecessary fields
def data_fetch(stock_name):
  url = 'data/{}.csv'.format(stock_name)
  stocks = pd.read_csv(url, header=0) 
  df = pd.DataFrame(stocks)
  df.drop(df.columns[[0,3,5,6,7,8,9,10,11,12]], axis=1, inplace=True)
  df.to_csv('data/{}_modified.csv'.format(stock_name))
  return df

#Normalize the data
def normalization(df):
  df['Open'] = df['Open']/100
  df['High'] = df['High']/100
  df['Close'] = df['Close']/100
  return df

#Load the data
def load_data(stock, seq_len):
  number_of_features = len(stock.columns)
  data = stock.values
  print(data)
  sequence_length = seq_len + 1
  result = []
  for index in range(len(data) - sequence_length):
    result.append(data[index: index + sequence_length])

  print(result)
  result = np.array(result)
  row = round(0.8 * result.shape[0])
  train = result[:int(row), :]
  x_train = train[:, :-1]
  y_train = train[:, -1][:,-1]
  x_test = result[int(row):, :-1]
  y_test = result[int(row):, -1][:,-1]

  x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], number_of_features))
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], number_of_features))  

  return [x_train, y_train, x_test, y_test]

#Build the model
def build_model(layers):
  d = 0.2
  model = Sequential()
  model.add(LSTM(128, input_shape=(layers[1], layers[0]), return_sequences=True))
  model.add(Dropout(d))
  model.add(LSTM(64, input_shape=(layers[1], layers[0]), return_sequences=False))
  model.add(Dropout(d))
  model.add(Dense(16,init='uniform',activation='relu'))        
  model.add(Dense(1,init='uniform',activation='linear'))
  model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
  return model

if __name__ == '__main__':
  stock_name = 'GOOGL'
  df = data_fetch(stock_name)
  print(df.head())
  print('_________________________DATA FETCH COMPLETED__________________________')
  df = normalization(df)
  print(df.head())
  print('_________________________NORMALIZATION COMPLETED_______________________')
  window = 22
  X_train, Y_train, X_test, Y_test = load_data(df[::-1], window)
  print("X_train", X_train.shape)
  print("y_train", Y_train.shape)
  print("X_test", X_test.shape)
  print("y_test", Y_test.shape)
  print('_________________________DATA DISTRIBUTION DONE________________________')
  model = build_model([3,window,1])
  print('_________________________MODEL BUILD DONE______________________________')
  model.fit(X_train, Y_train, batch_size=512, nb_epoch=500, validation_split=0.1, verbose=1)
  print('_________________________MODEL TRAINING DONE___________________________')

  trainScore = model.evaluate(X_train, Y_train, verbose=0)
  print('Train Score: %.2f MSE (%.2f RMSE)' % (trainScore[0], math.sqrt(trainScore[0])))

  testScore = model.evaluate(X_test, Y_test, verbose=0)
  print('Test Score: %.2f MSE (%.2f RMSE)' % (testScore[0], math.sqrt(testScore[0])))

  print('_________________________PLOTTING RESULTS___________________________')
  plt.plot(Y_train,color='red', label='prediction')
  plt.plot(Y_test,color='blue', label='Test Set Result')
  plt.legend(loc='upper left')
  plt.show()
  

  
