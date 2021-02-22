# Momentum_Model

## Introduction
#### In this project, I will be using Deep Learning and LSTM, a type of Recurrent Neural Network, to predict future stock market returns and use those predictions to construct a portfolio that rebalances on a weekly basis.

## Outline
* [Introduction](##Introduction)  
* [Data](##Data)  
* [Model](##Model)  
* [Trading Strategy and Portfolio Rebalancing](##Trading_Strategy)

## Data
#### All the historical stock price data will be collected using the yahoo finance API (yfinance). However, since there is no market screener API to automatically get results based on  specific criteria, I download a CSV file from the Nasdaq website containing the largest publicly traded companies and from those, I select the largest 30 to use in the model. Moreover, since I am interested in asset performance, I will convert prices to weekly returns. A weekly interval is preferred to a daily interval, as it better captures the general trend of the market and can be more reliable when making predictions.

```python

# Read the CSV file and keep only the largest 30 companies
df = pd.read_csv('nasdaq_screener_1613878479181.csv')
stocks = df[~(df['IPO Year']>2020)].dropna().sort_values(by='Market Cap')['Symbol'].iloc[-30:].tolist()
stocks.sort()

# Each element of the list is a dataframe with the Adjusted close price of each ticker.
securities_by_date = []
for i in stocks:
    df = yf.download(i,start= '2008-01-01',end=None , interval = '1wk')[['Adj Close']].dropna().pct_change(periods = 5).dropna().rename(columns={'Adj Close':i})
    securities_by_date.append(df)

```

## Model



#### It is very important to scale features before training a neural network, so before training the model using LSTM, we must normalize the data sets. To achieve that, I split the data into training 70% and testing 30% and subtract from both the mean and divide by the standard deviation. The mean and standard deviation should only be from the training set so that the model has no access to the values in the test set. 

```python
class Momentum_Model:
    
    def __init__(self,df, window, future):
        
        self.df, self.window, self.future, = df, window, future
        
        # Split the data to train and test
        train_df = self.df[0:int(len(self.df)*0.7)]
        test_df = self.df[int(len(self.df)*0.7):]
        # Get means and std for the columns of the dataframe
        train_mean = train_df.mean()
        train_std = train_df.std()
        # Normalize the data by subtracting the mean and std of the train set
        train_df = (train_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std
        # Convert the normalized data to numpy array.
        train_df, test_df = train_df.to_numpy(), test_df.to_numpy()
        
        self.train_df, self.test_df, self.train_mean, self.train_std = train_df, test_df, train_mean, train_std 

```
#### After normalizing the data, the next step is to reshape them in order to fit LSTM models. The LSTM input layer has three dimensions (samples, timesteps, features). There is only one feature (weekly stock return) and the number of samples is the length of the data set. Moreover, for all samples in the model we use the 5 previous observations in order to predict the 6th, which means that the ```X_train``` , ```X_test``` sets will be five-dimentional np.array with input shapes ```(len(X_train),5,1)``` , ```(len(X_test),5,1)```  and ```y_train``` , ```y_test```  will have input shapes ```(len(y_train),1)``` , ```(len(y_test),1)``` .

```python
def layout(self):
    # Train Layout
    X_train = []
    y_train = []
    for i in range(self.window,len(self.train_df) - self.future):
        X_train.append(self.train_df[i - self.window:i])  
        y_train.append(self.train_df[i + self.future,0:1])  
        
    X_train, y_train = np.array(X_train),np.array(y_train)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],self.train_df.shape[1])
        
    # Test Layout
    X_test = []
    y_test = []
    for i in range(self.window,len(self.test_df)-self.future):
        X_test.append(self.test_df[i - self.window:i])
        y_test.append(self.test_df[i + self.future,0:1])
            
    X_test, y_test = np.array(X_test),np.array(y_test)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], self.test_df.shape[1])
        
    return X_train, y_train, X_test, y_test
        
        
  ```

#### As far as the model is concerned, I will be using a stacked LSTM model comprised of one input layer, three LSTM layers and an output layer (which will be a dense layer with only one output). The first LSTM layer will have 50 units and the second and third will have 30 units. For both the first and the second layers ```return_sequences=True ``` because we want the internal state of the previous layer to pass onto the next layers. Also, I will be using a 20% dropout rate so that the model focuses on more recent data.

```python

def Rnn_model(self, X):
        
        self.X = X
        
        model = Sequential()
        model.add(LSTM(units=50,activation = 'tanh' , return_sequences=True, input_shape=(self.X.shape[1], self.X.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 30,activation ='sigmoid', return_sequences = True))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 30,activation ='sigmoid', return_sequences = False))
        model.add(Dense(1))
        
        return model
        
```

        
#### For the training of the model, I will be using the Adam algorithm as an optimizer with a learning rate of .001 and a decay of 1e-6. Also, when fitting the data, I will use 100 epochs along with a batch size of 30. Finally, once training is complete and we have predicted the values of the test set, we must rescale the predictions by multiplying the standard deviation and adding the mean of the train set. We do this because we want to establish a trading strategy that picks stocks based on higher performance; so, if we do not rescale the features, we will be unable to distinguish which stocks were performing better since each stock has different means and standard deviations.

```python
def predictions(self, model, epochs, X_train, y_train, X_test, y_test):
    self.model, self.epochs, self.X_train, self.y_train, self.X_test, self.y_test  = model, epochs, X_train, y_train, X_test, y_test
    opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)
    self.model.compile(optimizer = opt, loss = 'mean_squared_error')
    self.model.fit(self.X_train, self.y_train, epochs = self.epochs, batch_size = 30, verbose = 0)
        
    predictions = self.model.predict(self.X_test)*self.train_std.tolist()[0] + self.train_mean.tolist()[0]
        
    y_test_inverse_transform = self.y_test*self.train_std.tolist()[0] + self.train_mean.tolist()[0]
        
    return [predictions.flatten(), y_test_inverse_transform.flatten()]
    
    
```
#### Now that we have everything, we create a loop that runs the model and saves the predictions of future stock returns for all stocks.

```python

stock_predictions = []
for df in securities_by_date:  
    
    data = Momentum_Model(df,5,0)
    X_train, y_train, X_test, y_test = data.layout()
    volume_model = data.Rnn_model(X_train)
    predictions = data.predictions(volume_model, 100, X_train, y_train, X_test,y_test)
    stock_predictions.append(predictions)
    
```

#### Below you can see what the model predicts for Apple.
        
        
   





