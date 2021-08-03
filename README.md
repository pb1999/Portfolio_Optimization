# Momentum_Model

## Introduction
#### In this project, I will be using Deep Learning to predict future stock market returns and use those predictions to construct a portfolio with the top performing stocks that automatically rebalances on a weekly basis.



## Data
#### All the historical stock market data will be collected using the yahoo finance API (yfinance). However, since there is no market screener API to automatically get key market analytics, I download a CSV file from the Nasdaq website containing the largest publicly traded companies and from those, I select the largest 30 to use in the model. Moreover, since I am interested in asset performance, I will convert prices to weekly returns. A weekly interval is preferred to a daily interval, as it better captures the general trend of the market and can be more reliable when making predictions.

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


#### It is very important to scale features before training a neural network, so we must normalize the data sets. To achieve that, I split the data into training 70% and testing 30% and subtract from both the mean and divide by the standard deviation. The mean and standard deviation should only be from the training set so that the model has no access to the values in the test set.

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
#### After normalizing the data, the next step is to reshape them in order to fit our model. The LSTM input layer has three dimensions (samples, timesteps, features). There is only one feature (weekly stock return) and the number of samples is the length of the data set. Moreover, for all samples in the model we use the 5 previous observations in order to predict the 6th, which means that the ```X_train``` , ```X_test``` sets will be five-dimentional np.array with input shapes ```(len(X_train),5,1)``` , ```(len(X_test),5,1)```  and ```y_train``` , ```y_test```  will have input shapes ```(len(y_train),1)``` , ```(len(y_test),1)``` .

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
### Model Architecture:
#### I will be using a stacked LSTM model comprised of one input layer, three LSTM layers and an output layer (which will be a dense layer with only one output). The first LSTM layer will have 50 units and the second and third will have 30 units. For both the first and the second layers ```return_sequences=True ``` because we want the internal state of the previous layer to pass onto the next layers. Also, I will be using a 20% dropout rate so that the model focuses on more recent data.

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

        
#### Moreover, I will be using the Adam algorithm as an optimizer with a learning rate of .001 and a decay of 1e-6. Once training is complete and we have predicted the values of the test set, we must rescale the predictions by multiplying the standard deviation and adding the mean of the train set. We do this because we want to establish a trading strategy that picks stocks based on higher performance; so, if we do not rescale the features, we will be unable to distinguish which stocks were performing better since each stock has different means and standard deviations.

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

#### For reference, below you can see what the model predicts for Apple.

![alt text](https://github.com/pb1999/Momentum_Model/blob/main/Prediction_Apple.PNG)
        
 
## Trading Strategy and Portfolio Rebalancing

#### We already saved the predictions of the model for each stock. The next step is to combine all the predictions in one dataset and keep only the top 10. After that, at each timestep (week), we implement a simple trading algorithm of buying the open market price of the top 10 stocks today and selling the open market price one week from today. Therefore, we create a portfolio comprised of 10 stocks (each stock has equal weight) that rebalances on a weekly basis. Finally, we assume zero transaction costs when buying or selling securities. 

```python 

prediction_data = [stock_predictions[i][0].tolist() for i in range(len(stock_predictions))]

list_of_predictions = []
dates = pd.date_range(start='1/1/2008', end=datetime.today().strftime('%Y-%m-%d')).tolist()[::7]
for i in range(len(prediction_data)):
    
    df = pd.DataFrame(prediction_data[i]).rename(columns={0:stocks[i]})
    df['Date'] = dates[-len(df.index):]
    list_of_predictions.append(df.set_index('Date'))
    
df_predictions = reduce(lambda left,right: pd.merge(left,right,on='Date', how = 'outer'), list_of_predictions).fillna(-1000)

column_names = ['10th','9th', '8th', '7th', '6th', '5th', '4th', "3rd", "2nd", "1st"]
    
performance = pd.DataFrame(columns = column_names)
for i in range(len(df_predictions)):
    performance.loc[len(performance)] = df_predictions.iloc[i].T.sort_values().iloc[-10:].reset_index()['index'].tolist()
    
performance['Date'] = dates[-len(performance.index):]

```
 |  10th |   9th  | 8th  | 7th  | 6th |  5th |  4th |  3rd |  2nd |  1st  |  Date      |
 | ----- | ------ | ---- | ---- | --- | ---- | ---- | ---- | ---- | ----- | ---------- |     
 | SHOP  | ABBV   | QCOM |  TSM | JD  | AMGN |   SE |  PDD | ASML | TSLA  | 2021-01-19 |
 |  MSFT | GOOG   | AMGN | QCOM | TSM |  HDB | ASML |  JD  | TSLA |  SE   | 2021-01-26 |
 |   TSM |   JD   | HDB  |ASML  | MSFT| NVDA | TSLA | GOOG | BABA |  SE   |2021-02-02  |
 |  CRM  |  ASML  |  SHOP| NFLX | NVDA| TSLA |  MSFT| GOOG | BABA |   SE  | 2021-02-09 |
 | HDB   | CRM    | MSFT | BABA |  ZM | NVDA |  GOOG|  SQ  |  SHOP|   SE  |2021-02-16  |


```python

portfolio_performance = []
portfolio_weights = np.array([1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10,1/10])

for i in range(len(performance.index) - 2):
    weekly_securities = yf.download(performance.iloc[i][0:10].tolist(),start= performance.iloc[i][10] ,end= performance.iloc[i+1][10], interval = '1wk')['Open']
    
    weekly_securities_2 = weekly_securities.reset_index().dropna().drop_duplicates(subset=['Date']).set_index('Date').pct_change().dropna().values.flatten()
    portfolio_performance.append(weekly_securities_2)
 
 final_performance = []
for j in portfolio_performance:
        
    try:
        final = j@portfolio_weights
        
    except TypeError:
        final = math.nan
        
    except ValueError:
        final = math.nan
        
    final_performance.append(final)
    

```

#### Finally, we compute the performance of the portfolio that uses this simple trading algorithm and compare it to the performance of SPY. 

![alt text](https://github.com/pb1999/Momentum_Model/blob/main/Portfolio_Performance.PNG)


