# Momentum_Model

## Introduction
#### In this project, I will be using Deep Learning and LSTM, a type of Recurrent Neural Network, to predict future stock market returns and use those predictions to construct a portfolio that rebalances on a weekly basis.

## Outline
* [Introduction](##Introduction)  
* [Data](##Data)  
* [Model](##Model)  
* [Trading Strategy and Portfolio Rebalancing](##Trading_Strategy)

## Data
#### All the stock price data will be collected using the yahoo finance API (yfinance). However, since there is no market screener API to automatically get results based on  specific criteria, I download a CSV file from the Nasdaq website containing the largest publicly traded companies by Market Cap. and select the largest 30 to use in the model. Moreover, since I am interested in asset performance, I will convert prices to weekly returns. A weekly interval is preferred to a daily interval, as it better captures the general trend of the market and can be more reliable when making predictions.

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

#### It is very important to scale features before training a neural network, so before training the model using LSTM, we must normalize the data set. To achieve that, I split the data into training 70% and testing 30% and subtract from both the mean and divide by the standard deviation. The mean and standard deviation should only be from the training set so that the model has no access to the values in the test set. 

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
#### After normalizing the data, the next step is to reshape them in order to fit LSTM models. The LSTM input layer has 3 dimensions (samples, timestamps, features). There is only one feature (5 week stock return) and the number of samples is the lenght of the data set (training/testing). Moreover, for all samples, in the model we use the 5 previous observations and we predict the 6th, which means that there are 5 timestamps. Therefore, input shapes of training and testing sets must be `(len(training),5,1) , (len(testing),5,1)`.

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







