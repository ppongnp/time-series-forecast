import warnings                                  # `do not disturbe` mode
warnings.filterwarnings('ignore')

import numpy as np                               # vectors and matrices
import pandas as pd                              # tables and data manipulations
import matplotlib.pyplot as plt                  # plots
import seaborn as sns                            # more plots
from pandas import datetime
from dateutil.relativedelta import relativedelta # working with dates with style
from scipy.optimize import minimize              # for function minimization
from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
import statsmodels.formula.api as smf            # statistics and econometrics
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import warnings 
warnings.filterwarnings("ignore") 
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error,median_absolute_error
import scipy.stats as scs
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.ar_model import AR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from math import log,sqrt

from keras.models import Sequential
from keras.layers import Dense, Dropout,LSTM
from keras.optimizers import Adam
import itertools                    # some useful functions
from tqdm import tqdm_notebook

def ADF_Stationarity_Test(timeseries,showResult=False):
    #Dickey-Fuller test:
    adfTest = adfuller(timeseries,autolag='AIC')
    pValue = adfTest[1]
    significanceLV = 0.05
    dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
    #Add Critical Values
    for key,value in adfTest[4].items():
        dfResults['Critical Value (%s)'%key] = value
    if showResult:
        print('Augmented Dickey-Fuller Test Results:')
        print(dfResults)
    return dfResults

def get_stationarity(dataset,showResult=False):
    run = True
    stationnary_set = dataset
    temp = dataset
    time = 0
    while(run):
        temp = stationnary_set.diff(periods=1)
        temp = temp[1:]
        stationnary_set = temp
        time += 1
        adf = ADF_Stationarity_Test(stationnary_set)
        if(adf['ADF Test Statistic'] < adf['Critical Value (1%)'] and adf['P-Value'] < 0.05):
            run = False
            if showResult:
                print("difference time = ", time)
                print("ADF test")
                print(adf)
            return stationnary_set

def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

def get_NAV_dataset(path):
    check = datetime.strptime('2018-1-1','%Y-%m-%d')
    nav = pd.read_csv(path,index_col=2,parse_dates=[2],date_parser=parser)
    del nav['amount']
    del nav['fund_code']
    for index in nav.index: 
        if index < check:
            nav = nav.drop([index])
    return nav

def ARIMA_predict(dataset):
    x = dataset.values
    num_train_set = round(len(x) * 0.75)
    num_test_set = round(len(x) * 0.25)
    train_set = x[0:num_train_set]
    test_set = x[num_train_set-1:]
    history = [x for x in train_set]
    prediction = list()
    residual = []
    for t in range(len(test_set)):
        warnings.filterwarnings("ignore")
        model = ARIMA(history,order=(1,0,2))
        model_fit = model.fit(disp=0,transparams=False)
        output = model_fit.forecast()
        yhat = output[0]
        prediction.append(yhat)
        obs = test_set[t]
        history.append(obs)
        residual.append(obs-yhat)
        #print('predicted=%f, expected=%f' % (yhat, obs))

    error0 = mean_absolute_error(test_set,prediction)
    print('test MAE: ', error0)
    error = mean_squared_error(test_set, prediction)
    print('Test RMSE: ' , error)
    error2 = MAPE(test_set,prediction)
    print('Test MAPE: ',error2)
    plt.plot(test_set)
    plt.plot(prediction,color='red')
    sum1 = 0
    for item in residual:
        sum1 += item
    mean11 = sum1/len(residual)
    print("mean = ",mean11)
    residual1 = pd.DataFrame(residual)
    residual1.hist()
    residual1.plot.kde()
    residual1.plot()
    #plt.plot(mean11,color='red')
    plot_acf(residual1)
    plt.show()


def find_ARIMA_order(dataset):
    stepwise_model = auto_arima(dataset, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
    print(stepwise_model.aic())

# this is not Stationary ---> mean,var,covar is not constrant over period
nav = get_NAV_dataset('dataset/NAV-SI-TDEX.csv')

dataset = nav.values
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

train_size = int(len(dataset) * 0.70)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

model = Sequential()
model.add(LSTM(50, input_shape=(train_size, 1)))
model.add(Dense(test_size))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions

plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()