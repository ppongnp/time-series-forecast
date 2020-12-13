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
from sklearn.metrics import mean_squared_error
import scipy.stats as scs
from pandas.plotting import autocorrelation_plot
from statsmodels.tsa.ar_model import AR

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error


from itertools import product                    # some useful functions
from tqdm import tqdm_notebook

def ADF_Stationarity_Test(timeseries, printResults = True):
    #Dickey-Fuller test:
    adfTest = adfuller(timeseries,autolag='AIC')
    pValue = adfTest[1]
    significanceLV = 0.05
    if printResults:
        dfResults = pd.Series(adfTest[0:4], index=['ADF Test Statistic','P-Value','# Lags Used','# Observations Used'])
        #Add Critical Values
        for key,value in adfTest[4].items():
             dfResults['Critical Value (%s)'%key] = value
        print('Augmented Dickey-Fuller Test Results:')
        print(dfResults)

def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=3).mean()
    rolling_std = timeseries.rolling(window=3).std()
    
    # rolling statistics plot
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Dickey–Fuller test:
    result = adfuller(timeseries['value'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))

def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')

def get_NAV_dataframe(path):
    check = datetime.strptime('2018-1-1','%Y-%m-%d')
    nav = pd.read_csv(path,index_col=2,parse_dates=[2],date_parser=parser)
    del nav['amount']
    del nav['fund_code']
    for index in nav.index: 
        if index < check:
            nav = nav.drop([index])
    return nav

# this is not Stationary ---> mean,var,covar is not constrant over period
nav = get_NAV_dataframe('time-series-forecast/dataset/NAV-SI-1AM-TG.csv')

nav_diff = nav.diff(periods=1)
nav_diff = nav_diff[1:]


second_diff = nav_diff.diff(periods=1)
second_diff = second_diff[1:]


third_diff = second_diff.diff(periods=1)
third_diff = third_diff[1:]


fourth_diff = third_diff.diff(periods=1)
fourth_diff = fourth_diff[1:]

fifth_diff = fourth_diff.diff(periods=1)
fifth_diff = fifth_diff[1:]

six = fifth_diff.diff(periods=1)
six = six[1:]

seven = six.diff(periods=1)
seven = seven[1:]

rolling_mean = second_diff.rolling(window=12).std()
df_log_minus_mean = second_diff - rolling_mean
df_log_minus_mean.dropna(inplace=True)

x = seven.values

train_num = round(len(x) * 0.8)
test_num = round(len(x) * 0.2)

train_set = x[0:train_num]
test_set = x[train_num-1:]

"""
history = [x for x in train_set]
prediction = list()
for t in range(len(test_set)):
    model = ARIMA(history,order=(1,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    prediction.append(yhat)
    obs = test_set[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))

error = mean_squared_error(test_set, prediction)
print('Test MSE: %.5f' % error)

forecast_errors = [test_set[i]-prediction[i] for i in range(len(test_set))]
bias = sum(forecast_errors) * 1.0/len(test_set)
print('Bias: %f' % bias)
"""
# plot
"""
plt.plot(test_set)
plt.plot(prediction,color='red')
plt.show()
"""
"""
tototo = []
for i in range(len(prediction)-1):
    result = test_set[i] - prediction[i]
    tototo.append(result)


#plt.plot(test_set)
#plt.plot(prediction,color='red')

df2 = pd.DataFrame (tototo,columns=['value'])
df2.plot()
get_stationarity(df2)
plt.show()
"""


"""
model = ARIMA(train_set,order=(1,1,1))
model_fit = model.fit()

predict = model_fit.predict(start=190,end=237)
plt.plot(test_set)
plt.plot(predict,color='red')

"""


"""
count = 0
for item in nav.updated:
    date_time_obj = datetime.datetime.strptime(item, '%Y-%m-%d')
    nav.updated[count] = date_time_obj.date()
    count += 1
"""

"""
rolling_mean = nav.value.rolling(window = 50).mean()
plt.plot(nav.updated,nav.value,color = 'blue', label = 'Original')
plt.plot(nav.updated,rolling_mean, color = 'red', label = 'Rolling Mean')

plt.title('Rolling Mean & Rolling Standard Deviation')
"""
