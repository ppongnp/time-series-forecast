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

from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error
from math import log

import itertools                    # some useful functions
from tqdm import tqdm_notebook


def parser(x):
    return datetime.strptime(x,'%Y-%m-%d')

def get_NAV_dataset(path):
    check = datetime.strptime('2018-1-1','%Y-%m-%d')
    nav = pd.read_csv(path,index_col=2,parse_dates=[2],date_parser=parser)
    del nav['amount']
    del nav['fund_code']
    for index in nav.index: 
        if index < check:
            nav = nav.drop([index])
    return nav

nav = get_NAV_dataset('dataset/NAV-SI-TDEX.csv')
nav.head()