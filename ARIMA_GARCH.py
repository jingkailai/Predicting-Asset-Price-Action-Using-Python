# Dependency
# regular package
import datetime
import time
import logging
from numpy.core.fromnumeric import shape

# data processing package
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from empyrical import sharpe_ratio

# statistics tools
from arch import arch_model # import arch_model.
from dateutil.relativedelta import relativedelta
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# ignore warnings
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning, HessianInversionWarning
warnings.simplefilter("ignore", ConvergenceWarning)
warnings.simplefilter("ignore", HessianInversionWarning)
warnings.simplefilter("ignore", RuntimeWarning)
warnings.simplefilter("ignore", FutureWarning)

# set the logger
logging.basicConfig(
                    filename = "Arima Garch log",
                    filemode = "w+",
                    format='%(name)s %(levelname)s %(message)s',
                    datefmt = "%H:%M:%S",
                    level=logging.ERROR)
logger = logging.getLogger("GARCH")


# plot function
def plot_pred_vs_truth(y_fit, y_train, y_pred, y_test, prediction, ticker):
    """
    y_pred: the predicted values of dependent variable.
    y_truth: the ground truth values of dependent variable.
    label1: the label1 is the lable for y_pred.
    label2: the label2 is the label for y_truth.
    effect: plot predicted values v.s. the ground truth values.
    """
    plt.figure(figsize=(16,8))
    plt.plot(y_fit[1:], color='red', label='fit')
    plt.plot(y_train[1:], color='black', label='train truth')
    plt.plot(y_pred, color='red', label='predict', linestyle=':')
    plt.plot(y_test, color='black', label='test truth', linestyle=':')
    plt.plot(prediction, color='red', linestyle=':')

    plt.legend(loc='best')
    plt.title("Truth v.s. Predict of {}".format(ticker))
    plt.savefig("./figure/arima_pred_{}.png".format(ticker))
    plt.close()

# fit & predict function
def model_predict(trend_arima_fit, residual_arima_fit, 
                  trend_garch_order, residual_garch_order,
                  trend, residual, seasonal, 
                  if_pred, start, end, period):
    """
    trend_arima_fit: ARIMA model after fit the trend.
    residual_arima_fit: ARIMA model after fit the residual.
    trend_garch_order: best parameters for GARCH model after fit the trend_arima_fit.resid.
    residual_garch_order: best parameters for GARCH model after fit the residual_arima_fit.resid.
    trend: time series of trend.
    residual: time series of residual.
    seasonal: time series of seasonal.
    if_pred: boolen value indicating whether to predict or not. True presents predict, False means fit.
    start: string value indicating start date.
    end: string value indicating end date.
    period: int value indicating the period of seasonal.
    return predicted sequence.
    """
    if if_pred:
        # get the first date after the last date in train.
        date_after_train = str(trend.index.tolist()[-1] + relativedelta(days=1))
        # get the trend predicted sequence from the start of start to end
        trend_pred_seq = np.array(trend_arima_fit.predict(start = date_after_train, 
                                                          end = end,
                                                          dynamic = True)) # The dynamic keyword affects in-sample prediction. 
        
        # get the residual predicted sequence from the start of start to end
        residual_pred_seq = np.array(residual_arima_fit.predict(start = date_after_train,
                                                                end = end,
                                                                dynamic = True))
        
        # find the the corresponding seasonal sequence.
        pred_period = (datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(date_after_train, '%Y-%m-%d %H:%M:%S')).days + 1
        
        trend_pred_variance, residual_pred_variance = np.zeros(pred_period), np.zeros(pred_period)
        current_trend_resid, current_residual_resid = trend_arima_fit.resid, residual_arima_fit.resid
        for i in range(pred_period):
            trend_model = arch_model(current_trend_resid,
                                     mean = "Constant",
                                     p = trend_garch_order[0], 
                                     q = trend_garch_order[1], 
                                     vol = 'GARCH',
                                     rescale=False)
            trend_model_fit = trend_model.fit(disp = "off",
                                              update_freq = 0,
                                              show_warning = False)
            trend_pred_variance[i] = np.sqrt(trend_model_fit.forecast(horizon = 1).variance.values[-1,:][0]) + trend_model_fit.forecast(horizon = 1).mean.values[-1,:][0]
            current_trend_resid = current_trend_resid.append(pd.Series({current_trend_resid.index.tolist()[-1] + relativedelta(days= 1): trend_pred_variance[i]}))

            residual_model = arch_model(current_residual_resid,
                                        mean = "Constant",
                                        p = residual_garch_order[0], 
                                        q = residual_garch_order[1], 
                                        vol = 'GARCH',
                                        rescale=False)
            residual_model_fit = residual_model.fit(disp = "off",
                                                    update_freq = 0,
                                                    show_warning = False)
            residual_pred_variance[i] = np.sqrt(residual_model_fit.forecast(horizon = 1).variance.values[-1,:][0]) + residual_model_fit.forecast(horizon = 1).mean.values[-1,:][0]
            current_residual_resid = current_residual_resid.append(pd.Series({current_residual_resid.index.tolist()[-1] + relativedelta(days= 1): residual_pred_variance[i]}))

        trend_pred_seq = trend_pred_seq + trend_pred_variance 
        residual_pred_seq = residual_pred_seq + residual_pred_variance

        trend_pred_seq = np.concatenate((trend.values, trend_pred_seq))
        residual_pred_seq = np.concatenate((residual.values, residual_pred_seq))
        seasonal_pred_seq = list(seasonal[len(seasonal) - period:]) * (round((pred_period) / period) + 1) 
        seasonal_pred_seq = np.array(seasonal_pred_seq[0:pred_period])

        pred_period = (datetime.datetime.strptime(end, '%Y-%m-%d %H:%M:%S') - datetime.datetime.strptime(start, '%Y-%m-%d %H:%M:%S')).days + 1
        return trend_pred_seq[len(trend_pred_seq)-pred_period:] + \
               residual_pred_seq[len(residual_pred_seq)-pred_period:] + \
               seasonal_pred_seq[len(seasonal_pred_seq)- pred_period:]

    else:
        trend_pred_seq = np.array(trend_arima_fit.fittedvalues)
        residual_pred_seq = np.array(residual_arima_fit.fittedvalues)
        seasonal_pred_seq = np.array(seasonal)
        return trend_pred_seq + residual_pred_seq + seasonal_pred_seq


def GARCH_model(resid, args):
    """
    resid: stationary residual time_series after ARIMA. 
    args: arguments parsed before.
    name: the name of time_series_diff.
    return best parameters for GARCH model.
    """
    best_criteria = np.inf 
    best_model_order = (0, 0)
    best_model_fit = None
    for p in range(args.max_p):
        for q in range(args.max_q):
            try:
                model = arch_model(resid,
                                   mean = "Constant",
                                   p = p, 
                                   q = q, 
                                   vol = 'GARCH',
                                   rescale=False)
                model_fit = model.fit(disp = "off",
                                      update_freq = 0,
                                      tol = 1e-8,
                                      show_warning = False)

                current_criteria = model_fit.bic
                
                if current_criteria <= best_criteria:
                    best_criteria, best_model_order, best_model_fit = np.round(current_criteria, 0), (p, q), model_fit
            except:
                pass

    return best_model_fit, best_model_order


def ARIMA_model(time_series_diff, args):
    """
    time_series_diff: stationary time_series after diff. 
    args: arguments parsed before.
    name: the name of time_series_diff.
    return fitted ARIMA model, parameters for ARIMA model.
    """
    # find the optimal order of ARIMA model.
    evaluate = sm.tsa.arma_order_select_ic(time_series_diff,
                                           max_ar = args.max_ar,
                                           max_ma = args.max_ma)
    min_order = evaluate["bic_min_order"] # get the parameter for ARIMA model.

    # initial the success_flag to false
    success_flag = False
    while not success_flag:
        # construct the ARIMA model.
        model = ARIMA(time_series_diff, order=(min_order[0], 0, min_order[1])) # d is the order of diff, which we have done that perviously.
        # keep finding initial parameters until convergence.
        try:
            model_fit = model.fit(disp = False, 
                                  start_params = np.random.rand(min_order[0] + min_order[1] + 1),
                                  trend = "c", # Some posts' experimentation suggests that ARIMA models may be less likely to converge with the trend term disabled, especially when using more than zero MA terms.
                                  transparams = True,
                                  solver = "lbfgs", # we turn to use this one, which gives the best RMSE & executation time.
                                  tol = 1e-8, # The convergence tolerance. Default is 1e-08.
                                  )
            success_flag = True
        except:
            pass

    return model_fit, min_order


def mix_model(time_series_diff, args):
    """
    time_series_diff: stationary time_series after diff. 
    args: arguments parsed before.
    name: the name of time_series_diff.
    return fitted ARIMA model, parameters for ARIMA model, fitted GARCH model and parameters for GARCH model.
    """
    # get arima model
    arima_model_fit, arima_order = ARIMA_model(time_series_diff, args)
    # get garch model
    garch_model_fit, garch_order = GARCH_model(arima_model_fit.resid, args)

    return arima_model_fit, arima_order, garch_model_fit, garch_order


def decompose(time_series, season_period):
    """
    times_seris: time_series, pd.Dataframe.
    season_period: period of seasonality, float.
    return the decomposition of the time_series including trend, seasonal, residual.
    """    
    # find the filt for seasonal_decompose
    if season_period % 2 == 0:  # split weights at ends
        filt = np.array([.5] + [1] * (season_period - 1) + [.5]) / season_period
    else:
        filt = np.repeat(1. / season_period, season_period)

    decomposition = seasonal_decompose(time_series, 
                                       model = 'additive', # additive model is the default choice. 
                                                           # We tried "multiplicative" but it is totally meaningless.
                                       two_sided = True,
                                       filt = filt,
                                       extrapolate_trend = 'freq',
                                       period = season_period) 
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid

    return trend, seasonal, residual


def data_loader(ticker, month, end):
    """
    ticker: tuple, containing info of ticker and data sources.
    month: the range of data.
    end: the end date of the range.
    return the dataframes according to ticker from corresponding sources.
    """
    # get the start date and end date
    if end > 0:
        start_date = datetime.date.today() + relativedelta(months = -month)
        end_date = datetime.date.today()
    else:    
        start_date = datetime.date.today() + relativedelta(months = -month, days = end - 1)
        end_date = datetime.date.today() + relativedelta(days = end)
    # fetching data frames
    try:
        close = yf.download(ticker, start=start_date, end=end_date, adjusted=True)["Adj Close"]
        
    except:
        logger.error("The data can not be obtained from yahoo, plz try other ticker")
        exit(0)

    if close.index.tolist()[-1] != end_date:
        close = close.append(pd.Series({pd.Timestamp(str(end_date)):close.iloc[-1]}))
    # print(close.index.tolist()[-1] == end_date)
    
    close = close.resample('D').bfill().iloc[0:-1] # fullfill the time series.
     # data cleaning
    if np.sum(close.isnull()) > 0:
        logger.debug("The time series contain missing values & we use interpolation to resolve this issue")
        close = close.interpolate(method='polynomial', order=2, limit_direction='forward', axis=0)
    # Then, if there is still some missing values, we simply drop this value.abs
    close = close.dropna()
    
    return close


class Args:
    def __init__(self, start, days, ticker, month, period, testday, max_ar, max_ma, max_p, max_q):
        self.start = start # the date that prediction begins at
        self.days = days # the predict days
        self.ticker = ticker # ticker of stock
        self.month = month # historical data (mth)
        self.period = period # seasonal period (mth)
        self.testday = testday # test set days
        self.max_ar = max_ar # max 1st arima order
        self.max_ma = max_ma # max 2nd arima order
        self.max_p = max_p # max 1st garch order
        self.max_q = max_q # max 2nd garch order
        self.figdir = "./figure/"
        self.info = {'start':start, 'days':days, 'ticker':ticker, 'month':month, 'period':period, 
                        'testday':testday, 'max_ar':max_ar, 'max_ma':max_ma, 'max_p':max_p, 'max_q':max_q}

    def check_valid(self):
        # check if data range is legal.
        if  self.month <= 0:
            logger.warning("The data range is illegal. Turn to use default 3")
            self.month = 3

        # check if period is legal.
        if self.period < 1:
            logger.warning("Seasonal period is illegal. Turn to use default 3.")
            self.period = 3

        # check the days
        if self.days <= 0:
            logger.warning("The days for prediction is smaller 0. Turn to default 30.")
            self.days = 30

        return



def ARIMA_GARCH_prediction(ticker, duration):
    '''
    input: sticker (e.g. '^STI'), duration (days, e.g. 180)
    output: accuracy, accumulated return
    '''
    # parse arguments
    args = Args(start=0, # prediction start from today (0)
                days=duration, # predicted period length (days)
                ticker=ticker, # stock ticker
                month=10*12, # historical data (mths)
                period=12, # seasonal period length
                testday=30, # test data size (days)
                max_ar=2, 
                max_ma=3, 
                max_p=3, 
                max_q=3)
    args.check_valid()
    
    # set the level of logger
    # logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.INFO)

    logger.info("--------TEST enviroment start---------")
    logger.debug("--------DEBUG enviroment start---------")
    
    # show the hyperparameters
    logger.info("---------hyperparameter setting---------")
    logger.info(args.info)

    # set the random seed
    np.random.seed(0)

    # data fetching
    logger.info("-------------Data fetching-------------")
    close = data_loader(args.ticker, args.month, args.start) # get dataframes from "yahoo" finance.
    
    # data analyzing.
    logger.info("ADF test for {} - Close: {:.3f}".format(args.ticker, ADF(close.tolist())[1]))

    # data splitting
    logger.info("-------------Data splitting------------")
    train = close[0:(len(close) - args.testday)]
    test = close[(len(close) - args.testday):]

    # time serise decomposition
    logger.info("-------------decomposition-------------")
    trend, seasonal, residual = decompose(train, args.period)

    # ARIMA model
    logger.info("-------ARIMA GARCH construction--------")
    t = time.time()
    trend_arima_fit, trend_arima_order, trend_garch_fit, trend_garch_order = mix_model(trend, args)
    logger.info("Trend ARIMA parameters: " + str(tuple([trend_arima_order[0],
                                                        0,
                                                        trend_arima_order[1]])))
    logger.info("Trend GARCH parameters: " + str(trend_garch_order))
    residual_arima_fit, residual_arima_order, residual_garch_fit, residual_garch_order = mix_model(residual, args)
    logger.info("Residual ARIMA parameters: " + str(tuple([residual_arima_order[0],
                                                          0,
                                                          residual_arima_order[1]])))
    logger.info("Residual GARCH parameters: " + str(residual_garch_order))
    logger.info("Time cost of model construction: {:.3f}s".format(time.time() - t))

    # loss calculation
    logger.info("-----------Loss/Error calculation------------")
    fit_seq = model_predict(trend_arima_fit, residual_arima_fit,
                            trend_garch_order, residual_garch_order,
                            trend, residual, seasonal, 
                            False, "", "", args.period)
    fit_seq = pd.Series(fit_seq, index=train.index)

    # calculate training loss ()
    training_loss = np.sqrt(mean_squared_error(np.array(train), np.array(fit_seq)))
    logger.info("Training loss/error (RMSE): {:.3f}".format(training_loss))

    train_accuracy = 1 - np.mean(np.abs(np.array(train) - np.array(fit_seq)) / np.array(train))
    logger.info("Training accuracy: {:.3f}%".format(train_accuracy * 100))

    ## Evaluation on test set
    logger.info("--------------evaluation---------------")
    t = time.time()
    pred_seq = model_predict(trend_arima_fit, residual_arima_fit,
                            trend_garch_order, residual_garch_order,
                            trend, residual, seasonal, 
                            True, str(test.index.tolist()[0]), str(test.index.tolist()[-1]), args.period)
    pred_seq = pd.Series(pred_seq, index=test.index)
    # calculate testing loss
    testing_loss = np.sqrt(mean_squared_error(np.array(test), np.array(pred_seq)))
    logger.info("Testing loss/error (RMSE): {:.3f}".format(testing_loss))

    accuracy = 1 - np.mean(np.abs(np.array(test) - np.array(pred_seq)) / np.array(test))
    logger.info("Testing accuracy: {:.3f}%".format(accuracy * 100))
    logger.info("Time cost of evaluation: {:.3f}s".format(time.time()-t))

    # prediction
    logger.info("--------------prediction---------------")
    t = time.time()
    start_date = datetime.date.today() + relativedelta(days = args.start)
    end_date = datetime.date.today() + relativedelta(days = args.days + args.start - 1)
    
    prediction = model_predict(trend_arima_fit, residual_arima_fit,
                               trend_garch_order, residual_garch_order,
                               trend, residual, seasonal, 
                               True, str(start_date) + " 00:00:00", str(end_date) + " 00:00:00", args.period)
    prediction = pd.Series(prediction, index=pd.date_range(start_date, end_date))

    accumulate_ret = (prediction[-1] - close[-1]) / close[-1]
    logger.info("Accumulated return ratio: {:.3f}%".format(accumulate_ret * 100))
    logger.info("Time cost of prediction: {:.3f}s".format(time.time()-t))

    # plot
    logger.info("--------------plotting---------------")
    # plot train and fitted values in one graph.
    plot_pred_vs_truth(fit_seq, train, pred_seq, test, prediction, ticker)
    sharpeRate = sharpe_ratio(np.diff(np.concatenate((close, prediction))))

    logger.info("--------------Process ends-------------")

    return ticker, accuracy, accumulate_ret, sharpeRate

if __name__ == "__main__":
    t = time.time()

    ## test demo
    ticker, accuracy, accumulate_ret, _ = ARIMA_GARCH_prediction('^STI', 6)
    
    print("Time cost in total: {:.3f}s".format(time.time() - t))