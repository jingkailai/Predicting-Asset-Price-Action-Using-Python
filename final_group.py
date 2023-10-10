import yfinance as yf
import numpy as np

ASSETS = ['A17U.SI', 'C31.SI', 'C38U.SI', 'C09.SI', 'C52.SI', 'D01.SI', 
        'D05.SI', 'BUOU.SI', 'G13.SI', 'H78.SI', 'J36.SI', 'BN4.SI', 
        'AJBU.SI', 'N2IU.SI', 'ME8U.SI', 'M44U.SI', 'O39.SI', 'S58.SI', 
        'U96.SI', 'S68.SI', 'C6L.SI', 'Z74.SI', 'S63.SI', 'Y92.SI', 
        'U11.SI', 'U14.SI', 'V03.SI', 'F34.SI', 'BS6.SI']


def stock_filter(start_date, end_date, User_Risk):

    prices_df = yf.download(ASSETS, start=start_date, end=end_date, adjusted=True)

    #if calculating PCT_Change: pct_change,returns = prices_df['Adj Close'].pct_change().dropna()
    #if calculating Annual Returns: annual_returns = prices_df['Adj Close'].resample('Y').ffill().pct_change().dropna()

    #Calculating SD
    df = prices_df['Adj Close'].resample('Y').ffill().std().dropna()

    #Converting from pd.Series to DataFrame with default 0 index as "SD"
    df = df.to_frame("SD")

    #Check the datatype
    #print(type(df))

    newdf = df.reset_index().rename(columns={'index': 'Stock Ticker'})

    thrd = np.max(newdf['SD']/5)
    newdf['Risk'] = ''

    newdf.loc[newdf['SD'] < thrd, 'Risk'] = 'very low'
    newdf.loc[(newdf['SD'] >= thrd) & (newdf['SD'] < 2 * thrd), 'Risk'] = 'low'
    newdf.loc[(newdf['SD'] >= 2 * thrd) & (newdf['SD'] < 3 * thrd), 'Risk'] = 'medium'
    newdf.loc[(newdf['SD'] >= 3 * thrd) & (newdf['SD'] < 4 * thrd), 'Risk'] = 'high'
    newdf.loc[newdf['SD'] >= 4 * thrd, 'Risk'] = 'very high'

    #storing different stocks into different variables, based on their SD
    verylownewdf = newdf[newdf.Risk=='very low']
    lownewdf = newdf[newdf.Risk=='low']
    mediumnewdf = newdf[newdf.Risk=='medium']
    highnewdf = newdf[newdf.Risk=='high']
    veryhighnewdf = newdf[newdf.Risk=='very high']
    
    # User_Risk = str(input("What is your risk tolerance level when it comes to investing in stocks:\n very low\n low\n medium\n high\n very high "))    

    if User_Risk == "very low":
        res = verylownewdf
        print("These are the recommended stocks with very low risk:\n {}".format(res))

    elif User_Risk == "low":
        res = lownewdf
        print("These are the recommended stocks with low risk:\n {}".format(res))

    elif User_Risk == 'medium':
        res = mediumnewdf
        print("These are the recommended stocks with medium risk:\n {}".format(res))
        
    elif User_Risk == 'high':
        res = highnewdf
        print("These are the recommended stocks with high risk:\n {}".format(res))

    elif User_Risk == 'very high':
        res = veryhighnewdf
        print("These are the recommended stocks with very high risk:\n {}".format(res))

    else:
        raise ValueError("Input risk level is unvalid!")

    return res


if __name__ == '__main__':
    stock_filter("2016-08-20", "2021-08-20", "medium")