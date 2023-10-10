# lstm_model

from empyrical.stats import sharpe_ratio


def lstm_model(name, start_time, end_time, duration):
    import yfinance as yf
    import numpy as np

    df = yf.download(name, start=start_time, end=end_time)
    data = df.values[:, 4:5]
    di = 1
    do = 1
    s = int(duration / 3 * 2)

    # 数据z-score标准化
    vmean = df.iloc[:, 4:5].apply(lambda x: np.mean(x))
    vstd = df.iloc[:, 4:5].apply(lambda x: np.std(x))
    df_st = df.iloc[:, 4:5].apply(lambda x: (x - np.mean(x)) / np.std(x)).values
    print(df_st.shape)

    # 得到标准化的训练集和测试集
    x_train = np.zeros((df_st.shape[0] - s - duration, s, di))
    y_train = np.zeros((df_st.shape[0] - s - duration, do), )
    x_test = np.zeros((duration, s, di))
    y_test = np.zeros((duration, do), )

    for i in range(s + duration, df_st.shape[0] - duration):
        y_train[i - s] = df_st[i]
        x_train[i - s] = df_st[(i - duration - s):(i - duration)]
    for i in range(df_st.shape[0] - duration, df_st.shape[0]):
        y_test[i - df_st.shape[0] + duration] = df_st[i]
        x_test[i - df_st.shape[0] + duration] = df_st[(i - duration - s):(i - duration)]

    # 基于训练集对模型进行训练
    from keras.layers import LSTM, Dense
    from keras.models import Sequential
    model = Sequential()
    model.add(LSTM(64, input_shape=(s, di), activation='relu', recurrent_dropout=0.01))
    model.add(Dense(do, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    history = model.fit(x_train, y_train, epochs=20, batch_size=10, validation_split=0)  # 测试

    # 预测实现
    preddf = model.predict(x_test) * vstd.values + vmean.values
    accuracy = 100 * (1 - np.sum(np.sum(np.abs(preddf - df.iloc[-duration:df_st.shape[0], 4:5]).values) / np.sum(
        df.iloc[-duration:df_st.shape[0], 4:5].values)))

    # 预测未来
    x_pre = np.zeros((1, s, di))
    ss = df_st.shape[0]

    for i in range(ss, ss + duration):
        x_pre[0] = df_st[(i - duration - s):(i - duration)]
        y_pre = model.predict(x_pre)
        df_st = np.concatenate((df_st, y_pre))

    df_pre = df_st[ss:ss + duration] * vstd.values + vmean.values

    df_predit = np.concatenate((data, df_pre))

    return_rate = (df_predit[-1] - df_predit[-(duration+1)])/df_predit[-(duration+1)]

    sharpeRate = sharpe_ratio(np.diff(df_predit))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 7))
    plt.plot(range(200), df_predit[-(200+duration+1):-(duration+1)], c='black')
    plt.plot(range(199, 200 + duration + 1), df_predit[-(duration + 2):df_predit.shape[0]], '--', c='red')
    plt.ylabel("$" + df.columns[4] + "$")
    plt.title("Truth v.s. Predict of {}".format(name))
    plt.savefig("./figure/lstm_pred_{}.png".format(name))

    return name, accuracy, return_rate, sharpeRate


if __name__ == '__main__':
    name = 'AAPL'
    start_time = "2011-08-31"
    end_time = "2021-08-31"
    duration = 30
    name, accuracy, return_rate, _ = lstm_model(name, start_time, end_time, duration)
    print("evaluation on test data: accuracy = %0.2f%% \n" % accuracy)
    print("predict the return rate: return_rate = %0.2f%% \n" % return_rate)