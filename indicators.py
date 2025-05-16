import ta
def add_technical_indicators(df):
    df = df.copy()
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close']).rsi()
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['SMA'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
    df['EMA'] = ta.trend.EMAIndicator(close=df['Close'], window=20).ema_indicator()
    bb = ta.volatility.BollingerBands(close=df['Close'])
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df.dropna(inplace=True)
    return df
