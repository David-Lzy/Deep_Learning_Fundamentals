import numpy as np

# 使用“收盘价”（Close）来计算10天周期的简单移动平均（SMA_10）和指数移动平均（EMA_10）的原因在于以下几点：

# 市场共识：收盘价通常被认为是交易日中最重要的价格，因为它代表了市场对该资产的最终共识。这种价格通常被视为交易日的“结算价”。

# 稳定性：与日内的波动性价格（如最高价、最低价或开盘价）相比，收盘价通常更稳定，能更准确地反映市场的总体趋势。

# 广泛应用：在技术分析中，收盘价是计算各种指标（如移动平均、MACD、RSI等）的常用价格。因此，使用收盘价可以使您的分析与市场上的其他分析方法保持一致。

# 信息整合：收盘价包含了该交易时段内所有交易活动的信息，它是在市场闭市前的竞价过程中形成的，因此反映了市场参与者的综合观点。

# 当然，技术分析是灵活的，根据特定的分析需求和策略，也可以选择使用其他类型的价格（如开盘价、最高价、最低价）来计算这些指标。但在大多数情况下，收盘价是最常用的选择。


# Function to calculate Simple Moving Average (SMA)
def calculate_sma(data, period):
    return data.rolling(window=period).mean()


# Function to calculate Exponential Moving Average (EMA)
def calculate_ema(data, period):
    return data.ewm(span=period, adjust=False).mean()


# RSI是衡量股票或资产价格动量的一个指标，其数值范围在0到100之间。一般来说，RSI值超过70可能表明资产被超买，而RSI值低于30可能表明资产被超卖。

# 在这个例子中，我使用了14天的时间窗口来计算RSI，这是一个常用的设置。RSI的计算基于资产收盘价的变化，我们首先计算连续两天之间的价格变化，然后区分出上涨和下跌的情况来分别计算平均增益和平均损失。接着，利用这些增益和损失来计算相对强度（RS），最后将RS值转换为RSI。

# 在生成的数据中，你会看到前几行的RSI值是NaN，这是因为在计算期初，没有足够的数据来生成14天的移动平均。随着数据的累积，RSI值会开始显示。


def calculate_rsi(data, period=14):
    delta = data[f"Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# Bollinger Bands（布林带）是一种在价格图表上使用的技术分析工具，它由三条线组成：

# 上带（Upper Band）：计算为移动平均线加上标准差的两倍。
# 下带（Lower Band）：计算为移动平均线减去标准差的两倍。
# 中带（SMA）：通常是20期的简单移动平均线。
# 布林带是衡量价格波动性的一种工具。上下带的宽度会随着价格波动的增加而增宽，反之亦然。通常，价格触及上带可能表示市场过热，价格触及下带可能表示市场过冷。但请注意，布林带不是买卖信号的直接来源，而是辅助分析价格走势的工具。

# 根据计算结果，前几行的数据因为无法计算20期的移动平均和标准差，所以显示为NaN（Not a Number）。随着数据集中的数据量增加，这些值将被填充


# 计算Bollinger Bands
def calculate_bollinger_bands(data, n=20, num_std=2):
    """
    Calculate Bollinger Bands for a given data.

    :param data: Pandas DataFrame with the financial data (Close prices).
    :param n: The number of periods for the moving average.
    :param num_std: The number of standard deviations for the upper and lower bands.
    :return: A DataFrame with Bollinger Bands (upper band, lower band, and moving average).
    """
    data[f"SMA_{n}"] = data[f"Close"].rolling(window=n).mean()
    data[f"STD_{n}"] = data[f"Close"].rolling(window=n).std()
    data[f"Upper_Band_{n}"] = data[f"SMA_{n}"] + (data[f"STD_{n}"] * num_std)
    data[f"Lower_Band_{n}"] = data[f"SMA_{n}"] - (data[f"STD_{n}"] * num_std)
    return data[[f"Upper_Band_{n}", f"Lower_Band_{n}"]]


# MACD（Moving Average Convergence Divergence，移动平均收敛散度）。MACD是由Gerald Appel开发的，用于揭示资产价格走势的强度、方向、动量以及趋势的持续时间。它由三个部分组成：

# MACD Line（MACD线）：通常是12期指数移动平均（EMA）减去26期EMA。
# Signal Line（信号线）：通常是MACD线的9期EMA。
# Histogram（直方图）：MACD线与信号线之间的差异。
# MACD的核心思想是当短期EMA超过长期EMA时，可能是买入信号；反之，则可能是卖出信号。直方图的高度（正或负）表示MACD线与信号线之间的差距，是买卖动量的指标。


def calculate_MACD_Line(
    df,
):
    # 短期EMA
    df[f"EMA_12"] = df[f"Close"].ewm(span=12, adjust=False).mean()
    # 长期EMA
    df[f"EMA_26"] = df[f"Close"].ewm(span=26, adjust=False).mean()
    # MACD Line
    df[f"MACD_Line"] = df[f"EMA_12"] - df[f"EMA_26"]
    # Signal Line
    df[f"Signal_Line"] = df[f"MACD_Line"].ewm(span=9, adjust=False).mean()
    # Histogram
    df[f"Histogram"] = df[f"MACD_Line"] - df[f"Signal_Line"]
    return df[[f"MACD_Line", "Signal_Line", "Histogram"]]


# 布林带（Bollinger Bands）。布林带是由John Bollinger在20世纪80年代开发的，用于衡量价格的高低和波动性。布林带包括三条线：

# 中间线：通常是20期的简单移动平均（SMA）。
# 上带：中间线上方的两个标准差。
# 下带：中间线下方的两个标准差。
# 当价格接近上带时，可能表明市场过热；当价格接近下带时，可能表明市场过冷。价格经常在两带之间波动，这些带也可以根据市场波动性而变宽或变窄。


def calculate_Bollinger_Bands(df, window=20, num_std=2):
    # 计算布林带
    # 中间线（20期SMA）
    df[f"SMA_{window}"] = df[f"Close"].rolling(window=window).mean()
    # 计算标准差
    df[f"STD_{window}"] = df[f"Close"].rolling(window=window).std()
    # 上带
    df[f"Upper_Band"] = df[f"SMA_{window}"] + (df[f"STD_{window}"] * 2)
    # 下带
    df[f"Lower_Band"] = df[f"SMA_{window}"] - (df[f"STD_{window}"] * 2)

    # 显示结果
    return df[[f"Open Time", "Close", "SMA_{window}", "Upper_Band", "Lower_Band"]]


# ATR（Average True Range，平均真实范围）。ATR是由Welles Wilder开发的，用于衡量市场波动性。它不是方向性指标，但可以帮助衡量价格波动的程度。ATR的计算涉及以下步骤：

# 计算每个周期的真实范围（True Range, TR）。TR是以下三个数值中的最大者：
# 当前周期的最高价与最低价之差。
# 当前周期最高价与前一周期收盘价之差的绝对值。
# 当前周期最低价与前一周期收盘价之差的绝对值。
# 通过取特定周期（通常为14天）内TR的平均值来计算ATR。


def calculate_ATR(df, windows=14):
    # 真实范围（True Range, TR）
    df[f"High-Low"] = df[f"High"] - df[f"Low"]
    df[f"High-Prev_Close"] = np.abs(df[f"High"] - df[f"Close"].shift(1))
    df[f"Low-Prev_Close"] = np.abs(df[f"Low"] - df[f"Close"].shift(1))
    df[f"TR"] = df[[f"High-Low", "High-Prev_Close", "Low-Prev_Close"]].max(axis=1)

    # 计算14天平均真实范围（ATR）
    df[f"ATR_14"] = df[f"TR"].rolling(window=windows).mean()

    # 显示结果
    return df[[f"Open Time", "Close", "High", "Low", "TR", "ATR_14"]]


def calculate_OBV(df):
    # 计算OBV
    df[f"OBV"] = np.where(df[f"Close"] > df[f"Close"].shift(1), df[f"Volume"], 0)
    df[f"OBV"] = df[f"OBV"] + df[f"OBV"].shift(1)
    df[f"OBV"] = df[f"OBV"].fillna(0)

    # 显示结果
    return df[[f"Open Time", "Close", "Volume", "OBV"]]
