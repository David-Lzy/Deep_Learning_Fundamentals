import os
import sys
import getpass
from binance.spot import Spot

loc_list = os.path.abspath(__file__).split(os.sep)
HOME_LOC = os.path.join(os.sep, *loc_list[:-3])
sys.path.append(HOME_LOC)
os.chdir(HOME_LOC)
print(HOME_LOC)

from CODE.Utils.encrypt import *
from CODE.Utils.encrypt import Encrypted_API_key
from CODE.Utils.encrypt import Encrypted_API_secret
from CODE.Utils.utils import *
from CODE.Utils.indicators import *
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()


def Normalize(df):
    # 对于价格信息进行特殊的归一化处理
    for price_col in ["Open", "High", "Low", "Close"]:
        df[f"{price_col}_normalized"] = df[price_col].pct_change() * 100  # 后一天除以前一天再减去1

    # 移除原始的价格列
    df.drop(
        ["Open", "High", "Low", "Close", "Open Time", "Close Time"],
        axis=1,
        inplace=True,
    )

    # 处理NaN值
    df.fillna(method="ffill", inplace=True)  # 前向填充
    df.dropna(inplace=True)  # 删除仍然包含NaN的行

    # 对其它特征进行标准化处理
    scaler = StandardScaler()
    for col in df.columns:
        if "_normalized" not in col:  # 只标准化非归一化的列
            df[col] = scaler.fit_transform(df[[col]])

    return df


def Augmention(df):
    period = 10
    df["SMA_10"] = calculate_sma(df["Close"], period)
    df["EMA_10"] = calculate_ema(df["Close"], period)
    df["RSI_14"] = calculate_rsi(df, period)
    calculate_bollinger_bands(df, n=period, num_std=2)
    calculate_ATR(df)
    calculate_OBV(df)
    df = Normalize(df)
    return df


if __name__ == "__main__":
    df = pd.read_csv(
        "/Project/David_Li/Works/Courses/4339_COMP_SCI_7318_Deep_Learning_Fundamentals/Assignment3/DATA/RAW/1d/1514758223872.csv"
    )
    print(Augmention(df))
