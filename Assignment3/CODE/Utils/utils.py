import os
import sys
import re
import datetime
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

loc_list = os.path.abspath(__file__).split(os.sep)
HOME_LOC = os.path.join(os.sep, *loc_list[:-3])
sys.path.append(HOME_LOC)
os.chdir(HOME_LOC)

from CODE.Utils.normalize import *


def get_timestamp(Y=2023, M=1, D=1, H=0, m=0, s=0):
    """Generate a timestamp for a given date and time."""
    if M > 12:
        M = M % 12
        Y = Y + 1
    date = datetime.datetime(Y, M, D, H, m, s)
    return int(date.timestamp() * 1000)


def get_timestamp_now():
    """Generate a timestamp for the current date and time."""
    return int(datetime.datetime.now().timestamp() * 1000)


#### 获得不同尺度K线 ######


def get_time_delta(scale, length):
    """
    Calculate the time difference based on a time scale string (e.g., "20m", "3h").

    :param scale: String representing the time scale (e.g., "1m", "1h", "1d", "1w", "1M").
    :return: Time difference in milliseconds.
    """
    match = re.match(r"(\d+)([mhdwM])", scale)
    if not match:
        raise ValueError("Invalid time scale format: {}".format(scale))

    quantity, unit = int(match.group(1)), match.group(2)

    if unit == "m":
        return length * quantity * 60 * 1000  # Minutes to milliseconds
    elif unit == "h":
        return length * quantity * 60 * 60 * 1000  # Hours to milliseconds
    elif unit == "d":
        return length * quantity * 24 * 60 * 60 * 1000  # Days to milliseconds
    elif unit == "w":
        return length * quantity * 7 * 24 * 60 * 60 * 1000  # Weeks to milliseconds
    elif unit == "M":
        return (
            length * quantity * 30 * 24 * 60 * 60 * 1000
        )  # Months to milliseconds (assuming 30 days per month)
    else:
        raise ValueError("Unsupported time scale unit: {}".format(unit))


def get_klines_data(client, end_timestamp, time_scales, symbol="ETHUSDT", length=1000):
    data = []
    for scale in time_scales:
        # 计算开始时间
        start_time = end_timestamp - get_time_delta(scale, length)
        # 获取K线数据
        klines = client.klines(
            symbol, scale, startTime=start_time, endTime=end_timestamp, limit=length
        )
        data.append(klines)
    return data


def extract_features(depth_data):
    bids = np.array(depth_data["bids"], dtype=float)
    asks = np.array(depth_data["asks"], dtype=float)

    # 1. 订单簿不平衡度
    order_book_imbalance = np.sum(bids[:, 1]) - np.sum(asks[:, 1])

    # 2. 累积订单量
    cumulative_bid_volume = np.sum(bids[:, 1])
    cumulative_ask_volume = np.sum(asks[:, 1])

    # 3. 价格-数量斜率
    bid_slope = np.polyfit(bids[:, 0], bids[:, 1], 1)[0]
    ask_slope = np.polyfit(asks[:, 0], asks[:, 1], 1)[0]

    # 4. 流动性度量
    liquidity_measure = (cumulative_bid_volume + cumulative_ask_volume) / (
        np.max(bids[:, 0]) - np.min(asks[:, 0])
    )

    return (
        order_book_imbalance,
        cumulative_bid_volume,
        cumulative_ask_volume,
        bid_slope,
        ask_slope,
        liquidity_measure,
    )


# 均匀随机获得不同尺度的时间戳


def generate_random_timestamps(start_time, end_time, num_per_time_scale):
    random_timestamps = []
    start_timestamp = get_timestamp(*start_time)
    end_timestamp = get_timestamp(*end_time)
    for _ in range(num_per_time_scale):
        timestamp = random.randint(start_timestamp, end_timestamp)
        random_timestamps.append(timestamp)
    random_timestamps.sort()
    return random_timestamps


def save_klines_to_csv(data, time_scales, file_path, timestamp):
    """
    Save the kline data for different time scales into separate CSV files.

    :param data: List of kline data for different time scales.
    :param time_scales: List of time scales corresponding to the data.
    :param file_path: Base file path for the CSV files to be saved.
    """
    # Column names for the kline data, excluding "Ignore"
    kline_columns = [
        "Open Time",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Close Time",
        "Quote Asset Volume",
        "Number of Trades",
        "Taker Buy Base Asset Volume",
        "Taker Buy Quote Asset Volume",
        "_",
    ]
    dfs = []
    # Create and save a DataFrame for each time scale
    for i, scale in enumerate(time_scales):
        df = pd.DataFrame(data[i], columns=kline_columns)
        df = df.drop(columns=["_"])
        dfs.append(df)
        scale_file_path = os.path.join(file_path, scale, f"{timestamp}.csv")
        df.to_csv(scale_file_path, index=False)
        print(f"Saved: {scale_file_path}")
    return dfs


def process_csv(file_path, x_length=58, y_length=2, augmention=True):
    """
    Process a single CSV file to extract X and Y data.

    :param file_path: Path to the CSV file.
    :param x_length: Number of records to be used for X.
    :param y_length: Number of records to be used for Y.
    :return: X and Y data DataFrames.
    """
    df = pd.read_csv(file_path)
    Y = df["Close"].iloc[-(y_length + 1) :].pct_change().dropna()
    if not df.shape[0] == x_length+y_length:
        raise ValueError
    if augmention:
        df = Augmention(df)
    X = df.iloc[:x_length]
    return X, Y


def split_data(file_paths, train_ratio=0.85, x_length=58, y_length=2, augmention=True):
    """
    Split the files into training and testing sets and extract X and Y data.

    :param file_paths: List of paths to CSV files.
    :param train_ratio: Ratio of files to be used for training.
    :param x_length: Number of records from each file to be used for X.
    :param y_length: Number of records from each file to be used for Y.
    :param augmention: Whether to apply data augmentation.
    :return: Training and testing X and Y data.
    """
    split_index = int(len(file_paths) * train_ratio)

    train_X_list, train_Y_list = [], []
    test_X_list, test_Y_list = [], []

    # 使用tqdm显示训练集的处理进度
    for file_path in tqdm(file_paths[:split_index], desc="Processing Training Files"):
        try:
            X, Y = process_csv(file_path, x_length, y_length, augmention)
            train_X_list.append(X)
            train_Y_list.append(Y)
        except ValueError:
            continue

    # 使用tqdm显示测试集的处理进度
    for file_path in tqdm(file_paths[split_index:], desc="Processing Testing Files"):
        try:
            X, Y = process_csv(file_path, x_length, y_length)
            test_X_list.append(X)
            test_Y_list.append(Y)
        except ValueError:
            continue
    train_X = np.stack(train_X_list)
    train_Y = np.stack(train_Y_list)
    test_X = np.stack(test_X_list)
    test_Y = np.stack(test_Y_list)

    return train_X, train_Y, test_X, test_Y


def load_and_sort_files(directory):
    """
    Load all CSV files in the given directory, and sort them based on the timestamp in the file name.

    :param directory: Path to the directory containing the CSV files.
    :return: Sorted list of file paths.
    """
    # 获取目录中所有CSV文件的路径
    file_paths = [
        os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".csv")
    ]

    # 根据文件名中的时间戳对文件进行排序
    file_paths.sort(key=lambda x: int(x.split("/")[-1].split(".")[0]))

    return file_paths
