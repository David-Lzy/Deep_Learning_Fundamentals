import os
import sys
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


year = [2018, 2019, 2020, 2021, 2022]
month = [i for i in range(1, 13)]
time_scales = ["1m", "5m", "1h", "6h", "1d", "1w"]

client = Spot()
symbol = "ETHUSDT"
limit = 60
length = 60
X_density = 60

for y in year:
    for m in month:
        random_end_timestamps = generate_random_timestamps(
            (y, m), (y, m + 1), X_density
        )

        for i_end_timestamp in random_end_timestamps:
            data = get_klines_data(client, i_end_timestamp, time_scales, length=length)
            save_klines_to_csv(
                data,
                time_scales,
                os.path.join(HOME_LOC, "DATA", "RAW"),
                i_end_timestamp,
            )
