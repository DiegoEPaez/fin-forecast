import logging
import math
from datetime import timedelta, datetime

# module-specific logger
logger = logging.getLogger(__name__)

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import pandas_ta as ta
from pandas.tseries.offsets import MonthEnd, QuarterEnd
import numpy as np

from data.banxico_data import dwld_bmx
from data.inegi_data import dwld_inegi
from data.multpl_data import dwld_multpl
from series_data import *

# INEGI: https://www.inegi.org.mx/servicios/api_biinegi.html
# BANXICO: https://www.banxico.org.mx/SieAPIRest/service/v1/doc/catalogoSeries

na_values = [".", "N/E", "null", "None", "#N/A N/A"]
start_dt = datetime(1970, 1, 1)
end_dt = datetime.now() - timedelta(1)


def get_series(info_dict):
    # Should store data somewhere to avoid reconsulting
    dfs = []
    for series, values in info_dict.items():
        if values["source"] == "YAHOO":
            df = get_data_yahoo(
                series,
                start_dt,
                end_dt,
                values["yahoo_ticker"],
                values.get("col", "Adj Close"),
            )
        elif values["source"] == "FRED":
            df = get_data_fred(series, start_dt, end_dt, values["freq_adj"])
        elif values["source"] == "BANXICO":
            df = get_data_banxico(
                series, start_dt, end_dt, values["bmx_serie"], values["freq_adj"]
            )
        elif values["source"] == "INEGI":
            df = get_data_inegi(series, values["series"])
        elif values["source"] == "MULTPL":
            df = get_data_multpl(values["name"])
        else:
            continue

        df = df.ffill()

        dfs.append(df)

    return dfs


def join_series(dfs):
    join = dfs[0]

    i = 0
    for df in dfs:
        if i == 0:
            i += 1
            continue

        join = join.merge(df, left_index=True, right_index=True, how="outer")

        i += 1

    return join


def tech_oscillators(df, symbol, symbol_low, symbol_high):
    df = df.rename(columns={symbol: "close", symbol_low: "low", symbol_high: "high"})

    # Relative Strength Index
    df["ta_rsi"] = df.ta.rsi(14)

    # Stochastic Oscillator
    # df['ta_stochk'] = df.ta.stoch(14, 3, 3).iloc[:, 0]  # Stochastic %K
    # df['ta_stochd'] = df.ta.stoch(14, 3, 3).iloc[:, 1]  # Stochastic %D

    # Commodity Channel Index
    df["ta_cci"] = df.ta.cci(20)

    # Average Directional Index
    # df['ta_adx'] = df.ta.adx(14).iloc[:, 0]
    # df['ta_adx_dmp'] = df.ta.adx(14).iloc[:, 1]
    df["ta_adx_dmn"] = df.ta.adx(14).iloc[:, 2]

    # Awesome Oscillator
    df["ta_ao"] = df.ta.ao()

    # Momentum
    # df['ta_mom'] = df.ta.mom(10)

    # MACD Level
    df["ta_macd"] = df.ta.macd(12, 26).iloc[:, 0]

    # Stochastic RSI Fast
    # df['ta_rsistochk'] = df.ta.stoch(14, 14, 3, 3).iloc[:, 0]
    df["ta_rsistochd"] = df.ta.stoch(14, 14, 3, 3).iloc[:, 1]

    # Williams Percentage Range
    # df['ta_williams'] = df.ta.willr(14)

    # Ultimate Oscillator
    # df['ta_uo'] = df.ta.uo(7, 14, 28)

    df = df.rename(columns={"close": symbol, "low": symbol_low, "high": symbol_high})

    return df


def tech_ma(df, symbol):
    df = df.rename(columns={symbol: "close"})

    # Simple averages
    df["ta_sma10"] = df.ta.sma(10)
    df["ta_sma20"] = df.ta.sma(20)
    # df['ta_sma30'] = df.ta.sma(30)
    # df['ta_sma50'] = df.ta.sma(50)
    # df['ta_sma100'] = df.ta.sma(100)
    # df['ta_sma200'] = df.ta.sma(200)

    # Exponential averages
    df["ta_ema10"] = df.ta.ema(10)
    df["ta_ema20"] = df.ta.ema(20)
    # df['ta_ema30'] = df.ta.ema(30)
    # df['ta_ema50'] = df.ta.ema(50)
    # df['ta_ema100'] = df.ta.ema(100)
    df["ta_ema200"] = df.ta.ema(200)

    # Ichimoku Cloud
    # df['ta_isa_9'] = df.ta.ichimoku(9, 26, 52, 26).iloc[:, 0]
    # df['ta_isb_26'] = df.ta.ichimoku(9, 26, 52, 26).iloc[:, 1]
    # df['ta_its_9'] = df.ta.ichimoku(9, 26, 52, 26).iloc[:, 2]
    # df['ta_iks_26'] = df.ta.ichimoku(9, 26, 52, 26).iloc[:, 3]

    # Volume Weighted Moving Average
    # df['ta_vwma20'] = df.ta.vwma(20)

    # Hull Moving Average
    df["ta_hma10"] = df.ta.hma(10)

    df = df.rename(columns={"close": symbol})

    return df


def build_usdmxn(info_dict):
    # Get dataframes with data of each series
    dfs = get_series(info_dict)

    # Get min_date for data frames
    # Information for cuenta_capital, cuenta_corriente is only available since 2002-03-31,
    # disregarding that information and training since 1995 has worse performance
    min_date = datetime(2002, 3, 31)
    fact_ppp_0203 = 897.95 / 720.614
    # min_date = datetime(1995, 1, 1)

    # Join data frames into one data frame
    join = join_series(dfs)
    join = join.interpolate("linear")
    join = join[join.index >= min_date]
    logger.info("Series starts at: %s", join.index[0])
    min_date = join.index[0]

    vars_required = ["INPC", "CPIAUCNS", "USDMXN", "CETES28", "FEDFUNDS"]
    if len(vars_required) > len(set(vars_required) & set(join.columns)):
        logger.info(
            "Unable to calculate difference of rates and inflation due to missing columns"
        )
        return join

    join["inf_mex"] = join["INPC"] / join["INPC"].loc[min_date] - 1
    join["inf_us"] = join["CPIAUCNS"] / join["CPIAUCNS"].loc[min_date] - 1
    join["usdmxn_inc"] = join["USDMXN"] / join["USDMXN"].loc[min_date] - 1

    join["inf_inc"] = (1 + join["inf_mex"]) / (1 + join["inf_us"]) - 1

    join["ppp_value"] = (
        fact_ppp_0203 * (1 + join["inf_inc"]) / (1 + join["usdmxn_inc"]) - 1
    )
    join["dif_rates"] = (
        (1 + join["CETES28"] / 100) / (1 + join["FEDFUNDS"] / 100) - 1
    ) * 100

    join = join.drop(["inf_mex", "inf_us", "usdmxn_inc", "inf_inc"], axis=1)

    # Seems to have better performance since december 2003
    min_date = datetime(2003, 12, 1)
    join = join[join.index >= min_date]

    # drop missing values
    join = join.dropna(how="any", axis=0)

    info_dict["DIF"] = {}
    info_dict["dif_rates"] = {}

    # Add technical indicators
    ## join = join.drop(['USDMXN_LOW', 'USDMXN_HIGH'], axis=1)
    tech = join[["USDMXN", "USDMXN_LOW", "USDMXN_HIGH"]]
    tech = tech_oscillators(tech, "USDMXN", "USDMXN_LOW", "USDMXN_HIGH")
    tech = tech_ma(tech, "USDMXN")
    # tech = tech.drop(['usdmxn_high'], axis=1)
    # tech = tech.dropna()

    join = join.drop(["USDMXN_LOW", "USDMXN_HIGH"], axis=1)
    tech = tech.drop(["USDMXN"], axis=1)
    join = join.merge(tech, left_index=True, right_index=True)
    join = join.drop("USDMXN_HIGH", axis=1)
    join = join.dropna()

    return join


def build_std(info_dict):
    # Get dataframes with data of each series
    dfs = get_series(info_dict)

    # Get min_date for data frames
    min_date = max([df.index.min() for df in dfs])

    # Join data frames into one data frame
    join = join_series(dfs)
    join = join.interpolate("linear")
    join = join.loc[min_date:]

    # drop missing values
    join = join.dropna(how="any", axis=0)

    return join


def move_by_freq(df, freq):
    # freq could also be calculated from series - leave that for later
    if freq == "M":
        df.index = df.index + MonthEnd(1)
    elif freq == "Q":
        df.index = df.index + QuarterEnd(1)

    return df


def get_data_yahoo(name, start, end, yahoo_ticker, yahoo_col="Adj Close"):
    logger.info(f"Downloading data from YAHOO for {name} (ticker: {yahoo_ticker}, col: {yahoo_col})")
    df = yf.download(yahoo_ticker, start, end, progress=False)

    if df.empty:
        logger.warning(f"No data returned for {yahoo_ticker}")
        return pd.DataFrame()

    df.index = df.index.tz_localize(None)
    df.index.name = "DATE"

    # Handle MultiIndex columns (common in recent yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten to single level (e.g. 'Adj Close' or 'Low')
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        # Or more robust: join levels if needed
        # df.columns = ['_'.join(str(c) for c in col if c).strip() for col in df.columns.values]

    # Now select the column safely
    possible_cols = [yahoo_col, yahoo_col.lower(), yahoo_col.upper()]
    selected_col = None
    for c in possible_cols:
        if c in df.columns:
            selected_col = c
            break

    if selected_col is None:
        # Fallback: take first available numeric column
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            selected_col = numeric_cols[0]
            logger.warning(f"Requested col '{yahoo_col}' not found for {yahoo_ticker}. Using '{selected_col}' instead.")
        else:
            logger.error(f"No numeric columns found for {yahoo_ticker}")
            return pd.DataFrame()

    df = df[[selected_col]].copy()
    df = df.rename(columns={selected_col: name})

    # Final safety: ensure single level
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [name]  # force it

    return df

    
def get_data_fred(name, start, end, freq):
    logger.info("Downloading data from FRED for " + str(name))

    FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?"
    var_ws = (
        FRED_URL
        + "cosd="
        + start.strftime("%Y-%m-%d")
        + "&coed="
        + end.strftime("%Y-%m-%d")
        + "&id="
        + name
    )

    # Read without header assumption, parse first column as date
    df = pd.read_csv(
        var_ws,
        index_col=0,                # first column becomes index
        parse_dates=True,
        header=0,                   # assume first row is header (but we ignore name)
        na_values=na_values,
        dayfirst=False,
        engine='python'             # fallback to python parser if C fails strangely
    )

    # Rename index to standard name (optional but clean)
    df.index.name = "DATE"

    df = move_by_freq(df, freq)

    # If the value column is still unnamed or wrong, rename it
    if len(df.columns) == 1:
        df.columns = [name]

    return df


def get_data_stockpup(name, filename, colname):
    """

    :param name: Name that will be assigned to desired column
    :param filename: Name of the file to be extracted from web page
    :param colname: Original column name in file
    :return:
    """

    STOCKPUP_URL = "http://www.stockpup.com/data/"
    var_ws = STOCKPUP_URL + filename

    df = pd.read_csv(
        var_ws, index_col="Quarter end", parse_dates=True, na_values=na_values
    )

    df = df.rename(columns={colname: name})

    return df


def get_data_banxico(name, start, end, bmx_series, freq):
    logger.info("Downloading data from BANXICO for " + str(name))

    df = dwld_bmx([bmx_series], start, end)[0]

    df = df.rename(columns={bmx_series: name})

    df = move_by_freq(df, freq)

    return df


def get_data_inegi(name, series):
    logger.info("Downloading data from INEGI for " + str(name))

    df = dwld_inegi(series)
    df = df.rename(columns={series: name})

    return df


def get_data_multpl(name):
    logger.info("Downloading data from MULTPL for " + str(name))

    df = dwld_multpl(name)

    return df


def get_data(target):
    info_dict = data_predict[target]
    if target == "USDMXN":
        return build_usdmxn(info_dict)
    else:
        return build_std(info_dict)


def create_ts(data, target, pred_interval, n_steps, ordered=False):
    scalers = {}
    info_dict = data_predict[target]

    # For each column in the data set:
    # 1. Scale the data
    # 2. Form as many as possible examples from the time series, this means create many time series.
    # Each of these time series is formed from the original by spacing by the prediction interval -
    # starting at a shift of plus i % pred_interval.
    # 3. Create dataset
    for col in data:
        # Apply log to the series, since an error (for MSE) of 10% should be the same @ $1 than @ $100
        # So without log mse for 10% @1 = (1.01 - 1)^2 = 1E-4 != (101 - 100)^2 = 1
        # But for log mse for 10% @1 = (log(1.01) - log(1))^2 = 9.9E-5 = (log(101) - log(100))^2

        if "transform" in info_dict[col]:
            trans_func = info_dict[col]["transform"]
            data[col] = trans_func(data[col])

            if np.isnan(data[col]).any():
                logger.warning(
                    f'Nans detected in column {col} when applying transformation {info_dict[col]["transform"]}'
                )

        # Next scale from 0 to 1
        scalers[col] = MinMaxScaler(feature_range=(0, 1))
        data.loc[:, col] = scalers[col].fit_transform(
            data.loc[:, col].values.reshape(-1, 1)
        )

    values = data.values

    # Create lags - CHECK N_STEPS
    no_ex = values.shape[0] - n_steps * pred_interval
    if no_ex < 0:
        logger.warning(
            f"Cannot create examples with given time series, there are: {values.shape[0]} values,and"
            f" number of steps is {n_steps} and prediction interval is {pred_interval}"
        )
        n_steps = math.floor(values.shape[0] / pred_interval)
        no_ex = values.shape[0] - n_steps * pred_interval
        logger.warning(f"Number of steps updated to {n_steps}")
    start = 0
    end = no_ex
    lags = [values[start:end]]
    for i in range(n_steps):
        start += pred_interval
        end += pred_interval
        lags.append(values[start:end])

    # stack lags in middle axis, so that final shape is examples, steps/ lags, columns
    res = np.stack(lags, axis=1)

    X_score = res[-1:, 1:, :]

    if ordered:
        X_data = res[:, :-1, :]
        y_data = res[:, 1:, 0:1]
    else:
        shuffle_index = np.random.permutation(res.shape[0])
        X_data = res[shuffle_index, :-1, :]
        y_data = res[shuffle_index, 1:, 0:1]

    return res, X_data, y_data, X_score, scalers, n_steps


def split_train_test(data, test_pc=0.2):
    no_rows = data.shape[0]
    split = round(no_rows * (1 - test_pc))
    train = data.iloc[:split, :]
    test = data.iloc[split:, :]

    return train, test


def inverse_scale(arr, scalers, col_predict):
    arr = scalers[col_predict].inverse_transform(arr.reshape(-1, 1))
    return np.exp(arr)
