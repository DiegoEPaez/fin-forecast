""" Module for querying Banxico API """
from datetime import datetime
import logging
import pandas as pd
import requests

# BANXICO: https://www.banxico.org.mx/SieAPIRest/service/v1/doc/catalogoSeries
BMX_TOKEN = "baaeab1b011e38e513f7b233e018828422d9ac973321b55ac122fb122737bcfb"


def query_bmx(series, start_date, end_date):
    """
    Queries "Banxico" API to get specified series (for example CETES rate 28 days) in time
    frame given
    :param series:     list of series to query for
    :param start_date: begin date for query
    :param end_date:   end date for query
    :return: a JSON object built from API response
    """

    url_1 = "https://www.banxico.org.mx/SieAPIRest/service/v1/series/"
    url_2 = "/datos/"

    logging.info("Querying Banxico...")
    headers = {"Bmx-Token": BMX_TOKEN, "Content-type": "application/json"}

    str_series = ",".join(series)
    url = url_1 + str_series + url_2 + start_date + "/" + end_date

    response = requests.get(url, headers=headers, timeout=60)

    if response.status_code == 200:
        logging.debug("Obtained response from Banxico")

        data = response.json()
        return data

    logging.debug("Response status code was not successful from Banxico API")
    return None


def dwld_bmx(series, start_dt, end_dt):
    """
    Dwld Government Bonds series (rates, terms) from Banxico API starting at given
    :param series:  list of series to dwld
    :param start_dt:
    :param end_dt:
    :return:  List of dataframes for each series
    """
    if not isinstance(series, list):
        if isinstance(series, str):
            series = [series]
        else:
            raise ValueError(
                f"Could not understand series {series} only lists and strings are accepted values"
            )

    logging.info("Downloading data from BANXICO for series %s",  series)
    sdate = start_dt.strftime("%Y-%m-%d")
    edate = end_dt.strftime("%Y-%m-%d")
    data = query_bmx(series, sdate, edate)

    if not data:
        return None

    return obtain_dfs(data, series)


def obtain_dfs(data, series):
    """
    Once data has been downloaded from the banxico api, format the data into dataframes
    :param data: the data downloaded from the API
    :param series: the names of the series that were downloaded
    :return: List of dataframes for each series
    """
    idata = data["bmx"]["series"]
    dfs = []

    # for each series
    for i in range(0, len(series)):
        # Current series name
        curr_series = idata[i]["idSerie"]
        idata2 = idata[i]["datos"]

        # Loop through data for current series
        data_builder = {}
        for j in range(0, len(idata2)):
            report_date = datetime.strptime(idata2[j]["fecha"], "%d/%m/%Y")
            value = idata2[j]["dato"]

            if value != "N/E":
                # Replace is used because values have commas in thousands.
                data_builder[report_date] = float(value.replace(",", ""))

        df = pd.DataFrame.from_dict(data_builder, orient="index")
        df.columns = [curr_series]
        dfs.append(df)

    return dfs
