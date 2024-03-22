import yfinance as yf
import numpy as np 
from config import config

def download_data(config):
    data = yf.Ticker(config["alpha_vantage"]["symbol"])
    hist = data.history(period="max")

    data_close_price = [float(x) for x in hist['Close'].values]
    #data_close_price.reverse()
    data_close_price = np.array(data_close_price)

    data_date = hist.index.strftime('%Y-%m-%d').tolist()

    #data_date.reverse()

    num_data_points = len(data_date)

    display_date_range = "from" , data_date[0] , "to" , data_date[num_data_points-1]

    return data_date, data_close_price, num_data_points, display_date_range