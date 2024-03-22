import pandas as pd
def getSP500():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    tables = pd.read_html(url)  
    sp500_table = tables[0]['Symbol']
    return sp500_table