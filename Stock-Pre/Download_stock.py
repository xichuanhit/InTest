import tushare as ts

stock = '600519'
# stock = '000001' # 上证50

data = ts.get_k_data(stock, ktype='D', start='2010-01-01')

data_path = "./SH"+stock+".csv"
data.to_csv(data_path)
