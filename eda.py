from candlestick import load_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import yfinance as yf


def spx_index():
	'''
	Retrieve S&P 500 data from Yahoo! Finance
	'''
	spx = yf.Ticker('^GSPC')
	hist = spx.history(period='max')
	hist = hist[hist.index >= '1998-01-01']

	return hist


def returns_df(data):
	'''
	Calculate log returns for given stock
	'''
	stocks = [s for s in stocks_in_spx()['Symbol']]
	df = pd.DataFrame()

	for stock in stocks:
		df[stock] = data['Close'][data['symbol'] == stock]
		df[stock] = df[stock].astype('float64')
		df[stock] = np.log(df[stock]) - np.log(df[stock].shift(1))

	return df


def returns_analysis(data):
	'''
	Plot distribution of price returns
	'''
	plt.figure(figsize=(5,4))
	plt.hist(data['Close'].pct_change(1), bins=80, cumulative=True, histtype='step')
	plt.title('Distribution of daily returns')
	plt.xlabel('1 day return')
	plt.ylabel('frequency')
	plt.show()


def volatility_analysis(data):
	'''
	Plot of 12 month price volatility 
	'''
	vol = np.log(data['Close']).rolling(250).std()
	avg_vol = np.mean(vol)

	plt.figure(figsize=(8,4))
	plt.plot(vol)
	plt.hlines(y=avg_vol, xmin=min(data.index), xmax=max(data.index), linestyles='dashed', colors='red')

	plt.title('Trailing 12mth Volatility')
	plt.xlabel('date')
	plt.ylabel('stdev of log price')
	plt.show()


def stocks_in_spx():
	'''
	Retrieve table of stocks in	S&P500 index from Wikipedia
	'''
	web = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
	
	source = pd.read_html(web)
	spx_tickers = source[0][['Symbol','GICS Sector']]

	tickers_to_drop = ['ABMD']

	for ticker in tickers_to_drop:
		spx_tickers.drop(index=spx_tickers[spx_tickers['Symbol']==ticker].index, inplace=True)

	# stock_list.to_csv('spx-tickers.csv', index=False)
	# spx_tickers = pd.read_csv('spx-tickers.csv')

	return spx_tickers


def corr_heatmap(data):
	'''
	Plots seaborn correlation heatmap
	'''
	df = returns_df()

	sns.heatmap(df.corr(), 
		vmin=-1, vmax=1, 
		annot=False, cmap='coolwarm',
		xticklabels=False, yticklabels=False)
	plt.show()




if __name__ == '__main__':

	# f = '../Stock_data.csv'
	# data = load_data(f)
	# spx_tickers = np.array(stocks_in_spx()['Symbol'])
	
	# stock = spx_tickers[0]
	# a = data[data['symbol'] == stock]
	# print(stock)
	# returns_analysis(a)
	# volatility_analysis(a)

	# corr_heatmap(returns_df(data))
	# mst(returns_df(data), spx_tickers)

	spx = spx_index()
	returns_analysis(spx)
	volatility_analysis(spx)