import os
import datetime
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mplfinance as mpf 
from PIL import Image
from correlations import returns_df, returns_df_year
pd.options.mode.chained_assignment = None


def load_data(path):
	'''
	Function to read raw data from csv and 
	return as a pandas dataframe object
	'''
	return pd.read_csv(path, index_col=0, parse_dates=True)


def ticker_lists(data):
	'''
	Separates all tickers from original dataset into two lists of those 
	with and without the maximum number of data points observed across all stocks
	'''
	max_len = 5033 		# hard-code to save runtime
	
	tickers = sorted(list(set(sym for sym in data['symbol'])))
	tickers_all_hist = [ticker for ticker in tickers if len(data[data['symbol']==ticker]) == max_len]
	tickers_other = [ticker for ticker in tickers if len(data[data['symbol']==ticker]) != max_len]

	return tickers_all_hist, tickers_other


def stock_data(data, symbol):
	'''
	Expands dataframe to include technical indicators
	and forecast target variables
	'''
	df_stock = data[data['symbol']==symbol]

	df_stock['RSI'] = rsi(df_stock, 14)
	df_stock['MACD'] = macd(df_stock, 12, 26, 9)[0]
	df_stock['MACD_Signal'] = macd(df_stock, 12, 26, 9)[1]
	df_stock['SMA'] = round(df_stock['Close'].rolling(window=15).mean(), 2)

	max_param = 34	# MACD signal
	df_stock = df_stock.iloc[max_param-1:]

	df_stock['forecast_5d'] = np.where(df_stock['Close'].shift(-5) > df_stock['Close'], 1, 0)
	df_stock['forecast_10d'] = np.where(df_stock['Close'].shift(-10) > df_stock['Close'], 1, 0)
	df_stock['forecast_20d'] = np.where(df_stock['Close'].shift(-20) > df_stock['Close'], 1, 0)

	return df_stock


def rsi(data, periods):
	'''
	Computes the RSI indicator
	'''
	price_chg = data['Close'].diff()
	up = price_chg.clip(lower=0)
	down = -1 * price_chg.clip(upper=0)
	ma_up = up.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
	ma_down = down.ewm(com=periods-1, adjust=True, min_periods=periods).mean()
	rsi = ma_up / ma_down
	rsi = 100 - (100 / (1 + rsi))
	
	return round(rsi, 3)


def macd(data, slow, fast, signal):
	'''
	Computes the MACD indicator
	'''
	sma_slow = data['Close'].rolling(window=slow).mean()
	sma_fast = data['Close'].rolling(window=fast).mean()
	macd = sma_fast - sma_slow
	signal = macd.rolling(window=signal).mean()
	
	return (round(macd, 3), round(signal, 3))


def plot_chart(data, image_dir, image_num):
	'''
	Creates candlestick chart with subplots for indicators.
	Saves as .png image in specified directory.
	'''
	matplotlib.use('Agg')
	mc = mpf.make_marketcolors(up='white', down='black', alpha=1.0,
								edge={'up':'black', 'down':'black'},
								wick={'up':'black', 'down':'black'})
	adps = [mpf.make_addplot((data['SMA']), panel=0, color='grey', width=2, alpha=0.8),
			mpf.make_addplot((data['RSI']), panel=1, color='black', width=1.5),
			mpf.make_addplot((data['MACD']), panel=2, color='black', width=2),
			mpf.make_addplot((data['MACD_Signal']), panel=2, linestyle='dashed', color='black'),
			mpf.make_addplot(((data['MACD'])-(data['MACD_Signal'])), panel=2, type='bar', color='grey', alpha=0.5),
			mpf.make_addplot((data['Volume']), panel=3, type='bar', color='#333232')]
	style = mpf.make_mpf_style(base_mpl_style='default', 
								facecolor='#ececec', 
								edgecolor='#000000', 
								mavcolors=['#000000', '#000000'],
								marketcolors=mc,
								rc={'axes.labelcolor': 'none', 
									'xtick.color':'none', 
									'ytick.color':'none',
									'xtick.labelsize':'xx-small',
									'ytick.labelsize':'xx-small'})

	mpf.plot(data, type='candle', addplot=adps, style=style, tight_layout=True,
				savefig=dict(fname=image_dir+str(data['symbol'][0])+'-original'+str(image_num), 
						dpi=60))
	plt.close('all')
	

def crop_images(image_directory):
	'''
	Crops and resizes original images, and saves 
	with anonymous name (ticker removed)
	'''
	images_as_array = []

	new_folder = 'cropped'
	if not os.path.exists(image_directory+new_folder):
		os.makedirs(image_directory+new_folder)

	n = 1
	for filename in os.listdir(image_directory):
		if os.path.isfile(image_directory+filename):
			img = Image.open(os.path.join(image_directory, filename))
			img = img.convert(mode='RGB')
			img = img.crop((25, 1, 454, 308))
			img = img.resize((128,128))
			img.save(image_directory+'cropped//image'+str(n)+'.png')
			n += 1


def classify_charts(data, forecast_period):
	'''
	Creates datasets of bullish and bearish chart images
	'''
	bullish_dir = 'chart-images//'+str(forecast_period)+'-day-forecast//bullish//'
	bearish_dir = 'chart-images//'+str(forecast_period)+'-day-forecast//bearish//'

	chart_history = 60
	n = 1

	for i in range(chart_history, len(data)):
		temp_df = data.iloc[i-chart_history+1:i+1]
		if data['forecast_'+str(forecast_period)+'d'].iloc[i] == 1:
			plot_chart(temp_df, bullish_dir, n)
		else:
			plot_chart(temp_df, bearish_dir, n)
		n += 1

	crop_images(bullish_dir)
	crop_images(bearish_dir)






if __name__ == '__main__':

	print(f'start: {datetime.datetime.now()}')

	f = '../Stock_data.csv'
	df = load_data(f)
	
	## 2008 - including enough pre-2008 data to compute features
	df = df[(df.index > '2007-11-01') & (df.index < '2009-01-01')]

	print(f'all data loaded: {datetime.datetime.now()}')

	## selecting tickers manually from MST
	tickers = ['COG', 'AMGN', 'AFL', 'MCD', 'ESS', 'FITB', 'AMAT', 'CTAS', \
				'KSU', 'CMG', 'NFLX', 'COST', 'DAL', 'MU', 'FMC', 'FAST', 'GOOG', \
				'KO', 'CMS', 'CF', 'NRG', 'WFC', 'ZION', 'FRT', 'ADM', 'T', 'DVN', \
				'XEL', 'CL', 'COO', 'WYNN', 'AIG', 'LRCX', 'CHRW', 'PG', 'OKE', \
				'NUE', 'TAP', 'CAH', 'ALL', 'YUM', 'C', 'IRM', 'PWR', 'DFS', \
				'ALB', 'ATVI', 'WMB', 'PSA', 'MDLZ', 'O', 'HIG', 'BAX', 'PXD', \
				'PEP', 'INCY', 'MAR', 'WMT', 'LH', 'MS'] 


	for ticker in tickers:
		a = stock_data(df, ticker)
		classify_charts(a, 5)
		print(ticker + ' complete: ' + str(datetime.datetime.now()))

	print(f'total end: {datetime.datetime.now()}')

