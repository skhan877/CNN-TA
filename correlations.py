import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import math
from networkx.drawing.nx_pydot import graphviz_layout



def returns_df(data):
	
  web = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
  source = pd.read_html(web)
  spx_tickers = source[0][['Symbol','GICS Sector']]

  stocks = [s for s in spx_tickers['Symbol']]
  df = pd.DataFrame()

  for stock in stocks:
    df[stock] = data['Close'][data['symbol'] == stock]
    df[stock] = df[stock].astype('float64')
    df[stock] = np.log(df[stock]) - np.log(df[stock].shift(1))

  df = df.dropna(axis=1, how='all')
  stocks = df.columns.values.tolist()

  return df, stocks


def returns_df_year(df, year):
  
  start_date = str(year) + '-01-01'
  end_date = str(year+1) + '-01-01'
  df = df[(df.index >= start_date) & (df.index < end_date)]
  df = df.dropna(axis=1, how='all')

  stocks = sorted(df.columns.values.tolist())

  return df, stocks


def sectors():
  
  web = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
  source = pd.read_html(web)
  spx_tickers = source[0][['Symbol','GICS Sector']]

  sectors = [sect for sect in spx_tickers['GICS Sector'].unique()]
  colours = ['red', 'blue', 'green', 'yellow', 'magenta', 'orange', 'aqua', 'purple', 'violet', 'brown', 'grey']
  sector_colours = {}
  
  for i in range(len(sectors)):
    sector_colours[sectors[i]] = colours[i]

  return spx_tickers, sector_colours


def returns_analysis(data):
  '''
  Plot distribution of price returns
  '''
  plt.figure(figsize=(5,4))
  plt.hist(data['Close'].pct_change(1), bins=80, cumulative=True, histtype='step')
  # plt.hist(np.log(data['Close']), bins=100)
  plt.title(data['symbol'][0] + ': Distribution of daily returns')
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

  plt.title(data['symbol'][0] + ': Trailing 12mth Volatility')
  plt.xlabel('date')
  plt.ylabel('stdev of price')
  plt.show()


### below code adapted from DSTA lab session ###

def mean(X):
  m = 0.0
  for i in X:
    m += i
  
  return m/len(X)


def covariance(X, Y):
  c = 0.0
  for i in range(len(X)):
    c += ((X[i]-mean(X)) * (Y[i]-mean(Y)))
  return c/len(X)


def pearson(X, Y):
  return covariance(X,Y) / ((covariance(X,X)**0.5) * (covariance(Y,Y)**0.5))


def corr_coeff(X, Y):

  X, Y = X.dropna(), Y.dropna()
  X_returns, Y_returns = [], []

  intersecting_dates = (X.index).intersection(Y.index)
  for date in intersecting_dates:
    X_returns.append(float(X.loc[date]))
    Y_returns.append(float(Y.loc[date]))

  return pearson(X_returns, Y_returns)


def corr_heatmap(data):

  df = returns_df()[0]

  plt.figure(figsize=(10,10))
  sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=False, cmap='coolwarm', xticklabels=False, yticklabels=False)
  plt.show()


def build_network(returns_data):
	
  corr_network = nx.Graph()

  stocks = returns_data[1]
  returns_data = returns_data[0]
  num_stocks = len(stocks)
  n = 1

  for i in range(num_stocks-1):
    for j in range(i+1, num_stocks):
      a = stocks[i] 
      b = stocks[j]
      try:
        distance = math.sqrt(2*(1.0 - corr_coeff(returns_data[a], returns_data[b])))
        corr_network.add_edge(a, b, weight=distance)
      except Exception as e:
        print(e, a, b)
      n += 1

  print('Complete!')
  print(f'no of nodes: {corr_network.number_of_nodes()}')
  print(f'no of edges: {corr_network.number_of_edges()}')

  return corr_network


def draw_network(corr_network):

  plt.figure(figsize=(8,8))
  nx.draw(corr_network, with_labels=True)


def mst(corr_network):
  # Minimum Spanning Tree (Prims Algorithm)

  seed = 'MMM'
  nodes = []
  edges = []

  nodes.append(seed)

  while len(nodes) < corr_network.number_of_nodes():
    min_weight = 1000000.0
    for node in nodes:
      for neighbour in corr_network.neighbors(node):
        if not neighbour in nodes:
          if corr_network[node][neighbour]['weight'] < min_weight:
            min_weight = corr_network[node][neighbour]['weight']
            min_weight_edge = (node, neighbour)
            neighbour_ext = neighbour
    edges.append(min_weight_edge)
    nodes.append(neighbour_ext)


  tree = nx.Graph()
  tree.add_edges_from(edges)

  sector_lookup = sectors()[0]
  colour_lookup = sectors()[1]

  plt.figure(figsize=(15,15))
  pos = graphviz_layout(tree, prog='neato')
  nx.draw_networkx_edges(tree, pos, width=2, edge_color='black', alpha=0.5, style='solid')
  nx.draw_networkx_labels(tree, pos)

  for node in tree.nodes():
    col = colour_lookup[sector_lookup[sector_lookup['Symbol']==node]['GICS Sector'].iloc[0]]
    nx.draw_networkx_nodes(tree, pos, [node], node_size=600, node_color=col, alpha=0.5, label=node)

  markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', linestyle='') for color in colour_lookup.values()]
  plt.legend(markers, colour_lookup.keys(), numpoints=1)



if __name__ == '__main__':

  f = '/content/drive/MyDrive/Stock_Data.csv'
  data = pd.read_csv(f, index_col=0, parse_dates=True)

  full_df = returns_df(data)[0]
  df2008 = returns_df_year(full_df, 2008)
  df2016 = returns_df_year(full_df, 2016)

  network = build_network(df2016)
  draw_network(network)
  mst(network)

