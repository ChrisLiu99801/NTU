import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der
from datetime import datetime
import math
figsize = (14, 8) 

st.set_page_config(
    page_title="今晚我想來點．．．",
    page_icon="random",
    layout="centered",
)

#今天日期、時間、星期幾
import datetime
import time
from datetime import date
import calendar
current_date = date.today()
current_weekday = calendar.day_name[current_date.weekday()]
localtime = time.localtime()
current_time = time.strftime("%H:%M", localtime)
time_result = current_time + ', ' + current_weekday

# 輸入體量
st.title(time_result + ':sunglasses:')
st.subheader('請輸入投資總額')
total_value = st.text_input('', key = 1)
if not total_value:
    st.stop()

#輸入標的
ticker_crawler_list = [] # 紀錄用戶輸入的標的代碼，必須要跟yfinance同格式 xxxx.TW
temp = ""
k = 2
while temp != "end":
    st.title(time_result + ':sunglasses:')
    st.subheader('請只輸入同一個市場的標的,並以xxxx.TW格式輸入')
    st.caption('結束時輸入end') 
    temp = st.text_input("", key = k)
    k = k + 1
    if not temp:
        st.stop()
    if temp != "end":
        ticker_crawler_list.append(temp)
    

# yfinance 抓資料
#測試用ticker_crawler_list = [0050.TW', '2330.TW] # 把用戶輸入的標的代碼，必須要跟yfinance同格式 xxxx.TW
tickers_information = [] # 股票的總資料(未經處理)
for ticker in ticker_crawler_list:
    tick = yf.Ticker(ticker)
    ticker_history = tick.history(period='max') # 時間長度固定為所有
    ticker_history['Symbol'] = ticker # 加上其股票代碼
    tickers_information.append(ticker_history) 


# 資料整理 (股價)
ticker_pd = pd.concat(tickers_information)
ticker_pd = ticker_pd.reset_index()
ticker_pd = ticker_pd[['Date', 'Close', 'Symbol']]
ticker_pd = ticker_pd.drop_duplicates()
ticker_pd['Date'] = pd.to_datetime(ticker_pd['Date'], utc=True).dt.strftime('%Y-%m-%d')
tickers_price = ticker_pd.pivot(index='Date', columns='Symbol', values='Close').reset_index()
tickers_price.index = tickers_price.Date
tickers_price.drop(columns=['Date'], inplace=True)

#投資組合中每間公司在過去190個月中的月報酬
tickers_price.index = pd.to_datetime(tickers_price.index)
tickers_monthly_price = tickers_price.resample("1m").agg("last") # 每個月的最後一天作指標，計算當月報酬
tickers_monthly_return = tickers_monthly_price.pct_change() # 以百分比來表達月報酬率
tickers_monthly_return = tickers_monthly_return.iloc[-191:-1] # 用這裡是因為希望計算包括 台股在不同時段的漲跌幅

#投資組合每月的期望報酬Expected Returns(mean)
tickers_expected_return = tickers_monthly_return.mean()
tickers_expected_return *= 12

#投資組合的標準差standard
tickers_std = tickers_monthly_return.std()
tickers_std *= np.sqrt(12)

#投資組合的風險分散能力(個股間漲跌的連動性) Covariance between stocks
tickers_cov = tickers_monthly_return.cov()
tickers_cov_metics = tickers_monthly_return.apply(lambda x: np.log(1+x)).cov()

#個別投資標的bp值
#計算大盤月報酬、計算大盤std、大盤E(r)
TWII = pd.read_csv("C:/Andrew/^TWII.csv")
TWII_return = TWII["Adj Close"].pct_change()

TWII_std = TWII_return.std()
TWII_Er = TWII_return.mean()

#計算個別投資標的對大盤cov
tickers_monthly_return_addTWII = tickers_monthly_return
tickers_monthly_return_addTWII["TWII"] = list(TWII_return)[1:len(list(TWII_return))]
tickers_covaddTWII = tickers_monthly_return_addTWII.cov()

#計算個別投資標的bp值
bp_dict = {}
for i in range(len(ticker_crawler_list)):
    bp_dict[ticker_crawler_list[i]] = tickers_covaddTWII.loc["TWII"][0:len(tickers_covaddTWII.loc["TWII"])-1][i] / (TWII_std ** 2)
tickers_bp = pd.DataFrame(bp_dict, index=[0]).transpose()

#投資組合分析
tickers_assets = pd.concat([tickers_expected_return, tickers_std], axis=1)
tickers_assets.columns = ['報酬率', '波動率']

# p stands for portfolio
tickers_p_return = [] # 組合報酬率
tickers_p_volume = [] # 
tickers_p_weights = [] # 各投資權重
tickers_p_bp = [] #投資組合的bp

tickers_num_assets = len(tickers_price.columns) # 標的數量
tickers_num_portfolios = 100000 # 假設做100000個投資組合

for tickers_portfolio in range(tickers_num_portfolios):
    tickers_weights = np.random.random(tickers_num_assets)
    tickers_weights = tickers_weights/np.sum(tickers_weights)
    tickers_p_weights.append(tickers_weights)
    tickers_returns = np.dot(tickers_weights, tickers_expected_return)
    tickers_p_return.append(tickers_returns)
    tickers_var = np.dot(tickers_weights.T, np.dot(tickers_cov, tickers_weights))
    tickers_sd = np.sqrt(tickers_var)
    tickers_ann_sd = tickers_sd*np.sqrt(12)
    tickers_p_volume.append(tickers_ann_sd)
    tickers_bp_temp = np.dot(tickers_weights, tickers_bp)
    tickers_p_bp.append(tickers_bp_temp[0])

tickers_data = {'投資組合報酬率':tickers_p_return, '投資組合波動率':tickers_p_volume, "投資組合bp":tickers_p_bp}
pd.DataFrame( tickers_data )

for tickers_counter, tickers_symbol in enumerate(tickers_price.columns.tolist()):
    tickers_data[tickers_symbol+' 的權重'] = [w[tickers_counter] for w in tickers_p_weights]

rf = 0.015 # 假設無風險報酬率為 0.015 這個數值可討論
tickers_portfolio = pd.DataFrame(tickers_data)
tickers_portfolio['投資組合的夏普率Sharpe Ratio'] = (tickers_portfolio['投資組合報酬率']-rf) / tickers_portfolio['投資組合波動率']
tickers_portfolio['投資組合的崔納指標'] = (tickers_portfolio['投資組合報酬率']-rf) / tickers_portfolio['投資組合bp']
tickers_portfolio['投資組合的詹森指標'] = ((tickers_portfolio['投資組合報酬率']-rf) - (TWII_Er - rf)) * tickers_portfolio['投資組合bp']
tickers_portfolio.head()

#達到預期報酬率水平的最低可能風險之投資組合
#tickers_portfolio[tickers_portfolio['投資組合波動率']==tickers_portfolio['投資組合波動率'].min()]
tickers_min_var_port = tickers_portfolio.iloc[tickers_portfolio['投資組合波動率'].idxmin()]

#最大化夏普率之投資組合
((tickers_portfolio['投資組合報酬率']-rf)/tickers_portfolio['投資組合波動率']).idxmax()

tickers_max_sharpe_port = tickers_portfolio.iloc[((tickers_portfolio['投資組合報酬率']-rf)/tickers_portfolio['投資組合波動率']).idxmax()]

#最大Treynor Ratio(崔納指標)
tickers_max_treynor_ratio = tickers_portfolio.iloc[((tickers_portfolio['投資組合報酬率']-rf)/tickers_portfolio['投資組合bp']).idxmax()]

#最大詹森大市逆市上升指標(Jensen Ratio)α=（Rp-RF）-（Rm-RF）* βp
tickers_max_jensen_ratio = tickers_portfolio.iloc[(((tickers_portfolio['投資組合報酬率']-rf) - (TWII_Er - rf)) * tickers_portfolio['投資組合bp']).idxmax()]
round(tickers_max_jensen_ratio[3], 5)
round(tickers_max_jensen_ratio[4], 5)


output = " "
# 輸出
st.subheader(' 建議最佳投資組合：')
for i in range(len(ticker_crawler_list)) :
    st.header(ticker_crawler_list[i] + ":arrow_right:" + str(float(format(tickers_max_jensen_ratio[i + 3], '.5f'))*100))

'''
left_col, mid_col, right_col = st.columns(3)
st.text('Google評分是' + str(output[1]) + '⭐️')
st.text('共有' + str(output[2]) + '則評論')
st.text('價錢定位是 ' + str('💲' * int(output[-1])))
st.subheader('祝您用餐愉快')'''
