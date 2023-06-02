import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen, rosen_der
from datetime import datetime
import math
from PIL import Image
figsize = (14, 8) 
   
st.set_page_config(
    page_title="Portfolio Analysis",
    page_icon="random",
    layout="centered"
)

st.image(S__17105008.jpg, caption='Sunrise by the mountains')

st.markdown("<h1 style='text-align: center; color: black;'>Analysmart 投資組合分析</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'> </h1>", unsafe_allow_html=True)


# 輸入體量
st.subheader('請輸入投資總額')
k = 1
while k < 3 :
    try :
        total_value = st.text_input(' ', key = 1)
        k = int(total_value) + 5
        if not total_value:
            st.stop()
    except :
        st.info('請以羅馬數字格式輸入')
        st.stop()

#輸入標的
ticker_crawler_list = [] # 紀錄用戶輸入的標的代碼，必須要跟yfinance同格式 xxxx.TW
temp = ""
k = 2
while temp != "end":
    st.subheader('請只輸入台股的投資標的,並以xxxx.TW格式輸入')
    st.caption('結束時輸入end') 
    temp = st.text_input("", key = k)
    k = k + 1
    if not temp:
        st.stop()
    if temp != "end":
        ticker_crawler_list.append(temp)

with st.spinner('Wait for it...'):
    try : 
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
        TWII = pd.read_csv("TWII.csv")
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
        tickers_portfolio['市場標準差'] = TWII_std
        tickers_portfolio['市場期望報酬'] = TWII_Er        
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
    except :
        e = RuntimeError('您需要輸入兩個以上的台股標的且須為XXXX.TW格式')
        st.exception(e)
        st.stop()

         
st.balloons()
definition = st.sidebar.subheader("Treynor Ratio (崔納指標)")
definition_treynor = st.sidebar.markdown("崔納指數是指，承擔每一單位的「系統風險」，可以獲得多少單位的風險溢酬。崔納指標愈大，表示該投資標的績效愈佳。")
definition = st.sidebar.subheader("Jensen Index (詹森指標)")
definition_treynor = st.sidebar.markdown("詹森指標是指投資組合的績效，是否優於相同風險水準下其他組合的績效。詹森指標可為正或負，正值愈大表示績效愈好。")
definition = st.sidebar.subheader("Sharp Ratio (夏普比率)")
definition_treynor = st.sidebar.markdown("夏普指標，是指承擔每一單位的總風險，可獲得多少單位的風險溢酬。")
definition = st.sidebar.subheader("β值（Beta)")
definition_treynor = st.sidebar.markdown("β值一般是被用來衡量一支股票的風險大小，如果β為1，則其波動性與市場一致。如果β小於1，則其波動性較廣泛市場小；如果大於1，則其波動性較廣泛市場大。")

subheader_text = []
st.header('投資組合的判斷基準_臺灣加權指數：')
col1, col2, col3 = st.columns(3)
col1.metric("市場波動率:", "{:.2%}".format(TWII_std))
col2.metric("市場期望報酬:", "{:.2%}".format(TWII_Er))

#最低風險
subheader_text = []
for i in range(len(ticker_crawler_list)):
    subheader_text.append(ticker_crawler_list[i] + ":arrow_right:" + str(format(float(tickers_min_var_port[i + 3])*100, '.3f')) + "%" + " (" + str(format(float(tickers_min_var_port[i + 3])*float(total_value), '.2f')) + " 元)")

st.header('達到預期報酬的最低風險組合：')
for text in subheader_text:
    st.subheader(text)

col1, col2, col3 = st.columns(3)
col1.metric("投資組合報酬率", format(tickers_min_var_port[0], '.3f'))
col2.metric("投資組合波動率", format(tickers_min_var_port[1], '.3f'))
col3.metric("投資組合BP", format(tickers_min_var_port[2], '.3f'))

col4, col5, col6 = st.columns(3)
col4.metric("投資組合Sharpe Ratio", format(tickers_min_var_port[3], '.3f'))
col5.metric("投資組合崔納指標", format(tickers_min_var_port[4], '.3f'))
col6.metric("投資組合詹森指標", format(tickers_min_var_port[7], '.3f'))
st.divider()

#最大化夏普率之投資組合
# for i in range(len(ticker_crawler_list)) :
#      st.subheader(ticker_crawler_list[i] + ":arrow_right:" + str(format(float(tickers_max_sharpe_port[i + 3])*100, '.3f')) + "%")
subheader_text = []
for i in range(len(ticker_crawler_list)):
    subheader_text.append(ticker_crawler_list[i] + ":arrow_right:" + str(format(float(tickers_max_sharpe_port[i + 3])*100, '.3f')) + "%" + " (" + str(format(float(tickers_max_sharpe_port[i + 3])*float(total_value), '.2f')) + " 元)")

st.header('最大化夏普率組合：')
for text in subheader_text:
    st.subheader(text)

# col1, col2, col3 = st.columns(3)
# col1.metric("投資組合報酬率", format(tickers_max_sharpe_port[0], '.3f'))
# col2.metric("投資組合波動率", format(tickers_max_sharpe_port[1], '.3f'))
# col3.metric("投資組合BP", format(tickers_max_sharpe_port[2], '.3f'))

# col4, col5, col6 = st.columns(3)
# col4.metric("投資組合Sharpe Ratio", format(tickers_max_sharpe_port[3], '.3f'))
# col5.metric("投資組合崔納指標", format(tickers_max_sharpe_port[4], '.3f'))
# col6.metric("投資組合詹森指標", format(tickers_max_sharpe_port[5], '.3f'))

# col7, col8, col9 = st.columns(3)
# col8.metric("市場標準差", format(tickers_max_sharpe_port[8], '.3f'))
# col9.metric("市場期望報酬", format(tickers_max_sharpe_port[9], '.3f'))

col1, col2, col3 = st.columns(3)
col1.metric("投資組合報酬率", format(tickers_max_sharpe_port[0], '.3f'))
col2.metric("投資組合波動率", format(tickers_max_sharpe_port[1], '.3f'))
col3.metric("投資組合BP", format(tickers_max_sharpe_port[2], '.3f'))

col4, col5, col6 = st.columns(3)
col4.metric("投資組合Sharpe Ratio", format(tickers_max_sharpe_port[3], '.3f'))
col5.metric("投資組合崔納指標", format(tickers_max_sharpe_port[4], '.3f'))
col6.metric("投資組合詹森指標", format(tickers_max_sharpe_port[5], '.3f'))

st.divider()
#最大崔納指標組合
# for i in range(len(ticker_crawler_list)) :
#     st.subheader(ticker_crawler_list[i] + ":arrow_right:" + str(format(float(tickers_max_treynor_ratio[i + 3])*100, '.3f')) + "%")

subheader_text = []
for i in range(len(ticker_crawler_list)):
    subheader_text.append(ticker_crawler_list[i] + ":arrow_right:" + str(format(float(tickers_max_treynor_ratio[i + 3])*100, '.3f')) + "%" + " (" + str(format(float(tickers_max_treynor_ratio[i + 3])*float(total_value), '.2f')) + " 元)")

st.header('最大崔納指標組合：')
for text in subheader_text:
    st.subheader(text)


col1, col2, col3 = st.columns(3)
col1.metric("投資組合報酬率", format(tickers_max_treynor_ratio[0], '.3f'))
col2.metric("投資組合波動率", format(tickers_max_treynor_ratio[1], '.3f'))
col3.metric("投資組合BP", format(tickers_max_treynor_ratio[2], '.3f'))

col4, col5, col6 = st.columns(3)
col4.metric("投資組合Sharpe Ratio", format(tickers_max_treynor_ratio[3], '.3f'))
col5.metric("投資組合崔納指標", format(tickers_max_treynor_ratio[4], '.3f'))
col6.metric("投資組合詹森指標", format(tickers_max_treynor_ratio[5], '.3f'))
st.divider()



#最大詹森
# for i in range(len(ticker_crawler_list)) :
#     st.subheader(ticker_crawler_list[i] + ":arrow_right:" + str(format(float(tickers_max_jensen_ratio[i + 3])*100, '.3f')) + "%")

subheader_text = []
for i in range(len(ticker_crawler_list)):
    subheader_text.append(ticker_crawler_list[i] + ":arrow_right:" + str(format(float(tickers_max_jensen_ratio[i + 3])*100, '.3f')) + "%" + " (" + str(format(float(tickers_max_jensen_ratio[i + 3])*float(total_value), '.2f')) + " 元)")

st.header('最大詹森指標組合：')
for text in subheader_text:
    st.subheader(text)


col1, col2, col3 = st.columns(3)
col1.metric("投資組合報酬率", format(tickers_max_jensen_ratio[0], '.3f'))
col2.metric("投資組合波動率", format(tickers_max_jensen_ratio[1], '.3f'))
col3.metric("投資組合BP", format(tickers_max_jensen_ratio[2], '.3f'))

col4, col5, col6 = st.columns(3)
col4.metric("投資組合Sharpe Ratio", format(tickers_max_jensen_ratio[3], '.3f'))
col5.metric("投資組合崔納指標", format(tickers_max_jensen_ratio[4], '.3f'))
col6.metric("投資組合詹森指標", format(tickers_max_jensen_ratio[5], '.3f'))
