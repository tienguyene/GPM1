import pandas as pd
#Import the libraris
import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from PIL import Image
import requests
import numpy as np
import cv2

# Thay đổi tên tệp Excel và trang tính tương ứng
url = 'https://github.com/tienguyene/GPM1/raw/main/data-Fintech2023.xlsx'
sheet = 'Price'

@st.cache_data
def load_data(file_path,sheet_name):
# Đọc tệp Excel thành một DataFrame
 df = pd.read_excel(file_path, sheet_name=sheet_name, skiprows = 1)
 df.rename(columns={'Code':'Date'}, inplace=True)
 df.set_index('Date', inplace=True)

# Đổi tên mỗi mã chứng khoán thành dạng chuẩn
 import re
 for column in df.columns:
    match = re.search(r'VT:(.*?)\(P\)', column)
    if match:
        code = match.group(1)
        df.rename(columns={column: code}, inplace=True)

# Tạo 1 dictionary, trong đó key là mã chứng khoán, value tương ứng là 1 dataframe

 df_dict = {}
 for col_name in df.columns:
    col_df = df[[col_name]] 
    col_df = col_df.dropna() #Lọc các NA để mỗi mã chứng khoán bắt đầu từ ngày niêm yết
    df_dict[col_name] = col_df
 return(df_dict)

df_dict=load_data(url,sheet)
keys = df_dict.keys()
values = df_dict.values() 
#Description: This is a stock market dashboard to show some charts and data on some stock


#Add a title and an image
st.header("**Your Technical Analysis Web Application***")
st.write("""
**Visually** show technical indicator on any stock on the market
""")
image_url = 'https://github.com/tienguyene/GPM1/raw/main/image.jpg'
response = requests.get(image_url)
image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
st.image(image, use_column_width=True)

#Create a sidebar header
st.sidebar.header('Enter the stock code and type of indicator you want to see')

#Create a function to get the users input
def get_input():
   stock_symbol = st.sidebar.text_input("Enter the stock code:", "VCB")
   chart = st.sidebar.text_input("""Enter a number for the indicator:\n1: Moving\n2: Bollinger Bands\n3: RSI\n4: MACD\n5: Stochastic Oscillator""","1")
   
   return stock_symbol, chart

stock_symbol, chart = get_input()

if stock_symbol.upper() in keys:
#Create a function to get the propper company data and the propper timeframe from the user start date to the user end datee

 df= df_dict[stock_symbol.upper()]

 df.rename(columns={stock_symbol.upper(): 'close'}, inplace=True)
 if chart.upper() =="1":
      df['50_SMA'] = df['close'].rolling(window=50).mean()
      df['100_SMA'] = df['close'].rolling(window=100).mean()
      df['200_SMA'] = df['close'].rolling(window=200).mean()
                 
      plain = go.Figure()
      plain.add_trace(go.Scatter(x=df.index, y=df.close, name='Price', line=dict(color='orange', width=1)))
      plain.add_trace(go.Scatter(x=df.index, y=df['50_SMA'], name='50 MA', line=dict(color='blue', width=0.5)))
      plain.add_trace(go.Scatter(x=df.index, y=df['100_SMA'], name='100 MA', line=dict(color='green', width=0.5)))
      plain.add_trace(go.Scatter(x=df.index, y=df['200_SMA'], name='200 MA', line=dict(color='purple', width=0.5)))
      st.header(stock_symbol.upper()+" Moving Average\n")
      st.plotly_chart(plain)
 elif chart.upper() == "2" :
      from ta.volatility import BollingerBands
       # Initialize Bollinger Bands Indicator
                
            

                
      indicator_bb = BollingerBands(close=df["close"], window=20, window_dev=2)

   # Add Bollinger Bands features
      df['bb_bbm'] = indicator_bb.bollinger_mavg()
      df['bb_bbh'] = indicator_bb.bollinger_hband()
      df['bb_bbl'] = indicator_bb.bollinger_lband()

                # Add Bollinger Band high indicator
      df['bb_bbhi'] = indicator_bb.bollinger_hband_indicator()

                # Add Bollinger Band low indicator
      df['bb_bbli'] = indicator_bb.bollinger_lband_indicator()
      plain = go.Figure()
      plain.add_trace(go.Scatter(x=df.index, y=df.close, name='Price', line=dict(color='orange', width=1)))
      plain.add_trace(go.Scatter(x=df.index, y=df['bb_bbm'], name='Middle BB', line=dict(color='blue', width=0.5)))
      plain.add_trace(go.Scatter(x=df.index, y=df['bb_bbh'], name='Upper BB', line=dict(color='green', width=0.5)))
      plain.add_trace(go.Scatter(x=df.index, y=df['bb_bbl'], name='Lower BB', line=dict(color='purple', width=0.5)))
      st.header(stock_symbol.upper()+" Bollinger Band\n")
      st.plotly_chart(plain)
 elif chart.upper() == "3" :
      import ta
      from plotly.subplots import make_subplots


                

                
      n = 14  # Độ dài của RSI
      df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=n).rsi()

                # Tìm phân kì RSI
      rsi_bearish_divergence = (df['RSI'] > 80) & (df['RSI'].shift(1) > 80) & (df['close'] < df['close'].shift(1))
      rsi_bullish_divergence = (df['RSI'] < 20) & (df['RSI'].shift(1) < 20) & (df['close'] > df['close'].shift(1))

                # Tạo biểu đồ RSI sử dụng Plotly
      fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0)
                                
                # Biểu đồ RSI
      fig.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='blue')), row=2, col=1)

       
                # Cài đặt layout biểu đồ
      fig.update_layout(
                                  xaxis_rangeslider_visible=False,
                                  yaxis_title='Price / RSI')

                # Hiển thị biểu đồ
      st.header(stock_symbol.upper()+" RSI\n")
      st.plotly_chart(fig)

               
 elif chart.upper() == "4" :
      import pandas as pd
      import matplotlib.pyplot as plt
      import ta
      from plotly.subplots import make_subplots
      import numpy as np
                
               
                
                # Tính toán chỉ số MACD
      df['macd'] = ta.trend.macd(df['close'], 12, 26)
      df['signal'] = ta.trend.macd_signal(df['close'], 12, 26, 9)

                # Tạo cột histogram
      df['histogram'] = df['macd'] - df['signal']


                # Tạo biểu đồ MACD sử dụng Plotly
      fig = go.Figure()

                # Biểu đồ MACD
      fig.add_trace(go.Scatter(x=df.index, y=df['macd'], mode='lines', name='MACD', line=dict(color='blue')))

                # Đường signal
      fig.add_trace(go.Scatter(x=df.index, y=df['signal'], mode='lines', name='Signal', line=dict(color='red')))

                # Histogram
      colors = ['gray' if val < 0 else 'pink' for val in df['histogram']]
      fig.add_trace(go.Bar(x=df.index, y=df['histogram'], name='Histogram', marker_color=colors))

                

                # Cài đặt layout
      fig.update_layout(
                                  xaxis_title='Date',
                                  yaxis_title='MACD')

                # Hiển thị biểu đồ
      st.header(stock_symbol.upper()+" MACD\n")
      st.plotly_chart(fig)
 elif chart.upper() == "5" :
            
                # Tính toán Stochastic Oscillator
      def calculate_stochastic(df, n, k):
         df['%K'] = ((df['close'] - df['close'].rolling(window=n).min()) / (df['close'].rolling(window=n).max() - df['close'].rolling(window=n).min())) * 100
         df['%D'] = df['%K'].rolling(window=k).mean()

      n = 280  # Độ dài của %K
      k = 60   # Độ dài của %D

      calculate_stochastic(df, n, k)

                # Tạo biểu đồ Stochastic Oscillator
      fig = go.Figure()
      fig.add_trace(go.Scatter(x=df.index, y=df['%K'], mode='lines',line =dict(color='red'), name='%K'))
      fig.add_trace(go.Scatter(x=df.index, y=df['%D'], mode='lines',line =dict(color='green'), name='%D'))
      fig.update_layout(
                                  xaxis_title='Date',
                                  yaxis_title='Value',
                                  showlegend=True) 
      st.header(stock_symbol.upper()+" Stochastic Oscillator\n")
      st.plotly_chart(fig)
 else :
        st.write('Sorry! We do not have the chart you want to find in our database')
else: st.write('Sorry! We do not have the stock you want to find in our database')
