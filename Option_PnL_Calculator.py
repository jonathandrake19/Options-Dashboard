import streamlit as st
import numpy as np
import Data_Downloader as dd
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import Calculations as calc
import seaborn as sns
import yfinance as yf

#set up the page
st.title("Option PnL Calculator")

col1, col2 = st.columns(2)
r = 0.05
t = date.today()

st.sidebar.markdown('## Inputs ##')

#option type determines how the BSE is solved and what initial user inputs are required
ticker = st.sidebar.text_input('Stock ticker (leave blank for custom): ', '')
S_0 = 0
if ticker:
    tk = yf.Ticker(ticker)
    dates = tk.options
    if dates[0] == date.today():
        dates = dates[1:]
    expiry = st.sidebar.selectbox('Expiration date: ', dates)

    calls, puts = dd.option_data(ticker, expiry)

    K = st.sidebar.selectbox('Strike price: ', calls['strike'])
    V_0 = float(st.sidebar.text_input('Option purchase price: ', '0'))

    sigma_0 = calc.implied_vol(ticker, K, 0, expiry, 'call')
    S_0 = dd.current_price(ticker)
    st.write(f'Current stock price: ${round(S_0, 2)}')

elif not ticker:
    V_0 = float(st.sidebar.text_input('Option purchase price: ', '0')) # price option was purchased for, determines PnL
    K = float(st.sidebar.text_input('Strike price: ', '0'))
    today = datetime.today()
    max_date = today + timedelta(days=730)
    #expiry = st.sidebar.date_input('Expiration date: ', value='today', min_value=today, max_value=max_date)
    expiry = st.sidebar.number_input('Time to maturity', min_value=1, max_value=100, value=1, step=1)
    sigma_0 = st.sidebar.number_input('Volatility', min_value=0.05, max_value=1.00, 
                                value=0.3, step=0.05, format='%.2f')
    S_0 = 100

#once inputs are recieved determine whether European or US and solve BSE to generate a price matrix
if S_0 and K and expiry and sigma_0:
    if ticker:
        T = datetime.strptime(expiry, '%Y-%m-%d').date()
        tau_0 = np.busday_count(t,T+timedelta(days=1))
    else:
        tau_0 = expiry

    #recieve user input to change the limits of the graph axes
    st.sidebar.header("Variables")
    s_start = int(S_0) - 10
    s_end = int(S_0) + 10
    S_range = st.sidebar.slider('Range for underlying price', min_value=50, max_value=300, value=(s_start, s_end),
                                step=1, format='%d')
    S_vector = np.linspace(S_range[0], S_range[1], 10, endpoint='True')
    
    #this input alows the user to generate graphs for each time step towards expiry
    t_start = 0
    t_end = int(tau_0)
    t_range = st.sidebar.slider('Time to expiry: ', min_value=0, max_value=int(tau_0), value=(t_start, t_end),
                                    step=1, format='%d')
    t_vector = np.linspace(t_range[0], t_range[1], 10, endpoint='True')
    
    V_calls = np.zeros(shape=(len(S_vector), len(t_vector)))
    V_puts = np.zeros(shape=(len(S_vector), len(t_vector)))

    if ticker:
        for i, t_val in enumerate(t_vector):
            for j, S_val in enumerate(S_vector):
                V_calls[j][i] = max(S_val - (K * np.exp(-r*t_val)), calc.european_call(S_val, K, t_val, sigma_0/np.sqrt(252), r/252))
                V_puts[j][i] = max((K * np.exp(-r * t_val)) - S_val, calc.european_put(S_val, K, t_val, sigma_0/np.sqrt(252), r/252))

        V_call_current = calc.european_call(S_0, K, tau_0, sigma_0/np.sqrt(252), r/252)
        V_put_current = calc.european_put(S_0, K, tau_0, sigma_0/np.sqrt(252), r/252)

    else:    
        for i, t_val in enumerate(t_vector):
            for j, S_val in enumerate(S_vector):
                V_calls[j][i] = calc.european_call(S_val, K, t_val, sigma_0/np.sqrt(252), r/252)
                V_puts[j][i] = calc.european_put(S_val, K, t_val, sigma_0/np.sqrt(252), r/252)

    PnL_calls = V_calls - V_0
    PnL_puts = V_puts - V_0

    #scaling colourbar to ensure consistent colour (green for profit, red for loss)
    if abs(np.min(PnL_calls)) < np.max(PnL_calls):
        v_max = np.max(PnL_calls)
        v_min = v_max*-1
    else:
        v_min = np.min(PnL_calls)
        v_max = v_min*-1
        
    if abs(np.min(PnL_puts)) < np.max(PnL_puts):
        v_max_p = np.max(PnL_puts)
        v_min_p = v_max_p*-1
    else:
        v_min_p = np.min(PnL_puts)
        v_max_p = v_min_p*-1

    #plotting heatmaps
    fig_c, ax_c = plt.subplots(figsize=(8, 6))
    call_map = sns.heatmap(PnL_calls,
                        cmap='RdYlGn',
                        vmin= v_min,
                        vmax= v_max,
                        linewidths=0.5,
                        linecolor='k',
                        annot=True,
                        fmt = '.2f',
                        cbar=True,
                        cbar_kws={'label': 'PnL ($)'},
                        yticklabels= S_vector.round(2),
                        xticklabels= t_vector.round(2)
                        )
    ax_c.invert_yaxis()
    plt.xlabel('Time to maturity (days)')
    plt.ylabel('Stock price ($)')

    fig_p, ax_p = plt.subplots(figsize=(8, 6))
    put_map = sns.heatmap(PnL_puts,
                        cmap='RdYlGn',
                        vmin= v_min_p,
                        vmax= v_max_p,
                        linewidths=0.5,
                        linecolor='k',
                        annot=True,
                        fmt = '.2f',
                        cbar=True,
                        cbar_kws={'label': 'PnL ($)'},
                        yticklabels= S_vector.round(2),
                        xticklabels= t_vector.round(2)
                        )
    ax_p.invert_yaxis()
    plt.xlabel('Time to maturity (days)')
    plt.ylabel('Stock price ($)')
    
    with col1:
        st.markdown('### Call Option ###')
        if ticker:
            st.write(f'Current fair price: ${round(V_call_current, 2)}')
        st.pyplot(fig_c)
        
    with col2:
        st.markdown('### Put Option ###')
        if ticker:
            st.write(f'Current fair price: ${round(V_put_current, 2)}')
        st.pyplot(fig_p)
else:
    st.markdown('### Input your info ###')
