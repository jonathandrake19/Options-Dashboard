import streamlit as st
import Calculations as calc
import numpy as np
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from datetime import date, datetime, timedelta
from scipy.ndimage import median_filter
import yfinance as yf
import plotly.graph_objects as go

st.title('Implied Volatility Surface')

def remove_outliers(K, iv, threshold = 0.05):
    iv_median = median_filter(iv, size = 3, mode = 'nearest')
    residual = np.abs(iv - iv_median)

    std = np.std(residual)
    mask = residual < threshold * std

    return K[mask], iv[mask]

def fit_smile(K, iv, smoothing = 0.001):
    sorted_idx = np.argsort(K)
    K_sorted = K[sorted_idx]
    iv_sorted = iv[sorted_idx]
    if len(K_sorted) < 4:
        return None
    
    spline = UnivariateSpline(K_sorted, iv_sorted, s = smoothing)
    return spline

ticker = st.sidebar.text_input('Stock ticker:','')

if ticker:
    tk = yf.Ticker(ticker)
    expiry_dates = tk.options
    t = date.today()

    K_arrays = []
    vol_arrays =[]
    tau_list = []
    for expiry in expiry_dates:
        T = datetime.strptime(expiry, '%Y-%m-%d').date()
        tau = np.busday_count(t,T+timedelta(days=1))
        tau_list.append(tau)

        K, vol = calc.implied_vol_surface(ticker, 0, expiry, t, 'call')

        K_arrays.append(K)
        vol_arrays.append(vol)

    K_min = max(min(i) for i in K_arrays)
    K_max = min(max(j) for j in K_arrays)
    common_K = np.linspace(K_min, K_max, 200)

    z_matrix = np.empty(shape=(len(K_arrays), len(common_K)))

    for i, (T, k, vol) in enumerate(zip(tau_list, K_arrays, vol_arrays)):
        if len(k) < 5:
            z_matrix[i, :] = np.nan
            continue
        
        K_clean, vol_clean = remove_outliers(k, vol)
        spline = fit_smile(K_clean, vol_clean)
        if spline is not None:
            z_matrix[i, :] = spline(common_K)

        else:
            z_matrix[i, :] = np.nan

    valid_rows = ~np.isnan(z_matrix).any(axis=1)
    z_valid = z_matrix[valid_rows]
    tau_valid = np.array(tau_list)[valid_rows]

    if len(tau_valid) > 2:
        surface_spline = RectBivariateSpline(tau_valid, common_K, z_valid, s=1)
        z_smooth = surface_spline(tau_list, common_K)

    X, Y = np.meshgrid(common_K, tau_list)

    camera = dict(
        eye=dict(x=2, y=2, z=1.8),  
        up=dict(x=0, y=0, z=1)
        )  

    fig = go.Figure(data=[go.Surface(z=z_smooth, x=X, y=Y, colorscale='Viridis', showscale=True)])
    fig.update_layout(height = 600,
                      margin = dict(l=0, r=0, b=0, t=0),
                      scene_camera = camera,
                      scene=dict(
                          xaxis_title = 'Strike Price ($)',
                          yaxis_title = 'Time to Maturity (days)',
                          zaxis_title = 'Implied Volatility'
                        ))
    st.plotly_chart(fig, use_container_width=True)