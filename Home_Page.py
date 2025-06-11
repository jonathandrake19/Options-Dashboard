import streamlit as st

st.title('Option Data Dashboard')

st.write('Use the buttons in the side bar to navigate between the 3 pages of this dashboard.')

st.write('Option PnL calculator uses the Black Scholes equation to calculate the PnL of an option for a range of future states.'
' Input a stock ticker for current projections, or leave blank to explore custom scenarios.')

st.write('Implied volatility surface uses the Newton-Raphson method to calculate the implied volatility for each option strike '
'and expiry. Input a stock ticker to view its implied volatility surface')

st.write('All source code of this dashboard can be found at www.github.com/jonathandrake19')

